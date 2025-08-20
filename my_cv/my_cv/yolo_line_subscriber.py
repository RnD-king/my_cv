#!/usr/bin/env python3
import rclpy as rp
import numpy as np
import cv2
import math
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque, Counter
from robot_msgs.msg import LinePoint, LinePointsArray, LineResult # type: ignore
from rcl_interfaces.msg import SetParametersResult

class LineListenerNode(Node):
    def __init__(self):
        super().__init__('line_subscriber')
        
        # 공용 변수
        self.image_width = 640
        self.image_height = 480

        self.roi_x_start = self.image_width * 1 // 5 + 75
        self.roi_x_end   = self.image_width * 4 // 5 + 75
        self.roi_y_start = self.image_height * 1 // 12
        self.roi_y_end   = self.image_height * 11 // 12

        # zandi
        self.zandi_x = int((self.roi_x_start + self.roi_x_end) / 2)
        self.zandi_y = int(self.image_height - 100)

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = "AVG: --- ms | FPS: --"

        self.bridge = CvBridge()
        self.sub = self.create_subscription(  # 중심점 토픽
            LinePointsArray,                   
            'candidates',                   
            self.line_callback, 10)                             
        self.subscription_color = self.create_subscription(  # 이미지 토픽
            Image,
            '/camera/color/image_raw',  #  640x480 / 15fps
            self.color_image_callback, 10)
        
        # 파라미터 선언
        self.declare_parameter("limit_x", 50) # 핑크 라인
        self.declare_parameter("max_len", 10) # 최빈값
        self.declare_parameter("delta_s", 15) # 빨간 점이 얼마나 벗어나야 커브일까
        self.declare_parameter("vertical", 75) # 직진 판단 각도
        self.declare_parameter("horizontal", 15) # 수평 판단 각도

        # 파라미터 적용
        self.limit_x = self.get_parameter("limit_x").value
        self.max_len = self.get_parameter("max_len").value
        self.delta_s = self.get_parameter("delta_s").value
        self.vertical = self.get_parameter("vertical").value
        self.horizontal = self.get_parameter("horizontal").value
        
        self.candidates = [] 
        self.curve_count = 0
        self.tilt_text = "" # 화면 출력
        self.curve_text = "" # 화면 출력
        self.out_text = "" # 화면 출력
        self.recent_curve = deque(maxlen=self.max_len) # maxlen 프레임 중에서 최빈값
        self.recent_tilt = deque(maxlen=self.max_len)

        self.add_on_set_parameters_callback(self.parameter_callback)

        # 퍼블리셔
        self.line_result_pub = self.create_publisher(LineResult, '/line_result', 10)

    def parameter_callback(self, params):
        for param in params:
            if param.name == "limit_x":
                if param.value > 0 and param.value < 64:
                    self.limit_x = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "max_len":
                if param.value > 0:
                    self.max_len = param.value
                    self.recent_curve = deque(maxlen=self.max_len)
                    self.recent_tilt = deque(maxlen=self.max_len)
                else:
                    return SetParametersResult(successful=False)
            if param.name == "delta_s":
                if param.value > 0:
                    self.delta_s = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "vertical":
                if param.value > self.horizontal and param.value < 90:
                    self.vertical = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "horizontal":
                if param.value > 0 and param.value < self.vertical:
                    self.horizontal = param.value
                else:
                    return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)

    def line_callback(self, msg: LinePointsArray):  # 좌표 구독
        self.candidates = [(i.cx, i.cy, i.lost) for i in msg.points]
        for idx, (cx, cy, lost) in enumerate(self.candidates):
            self.get_logger().info(f'[{idx}] cx={cx}, cy={cy}, lost={lost}')

    def color_image_callback(self, msg): # 이미지 받아오기

        start_time = time.time()  # 시작 시간 기록

        # ROS 이미지 CV 이미지로 받아오기
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        roi = cv_image[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]

#---------------------------------------------------------------판단 시작  
        res = 0  # 모드
        angle = 0  # 회전각
        sum_x = 0  # 점선 평균 x 좌표
        # ───────── 1. 위치 판단 ─────────
        if self.candidates:
            for i in range(len(self.candidates)):
                sum_x += self.candidates[i][0]

            avg_x = float(sum_x / len(self.candidates))
            if avg_x - self.zandi_x > self.limit_x:
                self.out_text = "Out Left"
                # angle -= (avg_x - self.zandi_x) / 2   <<<<  얼마나 벗어났느냐에 따라 목표 회전값 보정   <<<  수식 생각해~
            elif avg_x - self.zandi_x <  -self.limit_x:
                self.out_text = "Out Right"
                # angle += (avg_x - self.zandi_x) / 2
            else:
                self.out_text = "Straight"
        
        # ───────── 2. 방향 판단 ─────────
        if len(self.candidates) >= 3:  #--------------------------------------------------------- 1.세 점 이상 탐지
            tip_x, tip_y, tip_lost = self.candidates[-1] # 가장 위의 점
            cv2.circle(roi, (tip_x - self.roi_x_start, tip_y - self.roi_y_start),
                       4, (0, 255, 255) if tip_lost > 0 else (0, 0, 255), -1)

            for acx, acy, ac_lost in self.candidates[:-1]: # 나머지 점들 시각화
                color = (0, 255, 255) if ac_lost > 0 else (255, 0, 0)
                cv2.circle(roi, (acx - self.roi_x_start, acy - self.roi_y_start),
                           3, color, 2 if ac_lost > 0 else -1)

            xs = np.array([c[0] for c in self.candidates[:-1]], dtype=np.float32)
            ys = np.array([c[1] for c in self.candidates[:-1]], dtype=np.float32)

            if float(xs.max() - xs.min()) < 2.0: # 1) 거의 수직
                x_center = int(round(float(np.median(xs))))
                x1 = x2 = x_center
                y1, y2 = self.roi_y_end, self.roi_y_start
                line_angle = 90.0
                delta = float(x_center - tip_x)
                if delta >  self.delta_s: 
                    self.curve_text = "Turn Left"
                    tip_angle = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                elif delta < -self.delta_s: 
                    self.curve_text = "Turn Right"
                    tip_angle = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                else: self.curve_text = "Straight"
                self.tilt_text = "Straight"

            else: # 2) 일반적인 경우
                pts = np.stack([xs, ys], axis=1)  # (N,2)
                vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                vx = float(vx); vy = float(vy); x0 = float(x0); y0 = float(y0)

                # 각도/기울기
                m = vy / vx
                line_angle = math.degrees(math.atan(m))

                # tip 좌표로부터 커브 판단
                signed = (float(tip_x) - x0) * vy - (float(tip_y) - y0) * vx
                delta = signed / (math.hypot(vx, vy) + 1e-5)
                if delta >  self.delta_s: 
                    self.curve_text = "Turn Left"
                    tip_angle = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                elif delta < -self.delta_s: 
                    self.curve_text = "Turn Right"
                    tip_angle = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                else: self.curve_text = "Straight"

                # 직선
                b = y0 - m * x0
                if abs(m) > 1:
                    y1, y2 = self.roi_y_end, self.roi_y_start
                    x1 = int((y1 - b) / m)
                    x2 = int((y2 - b) / m)
                else:
                    x1, x2 = self.roi_x_end, self.roi_x_start
                    y1 = int(m * x1 + b)
                    y2 = int(m * x2 + b)

                # 기울기 판단
                if self.horizontal < line_angle < self.vertical:
                    self.tilt_text = "Spin Left"
                elif -self.horizontal > line_angle > -self.vertical:
                    self.tilt_text = "Spin Right"
                else:
                    self.tilt_text = "Straight"

            # 시각화
            cv2.line(roi,
                     (int(x1 - self.roi_x_start), int(y1 - self.roi_y_start)),
                     (int(x2 - self.roi_x_start), int(y2 - self.roi_y_start)),
                     (255, 0, 0), 2)
        
        elif len(self.candidates) == 2:  #------------------------------------------------------------ 2. 두 점 탐지
            # 단순 선
            x1, y1, down_lost = self.candidates[0]
            x2, y2, up_lost   = self.candidates[1]
            cv2.circle(roi, (x1 - self.roi_x_start, y1 - self.roi_y_start),
                       3, (0,255,255) if down_lost > 0 else (0,0,255), 2 if down_lost > 0 else -1)
            cv2.circle(roi, (x2 - self.roi_x_start, y2 - self.roi_y_start),
                       3, (0,255,255) if up_lost > 0 else (255,0,0), 2 if up_lost > 0 else -1)

            if abs(x1 - x2) < 2:  # 1) 기울기 거의 수직
                line_angle = 90.0
                self.tilt_text = "Straight"
                x = int(round((x1 + x2) * 0.5))
                x1 = x2 = x
                y1, y2 = self.roi_y_end, self.roi_y_start
            else:  # 2) 일반적
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                line_angle = math.degrees(math.atan(m))
                if abs(m) > 1:
                    y1, y2 = self.roi_y_end, self.roi_y_start
                    x1 = int((y1 - b) / m)
                    x2 = int((y2 - b) / m)
                else:
                    x1, x2 = self.roi_x_end, self.roi_x_start
                    y1 = int(m * x1 + b)
                    y2 = int(m * x2 + b)
                if self.horizontal < line_angle < self.vertical:
                    self.tilt_text = "Spin Left"
                elif -self.horizontal > line_angle > -self.vertical:
                    self.tilt_text = "Spin Right"
                else:
                    self.tilt_text = "Straight"
            self.curve_text = "Straight"
            cv2.line(roi, (int(x1 - self.roi_x_start), int(y1 - self.roi_y_start)), (int(x2 - self.roi_x_start), int(y2 - self.roi_y_start)),(255, 0, 0), 2)

        else:  #------------------------------------------------------------------------------ 3. 점 1개 or 탐지 실패
            self.curve_text = "Miss"
            self.tilt_text = "Miss"
            self.out_text = "Miss"
            line_angle = 0.0

        self.recent_curve.append(self.curve_text)
        self.recent_tilt.append(self.tilt_text)
        stable_curve = Counter(self.recent_curve).most_common(1)[0][0] if self.recent_curve else "Miss"  # 최빈값에 맞게 커브 판단  >> 후에 이거로 상태함수 넣어도 됨 / fsm
        stable_tilt = Counter(self.recent_tilt).most_common(1)[0][0] if self.recent_tilt else "Miss"

        #-------------------------------------------------------------------------------------------------------------- 판단 끝
        
        if stable_curve == "Straight" and stable_tilt == "Straight" and self.out_text == "Straight":
            res = 1 # 직진
            angle = 90 # 직선 각도
        elif  stable_tilt == "Spin Right" or self.out_text == "Out Left":
            res = 2 # 우회전 해라
            angle += line_angle
        elif  stable_tilt == "Spin Left" or self.out_text == "Out Right":
            res = 3 # 좌회전 해라
            angle += line_angle
        elif stable_curve == "Turn Left" or stable_curve == "Turn Right":
            res = 4
            angle = tip_angle # 끝점만 보고 가기. 나중에 끝점까지의 거리 같은 거 넣을까
        else: # 탐지 실패
            res = 99

        # 여기에 퍼블리시
        
        line_msg = LineResult()
        line_msg.res = res
        line_msg.angle = abs(int(angle))
        self.line_result_pub.publish(line_msg)

        #------------------------------------------------------------------------------------------------------------  출력

        # 판단 텍스트 출력
        cv2.putText(cv_image, f"Rotate: {line_angle:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Tilt: {stable_tilt}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Curve: {stable_curve}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Out: {self.out_text}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        cv_image[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end] = roi  # 전체화면
        cv2.rectangle(cv_image, (self.roi_x_start - 1, self.roi_y_start - 1), (self.roi_x_end + 1, self.roi_y_end), (0, 255, 0), 1) # ROI 구역 표시
        cv2.line(cv_image, (int(self.image_width / 2) - self.limit_x, self.roi_y_start), (int(self.image_width / 2) - self.limit_x, self.roi_y_end), (255, 22, 255), 1)
        cv2.line(cv_image, (int(self.image_width / 2) + self.limit_x, self.roi_y_start), (int(self.image_width / 2) + self.limit_x, self.roi_y_end), (255, 22, 255), 1)
        cv2.circle(cv_image, (self.zandi_x, self.zandi_y), 5, (255, 255, 255), -1)
        #-------------------------------------------------------------------------------------------------- 프레임 처리 시간 측정
        
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        now = time.time()

        # 1초에 한 번 평균 계산
        if now - self.last_report_time >= 1.0:
            avg_time = self.total_time / self.frame_count
            avg_fps = self.frame_count / (now - self.last_report_time)
            
            # 텍스트 준비
            self.last_avg_text = f"PING: {avg_time*1000:.2f}ms | FPS: {avg_fps:.2f}"
            
            # 타이머 리셋
            self.frame_count = 0
            self.total_time = 0.0
            self.last_report_time = now

        cv2.putText(cv_image, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        #---------------------------------------------------------------------------------------------------- 결과

        cv2.imshow("Subscriber", cv_image)  # 결과
        cv2.waitKey(1)

def main():
    rp.init()
    node = LineListenerNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()

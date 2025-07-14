import rclpy as rp
import numpy as np
import cv2
import math
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque, Counter
from my_cv_msgs.msg import LinePoint, LinePointsArray # type: ignore
from rcl_interfaces.msg import SetParametersResult

class LineListenerNode(Node):
    def __init__(self):
        super().__init__('line_subscriber')
        self.sub = self.create_subscription(  # 중심점 토픽
            LinePointsArray,                   
            'candidates',                   
            self.line_callback,             
            10)                             
        self.subscription_color = self.create_subscription(  # 이미지 토픽
            Image,
            '/camera/camera/color/image_raw',  #  640x480 / 15fps
            self.color_image_callback, 10)
        
        self.bridge = CvBridge()

        # 파라미터 선언
        self.declare_parameter("max_len", 30)
        self.declare_parameter("delta_s", 15)
        self.declare_parameter("vertical", 75)
        self.declare_parameter("horizonal", 15)

        # 파라미터 적용
        self.max_len = self.get_parameter("max_len").value
        self.delta_s = self.get_parameter("delta_s").value
        self.vertical = self.get_parameter("vertical").value
        self.horizonal = self.get_parameter("horizonal").value
        
        self.candidates = [] 
        self.curve_count = 0
        self.tilt_text = "" # 화면 출력
        self.curve_text = "" # 화면 출력
        self.out_text = "" # 화면 출력
        self.recent_curve = deque(maxlen=self.max_len) # maxlen 프레임 중에서 최빈값
        self.recent_tilt = deque(maxlen=self.max_len)

        self.add_on_set_parameters_callback(self.parameter_callback)

        # 시간 디버깅
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = "AVG: --- ms | FPS: --"

    def parameter_callback(self, params):
        for param in params:
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
                if param.value > self.horizonal and param.value < 90:
                    self.vertical = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "horizonal":
                if param.value > 0 and param.value < self.vertical:
                    self.horizonal = param.value
                else:
                    return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)


    def get_angle(self, c1, c2): # 단순 각도 계산
        dx = c2[0]-c1[0]
        dy = c2[1]-c1[1]
        angle = 180 / math.pi * math.atan2(dy, dx) + 90
        if angle > 89:  # 계산 과부하 방지
            angle = 89
        elif angle < -89:
            angle = -89
        return round(angle, 2)

    def line_callback(self, msg: LinePointsArray):  # 좌표 구독
        self.candidates = [(i.cx, i.cy, i.lost) for i in msg.points]
        for idx, (cx, cy, lost) in enumerate(self.candidates):
            self.get_logger().info(f'[{idx}] cx={cx}, cy={cy}, lost={lost}')

    def color_image_callback(self, msg): # 이미지 받아오기

        start_time = time.time()  # 시작 시간 기록

        # ROS 이미지 CV 이미지로 받아오기
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        screen_h, screen_w, _ = cv_image.shape

        # ROI 좌표 정의, 컷
        roi_y_start = screen_h * 1 // 3  # 위
        roi_y_end = screen_h // 1        # 아래
        roi_x_start = screen_w * 2 // 5  # 왼 
        roi_x_end = screen_w * 3 // 5    # 오

        roi = cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

#---------------------------------------------------------------라인 판단 시작  << 최소자승 직선 근사  << 얘 좀 이상함

        sum = 0

        if self.candidates:
            for i in range(len(self.candidates)):
                sum += self.candidates[i][0]

            avg_x = float(sum / len(self.candidates))
            if avg_x > (roi_x_end - roi_x_start) / 2 + 10:
                self.out_text = "Out Left"
            elif avg_x < (roi_x_end - roi_x_start) / 2 - 10:
                self.out_text = "Out Right"
            else:
                self.out_text = "Straight"
            
        if len(self.candidates) >= 3:  # 1. 3개 이상일 때: 맨 위 점 제외하고 직선 근사        
            tip_x, tip_y, tip_lost = self.candidates[-1] # 그 중 가장 위의 점 = 새로 탐지된 점
            if tip_lost > 0:
                cv2.circle(roi, (tip_x, tip_y), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
            else:
                cv2.circle(roi, (tip_x, tip_y), 4, (0, 0, 255), -1) # 가장 위의 새로운 점은 빨강
            
            for acx, acy, ac_lost in self.candidates[:-1]:
                if ac_lost > 0:
                    cv2.circle(roi, (acx, acy), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
                else:
                    cv2.circle(roi, (acx, acy), 4, (255, 0, 0), -1) # 나머지 파랑

            xs = np.array([c[0] for c in self.candidates[:-1]])
            ys = np.array([c[1] for c in self.candidates[:-1]])

            if abs(xs.max() - xs.min()) < 2:  # 점들이 수직에 가까운 경우
                x1, y1, _ = self.candidates[0]
                x2, y2, _ = self.candidates[-2]
                y1 = roi_y_end - roi_y_start
                y2 = 0
                line_angle = 90
                delta = (xs.max() + xs.min()) / 2 - tip_x
                # 커브 판단
                if delta < -1 * self.delta_s:
                    self.curve_text = "Turn Right"
                elif delta > self.delta_s:
                    self.curve_text = "Turn Left"
                else:
                    self.curve_text = "Straight"
                self.tilt_text = "Straight" # 기울기 판단 > 수직이니까 항상 직진

            else: # 일반적인 경우
                m, b = np.polyfit(xs, ys, 1) # 기울기와 y절편
                line_angle = math.degrees(math.atan2(m, 1))
                numerator = m * tip_x - tip_y + b # 점과 직선 사이의 수직 거리
                denominator = math.sqrt(m**2 + 1)
                delta = numerator / denominator
                # 직선 시각화 방식 선택 (기울기 크기에 따라)
                if abs(m) > 1:  # 기울기 크면 y 기준 (거의 수직)
                    y1 = roi_y_end - roi_y_start
                    y2 = 0
                    x1 = int((y1 - b) / m)
                    x2 = int((y2 - b) / m)
                else:  # 거의 수평일 땐 x 기준
                    x1 = roi_x_end - roi_x_start
                    x2 = 0
                    y1 = int(m * (x1) + b)
                    y2 = int(m * (x2) + b)
                # 커브 판단
                if delta < -1 * self.delta_s:
                    if m > 0:
                        self.curve_text = "Turn Left"
                    elif m < 0:
                        self.curve_text = "Turn Right"
                    else:
                        self.curve_text = "Horizon" # 예) 농구공 집고 라인을 찾는데 라인이 수평선인 경우 -> 별도의 알고리즘 필요 - 추후에,,,
                elif delta > self.delta_s:
                    if m > 0:
                        self.curve_text = "Turn Right"
                    elif m < 0:
                        self.curve_text = "Turn Left"
                    else:
                        self.curve_text = "Horizon" ###
                else:
                    self.curve_text = "Straight"
                # 기울기 판단
                if self.horizonal < line_angle < self.vertical:
                    self.tilt_text = "Spin Left"
                elif -self.horizonal > line_angle > -self.vertical:
                    self.tilt_text = "Spin Right"
                else: # 평행선 아직 처리 안 함(0~10 사이)
                    self.tilt_text = "Straight"
            cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)

        elif len(self.candidates) == 2:  # 2. 2개일 때: 걍 두 개 잇기 >>> 여기부턴 그럴 일 거의 없다고 봄
            x1, y1, down_lost = self.candidates[0] # 아래
            x2, y2, up_lost = self.candidates[1] # 위
            if down_lost > 0:
                cv2.circle(roi, (x1, y1), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
            else:
                cv2.circle(roi, (x1, y1), 4, (0, 0, 255), -1) #  파랑
            if up_lost > 0:
                cv2.circle(roi, (x2, y2), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
            else:
                cv2.circle(roi, (x2, y2), 4, (255, 0, 0), -1) # 빨강

            if abs(x1 - x2) < 2: # 점들이 수직에 가까운 경우
                line_angle = 90
                self.tilt_text = "Straight"
                y1 = roi_y_end - roi_y_start
                y2 = 0
            else:   # 일반적인 경우
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                line_angle = math.degrees(math.atan2(m, 1))
                if abs(m) > 1:  # 기울기 크면 y 기준 (거의 수직)
                    y1 = roi_y_end - roi_y_start
                    y2 = 0 
                    x1 = int((y1 - b) / m)
                    x2 = int((y2 - b) / m)
                else:  # 거의 수평일 땐 x 기준
                    x1 = roi_x_end - roi_x_start
                    x2 = 0
                    y1 = int(m * x1 + b)
                    y2 = int(m * x2 + b)
                
                if self.horizonal < line_angle < self.vertical:
                    self.tilt_text = "Spin Left"
                elif -self.horizonal > line_angle > -self.vertical:
                    self.tilt_text = "Spin Right"
                else: # 평행성 아직 처리 안 함
                    self.tilt_text = "Straight"
            self.curve_text = "Straight" # 커브 판단 불가
            cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 2)

        else: # 감지 실패
            self.curve_text = "Miss"
            self.tilt_text = "Miss"
            line_angle = 0

        self.recent_curve.append(self.curve_text)
        self.recent_tilt.append(self.tilt_text)
        stable_curve = Counter(self.recent_curve).most_common(1)[0][0] if self.recent_curve else "Miss"  # 최빈값에 맞게 커브 판단  >> 후에 이거로 상태함수 넣어도 됨 / fsm
        stable_tilt = Counter(self.recent_tilt).most_common(1)[0][0] if self.recent_tilt else "Miss"

        if stable_curve == "Straight" and stable_tilt == "Straight" and self.out_text == "Straight":
            res = 1 # 직진
        elif  stable_tilt == "Spin Right" or self.out_text == "Out Left":
            res = 2 # 우회전 해라
            angle = line_angle
        elif  stable_tilt == "Spin Left" or self.out_text == "Out Right":
            res = 3 # 우회전 해라
            angle = line_angle
        elif stable_curve == "Turn Left" and stable_curve == "Turn Right":
            res = 4 # 징검다리 가라

        #---------------------------------------------------------------------------------------------------------------라인 판단 끝

        # 방향 텍스트 출력
        cv2.putText(cv_image, f"Rotate: {line_angle:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Tilt: {stable_tilt}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Curve: {stable_curve}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(cv_image, f"Out: {self.out_text}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = roi  # 전체화면
        cv2.rectangle(cv_image, (roi_x_start - 1, roi_y_start - 1), (roi_x_end + 1, roi_y_end), (0, 255, 0), 1) # ROI 구역 표시
        
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

        # 평균 처리 시간 텍스트 영상에 출력
        if hasattr(self, "last_avg_text"):
            cv2.putText(cv_image, self.last_avg_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
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

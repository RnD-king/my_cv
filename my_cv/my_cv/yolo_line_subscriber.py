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
from my_cv_msgs.msg import LinePoint, LinePointsArray, LineResult, MotionEnd # type: ignore
from rcl_interfaces.msg import SetParametersResult

class LineListenerNode(Node):
    def __init__(self):
        super().__init__('line_subscriber')
        
        # 공용 변수
        self.image_width = 640
        self.image_height = 480

        self.roi_x_start = int(self.image_width * 1 // 5)  # 초록 박스 관심 구역
        self.roi_x_end   = int(self.image_width * 4 // 5)
        self.roi_y_start = int(self.image_height * 1 // 12)
        self.roi_y_end   = int(self.image_height * 11 // 12)

        # zandi
        self.zandi_x = int((self.roi_x_start + self.roi_x_end) / 2)
        self.zandi_y = int(self.image_height - 100)

        # 타이머
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = "AVG: --- ms | FPS: --"

        self.line_start_time = None        # 윈도우 시작 시각 (wall-clock)

        self.frames_left = 0       # 남은 프레임 수 < max_len
        
        self.collecting = False     # 수집 중 여부
        self.last_motion_state = False

        self.status_list = [] # 누적 값 저장
        self.angle_list = []

        self.bridge = CvBridge()
        self.sub = self.create_subscription(  # 중심점 토픽
            LinePointsArray,                   
            'candidates',                   
            self.line_callback, 10)      
        self.motion_end = self.create_subscription(  # 모션 끝나면 받아올 T/F
            MotionEnd,                   
            'end',                   
            self.motion_callback, 10)                         
        self.subscription_color = self.create_subscription(  # 이미지 토픽
            Image,
            '/camera/color/image_raw',  #  640x480 / 15fps
            self.color_image_callback, 10)
        
        # 파라미터 선언
        self.declare_parameter("limit_x", 100) # 핑크 라인 두께
        self.declare_parameter("max_len", 15) # 몇 프레임 동안 보고 판단할 거냐
        self.declare_parameter("delta_s", 15) # 빨간 점이 얼마나 벗어나야 커브일까
        self.declare_parameter("vertical", 15) # 직진 판단 각도
        self.declare_parameter("horizontal", 75) # 수평 판단 각도  <<<  아직 안 만듬

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

        self.add_on_set_parameters_callback(self.parameter_callback)

        # 퍼블리셔
        self.line_result_pub = self.create_publisher(LineResult, '/line_result', 10)

    def parameter_callback(self, params):
        for param in params:
            if param.name == "limit_x":
                if param.value > 0 and param.value < 320:
                    self.limit_x = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "max_len":
                if param.value > 0:
                    self.max_len = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "delta_s":
                if param.value > 0:
                    self.delta_s = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "vertical":
                if 0 < param.value and param.value < self.horizontal:
                    self.vertical = param.value
                else:
                    return SetParametersResult(successful=False)
            if param.name == "horizontal":
                if param.value < 90 and param.value > self.vertical:
                    self.horizontal = param.value
                else:
                    return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)

    def line_callback(self, msg: LinePointsArray):  # 좌표 구독
        self.candidates = [(i.cx, i.cy, i.lost) for i in msg.points]
        #for idx, (cx, cy, lost) in enumerate(self.candidates):
        #    self.get_logger().info(f'[{idx}] cx={cx}, cy={cy}, lost={lost}') 잘 받아오고 있나 확인용

    def motion_callback(self, msg: MotionEnd): # 모션 끝났다 신호 받으면
        motion_end = bool(msg.end)
        rising = (not self.last_motion_state) and motion_end # 직전 값이 0, 지금 값이 1일 때 1 (False -> True)
        self.last_motion_state = motion_end # 직전 값 갱신
        if rising and not self.collecting:
            self.collecting = True  # 탐지 시작해라
            self.frames_left = self.max_len  # 프레임 초기화
            
            self.out_text = ""
            self.curve_text = ""
            self.tilt_text = ""

            self.status_list.clear()
            self.angle_list.clear()

            self.line_start_time = time.time()
            
            self.get_logger().info(f'[Line] Start collecting {self.max_len} frames')

    def color_image_callback(self, msg): # 이미지 받아오기
        
        line_angle = None # 직선 각도 (맨 위의 점 제외)
        angle = None # 잔디가 회전해야할 각도
        start_time = time.time()

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        roi = cv_image[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]

        if self.collecting: # 모션이 끝났다고 신호 받으면 시작
            # 1. 위치 판단
            if len(self.candidates) >= 3:  #========================================================================================= I. 세 점 이상 탐지
                # 정렬)
                tip_x, tip_y, tip_lost = self.candidates[-1] # 맨위
                cv2.circle(roi, (tip_x - self.roi_x_start, tip_y - self.roi_y_start), 4, (0,255,255) if tip_lost > 0 else (0,0,255), -1)

                for acx, acy, ac_lost in self.candidates[:-1]: # 나머지 밑에
                    color = (0, 255, 255) if ac_lost > 0 else (255, 0, 0)
                    cv2.circle(roi, (acx - self.roi_x_start, acy - self.roi_y_start), 3, color, 2 if ac_lost > 0 else -1)

                line_x = np.array([c[0] for c in self.candidates[:-1]],dtype = np.float32)
                line_y = np.array([c[1] for c in self.candidates[:-1]],dtype = np.float32)
                avg_line_x = int(round(np.mean(line_x)))  # 1. out

                if float(line_x.max() - line_x.min()) < 2.0:  # 일직선인 경우
                    x1 = x2 = avg_line_x
                    y1, y2 = self.roi_y_end, self.roi_y_start # x1, x2, y1, y2는 모두 직선 시각화
                    line_angle = 0  # 2. tilt
                    delta = float(avg_line_x - tip_x)  # 3. curve

                else: # 일반적인 경우
                    pts = np.stack([line_x, line_y], axis = 1)  # (N,2)
                    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                    vx = float(vx); vy = float(vy); x0 = float(x0); y0 = float(y0)
                    # 각도/기울기
                    m = vy / (vx if abs(vx) > 1e-9 else 1e-9)
                    line_angle = math.degrees(math.atan2(vy, vx)) - 90  # 2. tilt
                    # tip 좌표로부터 커브 판단
                    signed = (float(tip_x) - x0) * vy - (float(tip_y) - y0) * vx
                    delta = signed / (math.hypot(vx, vy) + 1e-5)  # 3. curve

                    b = y0 - m * x0
                    if abs(m) > 1:
                        y1, y2 = self.roi_y_end, self.roi_y_start
                        x1 = int((y1 - b) / m)
                        x2 = int((y2 - b) / m)
                    else:
                        x1, x2 = self.roi_x_end, self.roi_x_start
                        y1 = int(m * x1 + b)
                        y2 = int(m * x2 + b)
                #-------------------------------------------------------------------------------------------------------#
                if abs(delta) > self.delta_s:  # 분기 1-1. turn  =  RL 
                    angle = math.degrees(math.atan2(-(tip_y - self.zandi_y), tip_x - self.zandi_x)) - 90  # tip_angle
                    status = 3 # turn
                    if delta > 0:
                        self.curve_text = "Turn Left"
                    else:
                        self.curve_text = "Turn Right"

                    if abs(avg_line_x-self.zandi_x) < self.limit_x:  # 분기 2-1. out  =  Straight
                        self.out_text = "Straight"

                        if abs(line_angle) < self.vertical:  # 분기 3-1. tilt = Straight
                            self.tilt_text = "Straight"
                        elif self.horizontal > line_angle > self.vertical:  # 분기 3-2. tilt = Left
                            self.tilt_text = "Spin Left"
                        elif -self.horizontal < line_angle < -self.vertical:  # 분기 3-3. tilt = Right
                            self.tilt_text = "Spin Right"
                    else:  # 분기 2-2. out  =  RL
                        if avg_line_x - self.zandi_x > 0:
                            self.out_text = "Out Left"
                        else:
                            self.out_text = "Out Right"
                        
                        if abs(line_angle) < self.vertical: 
                            self.tilt_text = "Straight"
                        elif self.horizontal > line_angle > self.vertical:
                            self.tilt_text = "Spin Left"
                        elif -self.horizontal < line_angle < -self.vertical:
                            self.tilt_text = "Spin Right"

                else:  # 분기 1-2. turn  =  Straight
                    self.curve_text = "Straight"
                    if abs(avg_line_x - self.zandi_x) < self.limit_x:  # 분기 2-1. out  =  Straight
                        self.out_text = "Straight"
                        if abs(line_angle) < self.vertical:  # 분기 3. tilt  =  Straight
                            angle = 0
                            self.tilt_text = "Straight"
                            status = 1 # straight
                        elif self.horizontal > line_angle > self.vertical:
                            angle = line_angle
                            status = 2 # spin
                            self.tilt_text = "Spin Left"
                        elif -self.horizontal < line_angle < -self.vertical:
                            angle = line_angle
                            status = 2 # spin
                            self.tilt_text = "Spin Right"
                        # else: << 수평선
                    else:  # 분기 2-2. out  =  RL
                        if avg_line_x - self.zandi_x > 0:
                            self.out_text = "Out Left"
                        else:
                            self.out_text = "Out Right"
                        angle = math.degrees(math.asin(self.zandi_x - avg_line_x) / 60.0)  # 60은 한번 걸을 때 픽셀 수 대충 넣자
                        status = 2 # spin
                        
                        if abs(line_angle) < self.vertical:  # 분기 3. tilt  =  Straight
                            self.tilt_text = "Straight"
                        elif self.horizontal > line_angle > self.vertical:
                            angle += line_angle
                            self.tilt_text = "Spin Left"
                        elif -self.horizontal < line_angle < -self.vertical:
                            angle += line_angle
                            self.tilt_text = "Spin Right"

                # 시각화
                cv2.line(roi, (int(x1 - self.roi_x_start), int(y1 - self.roi_y_start)), (int(x2 - self.roi_x_start), int(y2 - self.roi_y_start)), (255, 0, 0), 2)

            elif len(self.candidates) == 2:  #=============================================================================================== II. 두 점 탐지
                # 정렬
                self.curve_text = "Straight" # 커브길 판단 X
                down_x, down_y, down_lost = self.candidates[0] # 아래
                up_x, up_y, up_lost = self.candidates[1] # 위
                cv2.circle(roi, (down_x - self.roi_x_start, down_y - self.roi_y_start), 3, (0, 255, 255) if down_lost > 0 else (0, 0, 255), 2 if down_lost > 0 else -1)
                cv2.circle(roi, (up_x - self.roi_x_start, up_y - self.roi_y_start), 3, (0, 255, 255) if up_lost > 0 else (255, 0, 0), 2 if up_lost > 0 else -1)

                avg_x = int(round((down_x + up_x) / 2)) # 1. out

                if abs(down_x - up_x) < 2:  # 일직선인 경우
                    line_angle = 0 # 2. tilt
                    x1 = x2 = avg_x
                    y1, y2 = self.roi_y_end, self.roi_y_start
                else:  # 일반적인 경우
                    m = (up_y - down_y) / ((up_x - down_x) if abs(up_x - down_x) > 1e-9 else 1e-9)
                    b = down_y - m * down_x
                    line_angle = math.degrees(math.atan2(up_y - down_y, up_x - down_x)) - 90 # 2. tilt
                    if abs(m) > 1:
                        y1, y2 = self.roi_y_end, self.roi_y_start
                        x1 = int((y1 - b) / m)
                        x2 = int((y2 - b) / m)
                    else:
                        x1, x2 = self.roi_x_end, self.roi_x_start
                        y1 = int(m * x1 + b)
                        y2 = int(m * x2 + b)
                #----------------------------------------------------------------------------------------------------------#
                if abs(avg_x - self.zandi_x) < self.limit_x:  # 분기 1-1. out  =  Straight
                    self.out_text = "Straight"
                    if abs(line_angle) < self.vertical:  # 분기 2-1. tilt  =  Straight
                        angle = 0
                        self.tilt_text = "Straight"
                        status = 1 # straight
                    elif self.horizontal > line_angle > self.vertical:  # 분기 2-2. tilt  =  Left
                        angle = line_angle
                        status = 2 # spin
                        self.tilt_text = "Spin Left"
                    elif -self.horizontal < line_angle < -self.vertical:  # 분기 2-3. tilt  =  Right
                        angle = line_angle
                        status = 2 # spin
                        self.tilt_text = "Spin Right"
                    # else: << 수평선

                else:  # 분기 1-2. out  =  RL
                    if avg_x-self.zandi_x > 0:
                        self.out_text = "Out Left"
                    else:
                        self.out_text = "Out Right"
                    r = float(np.clip((avg_x - self.zandi_x) / 60.0, -1.0, 1.0))
                    angle = math.degrees(math.asin(r))  # 60은 한번 걸을 때 픽셀 수 대충 넣자
                    status = 2 # spin
                    
                    if abs(line_angle) < self.vertical:  # 분기 2. tilt  =  Straight
                        self.tilt_text = "Straight"
                    elif self.horizontal > line_angle > self.vertical:
                        angle += line_angle
                        self.tilt_text = "Spin Left"
                    elif -self.horizontal < line_angle < -self.vertical:
                        angle += line_angle
                        self.tilt_text = "Spin Right"

                cv2.line(roi, (int(x1 - self.roi_x_start), int(y1 - self.roi_y_start)), (int(x2 - self.roi_x_start), int(y2 - self.roi_y_start)), (255, 0, 0), 2)

            else:  #================================================================================================================== 3. 점 1개 or 탐지 실패
                self.curve_text = "Miss"
                self.tilt_text = "Miss"
                self.out_text = "Miss"
                angle = 999
                status = 4 # retry
            #=======================================================================================================================================================

            self.frames_left -= 1
            self.status_list.append(status) # 결과 저장
            self.angle_list.append(angle)

            if self.frames_left <= 0: # 15 프레임 지났으면
                cnt=Counter(self.status_list)
                mode_status=max(cnt.items(), key=lambda kv:kv[1])[0]  # 가장 많이 나온 status와
                angles=[a for s,a in zip(self.status_list,self.angle_list) if s==mode_status]
                mean_angle=float(np.mean(angles))  # 그 status에서의 각도 평균
                if abs(mean_angle) < 15: # 만약 회전해야할 각도가 너무 작으면 (out이랑 tilt가 부호만 반대로 거의 같을 때)
                    res = 1 # 그냥 직진 하셈

                process_time = (time.time() - self.line_start_time) / self.max_len if self.line_start_time is not None else 0.0

                if mode_status != 4:
                    if mode_status == 1:  # 직진 SSS (SLR SRL)
                        res = 1
                    elif mode_status == 2 and mean_angle > 0: # 왼쪽으로 회전해라 SLS SSL SLL (SLR SRL)
                        res = 2
                    elif mode_status == 2 and mean_angle < 0: # 오른쪽으로 회전해라 SRS SSR SRR (SLR SRL)
                        res = 3
                    elif mode_status == 3 and mean_angle > 0: # 왼쪽 커브길이다 Lxx
                        res = 4
                    elif mode_status == 3 and mean_angle < 0: # 오른쪽 커브길이다 Rxx
                        res = 5
                    self.get_logger().info(f"[Line] Window done: res={res}, angle_avg={mean_angle:.2f}, frames={len(self.status_list)}, "
                                           f"wall={process_time*1000:.1f} ms")
                else:
                    res = 99
                    self.get_logger().info(f"[Line] Window done: res={res}, angle_avg=None, frames={len(self.status_list)}, "
                                           f"wall={process_time*1000:.1f} ms")

                # 퍼블리시
                msg_out = LineResult()
                msg_out.res = res
                msg_out.angle = mean_angle
                self.line_result_pub.publish(msg_out)
                
                self.collecting = False # 초기화
                self.frames_left = 0
            
            #------------------------------------------------------------------------------------------------------------  출력

            cv2.putText(cv_image, f"Rotate: {angle}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Tilt: {self.tilt_text}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Curve: {self.curve_text}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Out: {self.out_text}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

        else: # 모션 중,,,
            cv2.putText(cv_image, "Idle", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            

        cv_image[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end] = roi  # 전체화면
        cv2.rectangle(cv_image, (self.roi_x_start - 1, self.roi_y_start - 1), (self.roi_x_end + 1, self.roi_y_end), (0, 255, 0), 1) # ROI 구역 표시
        cv2.line(cv_image, (self.zandi_x - self.limit_x, self.roi_y_start), (self.zandi_x - self.limit_x, self.roi_y_end), (255, 22, 255), 1)
        cv2.line(cv_image, (self.zandi_x + self.limit_x, self.roi_y_start), (self.zandi_x + self.limit_x, self.roi_y_end), (255, 22, 255), 1)
        cv2.circle(cv_image, (self.zandi_x, self.zandi_y), 5, (255, 255, 255), -1)
        #-------------------------------------------------------------------------------------------------- 프레임 처리 시간 측정

        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        
        if time.time() - self.last_report_time >= 1.0:  # 1초마다 평균 계산
            avg_time = self.total_time / self.frame_count
            avg_fps = self.frame_count / (time.time() - self.last_report_time)
            self.last_avg_text = f"PING: {avg_time*1000:.2f}ms | FPS: {avg_fps:.2f}"
            
            self.frame_count = 0  # 초기화
            self.total_time = 0.0
            self.last_report_time = time.time()

        cv2.putText(cv_image, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Line", cv_image)  # 이미지
        cv2.waitKey(1)

def main():
    rp.init()
    node = LineListenerNode()
    rp.spin(node)
    node.destroy_node()
    rp.shutdown()

if __name__ == '__main__':
    main()

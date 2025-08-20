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

        self.collect_t0 = None        # 윈도우 시작 시각 (wall-clock)
        self.collect_proc_sum = 0.0   # 프레임 처리 시간 누적 (compute-only)
        self.collect_frame_cnt = 0    # 수집한 프레임 수 (디버그용)

        self.last_final_curve = "-"
        self.last_final_tilt  = "-"
        self.last_final_res   = 0
        self.last_final_angle = 0

        self.frames_left = 0       # 남은 프레임 수 < max_len
        
        self.collecting = False     # 수집 중 여부
        self.last_motion_state = False

        self.line_angles_list = []  # 누적 각도 저장 (10프레임)
        self.tip_angles_list = []

        self.last_line_angle = 90.0 # 초기값
        self.last_tip_angle = 90.0

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
        self.declare_parameter("limit_x", 40) # 핑크 라인 두께
        self.declare_parameter("max_len", 15) # 몇 프레임 동안 보고 판단할 거냐
        self.declare_parameter("delta_s", 15) # 빨간 점이 얼마나 벗어나야 커브일까
        self.declare_parameter("vertical", 75) # 직진 판단 각도
        self.declare_parameter("horizontal", 15) # 수평 판단 각도  <<<  아직 안 만듬

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
        self.recent_out = deque(maxlen=self.max_len) 

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
                    self.recent_out   = deque(maxlen=self.max_len) 
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
        #for idx, (cx, cy, lost) in enumerate(self.candidates):
        #    self.get_logger().info(f'[{idx}] cx={cx}, cy={cy}, lost={lost}') 잘 받아오고 있나 확인용

    def motion_callback(self, msg: MotionEnd):
        motion_end = bool(msg.end)
        rising = (not self.last_motion_state) and motion_end # 직전 값이 0, 지금 값이 1일 때 1 (False -> True)
        self.last_motion_state = motion_end # 직전 값 갱신
        if rising and not self.collecting:
            self.collecting = True  # 탐지 시작해라
            self.frames_left = self.max_len  # 프레임 초기화
            
            self.recent_curve.clear() # 이전 탐지 기록 초기화
            self.recent_tilt.clear()
            self.recent_out.clear()
            self.line_angles_list.clear()
            self.tip_angles_list.clear()

            self.count_turn = 0
            self.count_spin = 0
            self.count_straight = 0

            self.collect_t0 = time.perf_counter()
            self.collect_proc_sum = 0.0
            self.collect_frame_cnt = 0
            
            self.get_logger().info(f'[Line] Start collecting {self.max_len} frames')

    def _mean_ignore_none(self, values, trim_ratio=0.0):  # 최종 판단 계산
        xs = [float(v) for v in values if v is not None]
        if not xs:
            return None
        xs.sort()
        if 0.0 < trim_ratio < 0.5 and len(xs) >= 5:
            k = int(len(xs) * trim_ratio)
            xs = xs[k: len(xs)-k] if len(xs) - 2*k >= 1 else xs
        return float(np.mean(xs))

    def color_image_callback(self, msg): # 이미지 받아오기

        line_angle = None
        tip_angle  = None

        start_time = time.time()  # 시작 시간 기록

        # ROS 이미지 CV 이미지로 받아오기
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        roi = cv_image[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end]

        if self.collecting:
        #---------------------------------------------------------------판단 시작  
            t_proc0 = time.perf_counter()   # 시간 
            
            res_final = 0  # 모드
            angle_spin = 0.0
            angle_turn = 0.0
            angle_final = 0.0  # 회전각
            sum_x = 0  # 점선 평균 x 좌표
            # 1. 위치 판단
            if self.candidates:
                if len(self.candidates) >= 3:  #--------------------------------------------------------- 1.세 점 이상 탐지
                    tip_x, tip_y, tip_lost = self.candidates[-1] # 가장 위의 점
                    cv2.circle(roi, (tip_x - self.roi_x_start, tip_y - self.roi_y_start),
                            4, (0, 255, 255) if tip_lost > 0 else (0, 0, 255), -1)

                    for acx, acy, ac_lost in self.candidates[:-1]: # 나머지 점
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
                            angle_turn = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                            self.count_turn += 1
                            return
                        elif delta < -self.delta_s: 
                            self.curve_text = "Turn Right"
                            angle_turn = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                            self.count_turn += 1
                            return
                        else: 
                            self.curve_text = "Straight"
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
                            angle_turn = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                            self.count_turn += 1
                            
                            if self.horizontal < line_angle < self.vertical:
                                self.tilt_text = "Spin Left"
                            elif -self.horizontal > line_angle > -self.vertical:
                                self.tilt_text = "Spin Right"
                            else:
                                self.tilt_text = "Straight"
                        elif delta < -self.delta_s: 
                            self.curve_text = "Turn Right"
                            angle_turn = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                            self.count_turn += 1
                            
                            if self.horizontal < line_angle < self.vertical:
                                self.tilt_text = "Spin Left"
                            elif -self.horizontal > line_angle > -self.vertical:
                                self.tilt_text = "Spin Right"
                            else:
                                self.tilt_text = "Straight"
                        else: 
                            self.curve_text = "Straight"
                            
                            if self.horizontal < line_angle < self.vertical:
                                self.tilt_text = "Spin Left"
                                angle_spin += line_angle
                                self.count_spin += 1
                            elif -self.horizontal > line_angle > -self.vertical:
                                self.tilt_text = "Spin Right"
                                angle_spin += line_angle
                                self.count_spin += 1
                            else:
                                self.tilt_text = "Straight"

                        # 직선 시각화 하려고
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
                    cv2.line(roi, (x1 - self.roi_x_start, y1 - self.roi_y_start), (x2 - self.roi_x_start, y2 - self.roi_y_start), (255, 0, 0), 2)














                for i in range(len(self.candidates)):
                    sum_x += self.candidates[i][0]

                avg_x = float(sum_x / len(self.candidates))
                if avg_x - self.zandi_x > self.limit_x:
                    self.out_text = "Out Left"
                    angle_spin -= (avg_x - self.zandi_x) / 2   # <<<<  얼마나 벗어났느냐에 따라 목표 회전값 보정   <<<  수식 생각해~
                elif avg_x - self.zandi_x <  -self.limit_x:
                    self.out_text = "Out Right"
                    angle_spin += (avg_x - self.zandi_x) / 2
                else:
                    self.out_text = "Straight"
            
            # 2. 방향 판단 
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
                        angle_turn = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                        self.count_turn += 1
                    elif delta < -self.delta_s: 
                        self.curve_text = "Turn Right"
                        angle_turn = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                        self.count_turn += 1
                    else: 
                        self.curve_text = "Straight"
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
                        angle_turn = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                        self.count_turn += 1
                        
                        if self.horizontal < line_angle < self.vertical:
                            self.tilt_text = "Spin Left"
                        elif -self.horizontal > line_angle > -self.vertical:
                            self.tilt_text = "Spin Right"
                        else:
                            self.tilt_text = "Straight"
                    elif delta < -self.delta_s: 
                        self.curve_text = "Turn Right"
                        angle_turn = math.degrees(math.atan(-(tip_y - self.zandi_y) / (tip_x - self.zandi_x)))
                        self.count_turn += 1
                        
                        if self.horizontal < line_angle < self.vertical:
                            self.tilt_text = "Spin Left"
                        elif -self.horizontal > line_angle > -self.vertical:
                            self.tilt_text = "Spin Right"
                        else:
                            self.tilt_text = "Straight"
                    else: 
                        self.curve_text = "Straight"
                        
                        if self.horizontal < line_angle < self.vertical:
                            self.tilt_text = "Spin Left"
                            angle_spin += line_angle
                            self.count_spin += 1
                        elif -self.horizontal > line_angle > -self.vertical:
                            self.tilt_text = "Spin Right"
                            angle_spin += line_angle
                            self.count_spin += 1
                        else:
                            self.tilt_text = "Straight"

                    # 직선 시각화 하려고
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
                cv2.line(roi, (x1 - self.roi_x_start, y1 - self.roi_y_start), (x2 - self.roi_x_start, y2 - self.roi_y_start), (255, 0, 0), 2)
            
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
                    tip_angle = None
                    self.tilt_text = "Straight"
                    x = int(round((x1 + x2) * 0.5))
                    x1 = x2 = x
                    y1, y2 = self.roi_y_end, self.roi_y_start
                else:  # 2) 일반적
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                    line_angle = math.degrees(math.atan(m))
                    tip_angle = None
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
                line_angle = tip_angle = None


            # 이번 프레임의 라벨 결정(한 프레임 기준)
            self.recent_curve.append(self.curve_text)
            self.recent_tilt.append(self.tilt_text)
            self.recent_out.append(self.out_text)
            # line_angle, tip_angle 계산 완료 후 (None 가능)
            if line_angle is not None:
                self.line_angles_list.append(float(line_angle))
                self.last_line_angle = float(line_angle)

            if tip_angle is not None:
                self.tip_angles_list.append(float(tip_angle))
                self.last_tip_angle = float(tip_angle)
            
            self.frames_left -= 1

            # 창이 다 찼으면 최종 결정
            if self.frames_left <= 0:
                curve_final = Counter(self.recent_curve).most_common(1)[0][0] if self.recent_curve else "Miss"
                tilt_final  = Counter(self.recent_tilt ).most_common(1)[0][0] if self.recent_tilt  else "Miss"
                out_final = Counter(self.recent_out).most_common(1)[0][0] if self.recent_out else "Miss"

                if curve_final == "Straight" and tilt_final == "Straight" and out_final == "Straight":
                    res_final = 1
                elif tilt_final == "Spin Right" or out_final == "Out Left":
                    res_final = 2
                elif tilt_final == "Spin Left" or out_final == "Out Right":
                    res_final = 3
                elif curve_final in ("Turn Left", "Turn Right"):
                    res_final = 4
                else:
                    res_final = 99

                # --- angle_final 계산 (산술 평균, None 제외) ---
                if res_final == 1:
                    angle_final = 90.0

                elif res_final in (2, 3):  # heading 정렬
                    m = self._mean_ignore_none(self.line_angles_list, trim_ratio=0.0)  # 트림 원하면 0.1
                    if m is None:
                        m = self.last_line_angle
                    angle_final = float(np.clip(m, -90.0, 90.0))  # 필요시 제한

                elif res_final == 4:       # 커브 진입
                    m = self._mean_ignore_none(self.tip_angles_list, trim_ratio=0.0)
                    if m is None:
                        m = self.last_tip_angle
                    angle_final = float(np.clip(m, -120.0, 120.0))  # 필요시 제한

                else:  # 99 Miss
                    angle_final = 0.0

                angle_pub = abs(int(round(angle_final)))  # 크기만 필요할 때

                # 퍼블리시
                msg_out = LineResult()
                msg_out.res = res_final
                msg_out.angle = angle_pub
                self.line_result_pub.publish(msg_out)

                self.last_final_curve = curve_final
                self.last_final_tilt  = tilt_final
                self.last_final_res   = res_final
                self.last_final_angle = angle_pub

                wall_s = (time.perf_counter() - self.collect_t0) if self.collect_t0 is not None else 0.0
                proc_s = self.collect_proc_sum
                self.get_logger().info(
                    f"[Line] Window done: res={res_final}, angle={angle_pub}, "
                    f"frames={self.collect_frame_cnt}, "
                    f"wall={wall_s*1000:.1f} ms, proc={proc_s*1000:.1f} ms")

                # 상태 종료
                self.collecting = False
                self.get_logger().info(f'[Line] Done: res={res_final}, angle={msg_out.angle}')
            
            #------------------------------------------------------------------------------------------------------------  출력

            disp_line = "--" if line_angle is None else f"{line_angle:.2f}"
            disp_tilt = self.tilt_text
            disp_curve = self.curve_text

            # 판단 텍스트 출력
            cv2.putText(cv_image, f"Rotate: {disp_line}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Tilt: {disp_tilt}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Curve: {disp_curve}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Out: {self.out_text}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            
            t_proc1 = time.perf_counter()
            self.collect_proc_sum += (t_proc1 - t_proc0)
            self.collect_frame_cnt += 1

        else: 
            cv2.putText(cv_image, "Idle", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        

        cv_image[self.roi_y_start:self.roi_y_end, self.roi_x_start:self.roi_x_end] = roi  # 전체화면
        cv2.rectangle(cv_image, (self.roi_x_start - 1, self.roi_y_start - 1), (self.roi_x_end + 1, self.roi_y_end), (0, 255, 0), 1) # ROI 구역 표시
        cv2.line(cv_image, (self.zandi_x - self.limit_x, self.roi_y_start), (self.zandi_x - self.limit_x, self.roi_y_end), (255, 22, 255), 1)
        cv2.line(cv_image, (self.zandi_x + self.limit_x, self.roi_y_start), (self.zandi_x + self.limit_x, self.roi_y_end), (255, 22, 255), 1)
        cv2.circle(cv_image, (self.zandi_x, self.zandi_y), 5, (255, 255, 255), -1)
        #-------------------------------------------------------------------------------------------------- 프레임 처리 시간 측정

        # -------------------------------------------------------------------------------------------------
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

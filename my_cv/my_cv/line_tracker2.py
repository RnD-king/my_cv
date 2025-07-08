

### 

import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
# 디버깅용
import time 

class RectangleTracker:  # 사각형(점선) 추적기 - 점선 감지 실패하면 몇 프레임 정도는 이전 위치 기억하기
    def __init__(self, max_lost, max_dist, min_found=10):  # 몇 프레임, 몇 픽셀 이내, 검출 안정화 프레임
        # rectangles: {ID: (중심 x, 중심 y, 놓친 프레임 수, 검출된 프레임 수)}
        self.rectangles = {}
        self.next_id = 0
        self.max_lost = max_lost
        self.max_dist = max_dist
        self.min_found = min_found

    def update(self, detected_centroids):
        updated_rects = {}
        matched_ids = set()

        # 1) 새로운 중심점들과 기존 중심점 매칭
        for detected_cx, detected_cy in detected_centroids:
            best_id = None
            min_dist = self.max_dist

            for id, (previous_cx, previous_cy, lost, found) in self.rectangles.items():
                if id in matched_ids:
                    continue
                dist = math.hypot(detected_cx - previous_cx, detected_cy - previous_cy)
                if dist < min_dist:
                    min_dist = dist
                    best_id = id

            if best_id is not None:
                # 기존 트랙 보정, lost 리셋, found+1
                _, _, _, prev_found = self.rectangles[best_id]
                updated_rects[best_id] = (detected_cx, detected_cy, 0, min(prev_found + 1, self.min_found))
                matched_ids.add(best_id)
            else:
                # 신규 트랙 생성, found=1
                updated_rects[self.next_id] = (detected_cx, detected_cy, 0, 1)
                self.next_id += 1

        # 2) 매칭되지 않은 중심점은 lost 증가, found 유지
        for id, (previous_cx, previous_cy, lost, found) in self.rectangles.items():
            if id not in matched_ids and lost < self.max_lost:
                updated_rects[id] = (previous_cx, previous_cy, lost + 1, found)

        self.rectangles = updated_rects
        return self.rectangles

class ImgSubscriber(Node):
    def __init__(self):
        super().__init__('img_subscriber')
        self.subscription_color = self.create_subscription(  # 컬러
            Image,
            '/camera/camera/color/image_raw',  # RealSense에서 제공하는 컬러 이미지 토픽
            self.color_image_callback, 10)
        
        self.bridge = CvBridge()

        self.curve_count = 0
        self.tracker = RectangleTracker(max_lost=20, max_dist=50, min_found=60) #신규 10프레임 실종 20프레임 / 50픽셀
        self.tilt_text = "" # 화면 출력
        self.curve_text = "" # 화면 출력
        # 시간 디버깅
        self.frame_count = 0
        self.total_time = 0.0
        self.last_report_time = time.time()
        self.last_avg_text = "AVG: --- ms | FPS: --"


    def get_angle(self, c1, c2): # 단순 각도 계산
        dx = c2[0]-c1[0]
        dy = c2[1]-c1[1]
        angle = 180 / math.pi * math.atan2(dy, dx) + 90
        if angle > 89:  # 계산 과부하 방지
            angle = 89
        elif angle < -89:
            angle = -89
        return round(angle, 2)

    def color_image_callback(self, msg):

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

            # GPU 메모리에 이미지 업로드   ----------------------------------------------------------------------------------------------CUDA 시작
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(roi)

            # CUDA용 Gaussian 필터 생성
            gpu_gaussian = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 0)

            # Gaussian 필터 적용
            gpu_blurred = gpu_gaussian.apply(gpu_image)

            # BGR -> HSV 변환 (CUDA 버전)
            gpu_hsv = cv2.cuda.cvtColor(gpu_blurred, cv2.COLOR_BGR2HSV)

            # GPU 메모리에서 HSV 이미지 다운로드   ---------------------------------------------------------------------------------------CUDA 끝
            hsv = gpu_hsv.download()   

            # CLAHE 적용
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            h, s, v = cv2.split(hsv) 
            v = clahe.apply(v) # v값만 보정
            hsv_image = cv2.merge((h, s, v))

            # 1. RGB 이미지로 변환
            rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)  # 보정 끝낸 rgb 화면 < 내가 보는 세상

            # 2. HSV 흰색 범위 지정
            hsv_upper_white = np.array([180, 40, 255])
            hsv_lower_white = np.array([0, 0, 70])
            hsv_mask = cv2.inRange(hsv_image, hsv_lower_white, hsv_upper_white)

            # 컨투어 찾기
            contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rect_centroids = []  # 무게중심 리스트 (cx, cy)
            
            for cnt in contours:    
                # -------------------------------------------------------------- 빨간 사각형 -- 근사화를 해서 사각형을 찾음
                epsilon = 0.08 * cv2.arcLength(cnt, True)  # 근사화 정확도 낮춤 / 0.05정도로 하라는데 조정 필요
                approx = cv2.approxPolyDP(cnt, epsilon, True)  # 윤곽선 근사

                if len(approx) == 4:  # 꼭짓점이 4개면 사각형으로 간주
                    # 회전된 외접 사각형(rectangle) 정보 얻기
                    rect = cv2.minAreaRect(approx) # 외접 박스
                    (box_cx, box_cy), (box_w, box_h), box_ang = rect
                    area_rect = box_w * box_h
                    if 2000 > area_rect > 500:  # 외접 박스 면적이 500 이상인 경우만 선택
                        angles = []
                        for i in range(4):  # 각도 계산
                            p1, p2, p3 = approx[i - 1][0], approx[i][0], approx[(i + 1) % 4][0]
                            v1 = np.array(p1) - np.array(p2)
                            v2 = np.array(p3) - np.array(p2)
                            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # 라디안 → 도 변환
                            angles.append(angle)

                        # 모든 내각이 60°~120° 사이인지 확인
                        if all(60 <= angle <= 120 for angle in angles):
                            cv2.drawContours(rgb_image, [approx], 0, (200, 0, 255), 2)  # 모든 조건 맞으면 빨간색 사각형 그리기 
                            M = cv2.moments(cnt)  # 그 사각형 중심 구하기
                            if M["m00"] != 0:
                                rcx = int(M["m10"] / M["m00"])  # 중심 x 좌표
                                rcy = int(M["m01"] / M["m00"])  # 중심 y 좌표
                                rect_centroids.append((rcx, rcy))  # 사각형 중심 리스트에 추가 
                                cv2.drawContours(rgb_image, [approx], 0, (200, 0, 255), 2) # 테두리 핑크 / 일단 사각형 검출 하는지 확인용 / 지금 프레임에서 사각형이라고 검출됨
                                
            # -------------------------------------------------------------------- 사각형 추적
            tracked_rects = self.tracker.update(rect_centroids)  # 사각형 중심을 추적기에 보냄 {id: (cx, cy, lost, found)} >>> 원래 있던 사각형이랑 맞는지 비교하고 나온 사각형
            valid_rects = {
                id:(cx, cy, lost, found)
                for id,(cx,cy,lost,found) in tracked_rects.items()
                if lost <= self.tracker.max_lost and found >= self.tracker.min_found
            }  # 감지한 사각형 중에, lost, found 조건 맞는 애들만
            
            for id,(cx,cy,lost,found) in valid_rects.items(): 
                cv2.putText(rgb_image, f"ID: {id}", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # id 출력

            #---------------------------------------------------------------라인 판단 시작  << 최소자승 직선 근  사
            
            candidates = sorted(
                [(cx, cy, lost) for (cx, cy, lost, found) in valid_rects.values()],
                key=lambda c: c[1], reverse=True)[:5] # 후보점 5개 선택 (화면 아래쪽부터)
            
            if len(candidates) >= 3:  # 3개 이상일 때: 맨 위 점 제외하고 직선 근사              
                tip_x, tip_y, tip_lost = candidates[-1] # 그 중 가장 위의 점 = 새로 탐지된 점
                if tip_lost > 0:
                    cv2.circle(rgb_image, (tip_x, tip_y), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
                else:
                    cv2.circle(rgb_image, (tip_x, tip_y), 4, (0, 0, 255), -1) # 가장 위의 새로운 점은 빨강
                
                for acx, acy, ac_lost in candidates[:-1]:
                    if ac_lost > 0:
                        cv2.circle(rgb_image, (acx, acy), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
                    else:
                        cv2.circle(rgb_image, (acx, acy), 4, (255, 0, 0), -1) # 나머지 파랑

                xs = np.array([c[0] for c in candidates[:-1]])
                ys = np.array([c[1] for c in candidates[:-1]])

                if abs(xs.max() - xs.min()) < 2:  # 수직에 가까운 경우
                    x1, y1, _ = candidates[0]
                    x2, y2, _ = candidates[-2]
                    line_angle = 90
                    delta = (xs.max() + xs.min()) / 2 - tip_x
                    
                    # 커브 판단
                    if delta < -15:
                        self.curve_text = "Turn Right"
                    elif delta > 15:
                        self.curve_text = "Turn Left"
                    else:
                        self.curve_text = "Straight"
                    self.tilt_text = "Straight"

                else: # 일반적인 경우
                    m, b = np.polyfit(xs, ys, 1)
                    line_angle = math.degrees(math.atan2(m, 1))

                    # 점과 직선 사이의 수직 거리
                    numerator = m * tip_x - tip_y + b
                    denominator = math.sqrt(m**2 + 1)
                    delta = numerator / denominator

                    # 직선 시각화 방식 선택 (기울기 크기에 따라)
                    if abs(m) > 1:  # 기울기 크면 y 기준 (거의 수직)
                        y1 = 0
                        y2 = roi_y_end - roi_y_start
                        x1 = int((y1 - b) / m)
                        x2 = int((y2 - b) / m)
                    else:  # 거의 수평일 땐 x 기준
                        x1 = 0
                        x2 = roi_x_end - roi_x_start
                        y1 = int(m * x1 + b)
                        y2 = int(m * x2 + b)

                    # 커브 판단
                    if delta < -15 and m > 0:
                        self.curve_text = "Turn Left"
                    elif delta > 15 and m > 0:
                        self.curve_text = "Turn Right"
                    elif delta < -15 and m < 0:
                        self.curve_text = "Turn Right"
                    elif delta > 15 and m < 0:
                        self.curve_text = "Turn Left"
                    else:
                        self.curve_text = "Straight"

                    if 10 < line_angle < 80:
                        self.tilt_text = "Spin Left"
                    elif -10 > line_angle > -80:
                        self.tilt_text = "Spin Right"
                    else: # 평행선 아직 처리 안 함
                        self.tilt_text = "Straight"

                cv2.line(rgb_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            elif len(candidates) == 2:

                x1, y1, down_lost = candidates[0] # 아래
                x2, y2, up_lost = candidates[1] # 위
                if down_lost > 0:
                    cv2.circle(rgb_image, (x1, y1), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
                else:
                    cv2.circle(rgb_image, (x1, y1), 4, (0, 0, 255), -1) #  파랑
                if up_lost > 0:
                    cv2.circle(rgb_image, (x2, y2), 3, (0, 255, 255), 2) # lost가  0이상이면 노랑
                else:
                    cv2.circle(rgb_image, (x2, y2), 4, (255, 0, 0), -1) # 빨강

                if abs(x1 - x2) < 2:
                    line_angle = 90
                    self.tilt_text = "Straight"
                else:
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                    line_angle = math.degrees(math.atan2(m, 1))
                    # 직선 시각화 방식 선택 (기울기 크기에 따라)
                    if abs(m) > 1:  # 기울기 크면 y 기준 (거의 수직)
                        y1 = 0
                        y2 = roi_y_end - roi_y_start
                        x1 = int((y1 - b) / m)
                        x2 = int((y2 - b) / m)
                    else:  # 거의 수평일 땐 x 기준
                        x1 = 0
                        x2 = roi_x_end - roi_x_start
                        y1 = int(m * x1 + b)
                        y2 = int(m * x2 + b)
                    
                    if 10 < line_angle < 80:
                        self.tilt_text = "Spin Left"
                    elif -10 > line_angle > -80:
                        self.tilt_text = "Spin Right"
                    else: # 평행성 아직 처리 안 함
                        self.tilt_text = "Straight"

                self.curve_text = "Straight"
                cv2.line(rgb_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            else:
                self.curve_text = "Miss"
                self.tilt_text = "Miss"
                line_angle = 0


            #---------------------------------------------------------------------------------------------------------------라인 판단 끝

            # 방향 텍스트 출력
            cv2.putText(cv_image, f"Rotate: {line_angle:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Tilt: {self.tilt_text}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(cv_image, f"Curve: {self.curve_text}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

            cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = rgb_image  # 전체화면
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

            # 평균 텍스트 영상에 출력 (매 프레임에 동일한 텍스트 유지)
            if hasattr(self, "last_avg_text"):
                cv2.putText(cv_image, self.last_avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

            # 결과 출력
            cv2.imshow("Only HSV White", hsv_mask)
            cv2.imshow("Contours Rectangle", cv_image)  # 윤곽선 근사 사각형 - 빨강
            cv2.waitKey(1)

def main():
    rclpy.init()
    node = ImgSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
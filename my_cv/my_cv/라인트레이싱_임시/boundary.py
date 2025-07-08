# 흰색을 먼저 보고, 1. 최소 사각형 그려서 초록 사각형
#                2. 꼭짓점 4개 찾아서 윤곽선 사각형 그려서 빨강 사각형

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class RectangleTracker:  # 사각형 추적기
    def __init__(self, max_lost=20, dist_thresh=100): # 10프레임, 50 픽셀
        self.rectangles = {}  # {ID: (x, y, w, h, angle, lost_count, centroid)}
        self.next_id = 0
        self.max_lost = max_lost
        self.dist_thresh = dist_thresh

    def update(self, detected_rects):
        updated_rects = {}
        matched_ids = set()  # 이미 매칭된 ID 저장

        # 새로운 사각형과 기존 사각형 매칭
        for rect in detected_rects:
            x, y, w, h, angle = rect
            best_match = None
            min_dist = 100. # 얘보다 가까우면 같은 사각형임

            for id, (px, py, pw, ph, p_angle, lost) in self.rectangles.items():
                if id in matched_ids:
                    continue  # 이미 매칭된 사각형은 제외

                # 거리 계산
                dist = np.linalg.norm([x - px, y - py])

                if dist < self.dist_thresh:  # 크기 차이가 너무 크면 제외
                    if dist < min_dist:
                        min_dist = dist
                        best_match = id # 아까 그 놈

            if best_match is not None:
                updated_rects[best_match] = (x, y, w, h, angle, 0)
                matched_ids.add(best_match)
            else:
                updated_rects[self.next_id] = (x, y, w, h, angle, 0)
                self.next_id += 1

        # 못 찾으면 lost 증가
        for id, (px, py, pw, ph, p_angle, lost) in self.rectangles.items():
            if id not in updated_rects:
                if lost < self.max_lost:
                    updated_rects[id] = (px, py, pw, ph, p_angle, lost + 1)

        self.rectangles = updated_rects
        return self.rectangles

class ImgSubscriber(Node):
    def __init__(self):
        super().__init__('img_subscriber')

        self.get_logger().info("Start Canny Edge Detector.")  # 로그

        self.subscription_color = self.create_subscription(  # 컬러
            Image,
            '/camera/camera/color/image_raw',  # RealSense에서 제공하는 컬러 이미지 토픽
            self.color_image_callback,
            10)
        
        self.bridge = CvBridge()
        self.curve_count = 0
        self.red_tracker = RectangleTracker(max_lost=10, dist_thresh=50)
        self.green_tracker = RectangleTracker(max_lost=10, dist_thresh=50)

    def get_angle(self, c1, c2):
        dx = c2[0]-c1[0]
        dy = c2[1]-c1[1]
        angle = 180 / math.pi * math.atan2(dy, dx) + 90
        if angle > 89:
            angle = 89
        elif angle < -89:
            angle = -89
        return round(angle, 2)

    def color_image_callback(self, msg):

            # ROS 이미지 CV 이미지로 받아오기
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            h, w, _ = cv_image.shape

            # ROI 좌표 정의
            roi_y_start = h * 1 // 6  # 위
            roi_y_end = h // 1  # 아래
            roi_x_start = w * 2 // 5  # 왼 
            roi_x_end = w * 3 // 5  # 오

            # 블러 적용
            blr = cv2.GaussianBlur(cv_image, (5, 5), 0)

            # BGR -> HSV, CLAHE 적용
            hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            h_, s, v = cv2.split(hsv)
            v = clahe.apply(v)
            hsv_image = cv2.merge((h_, s, v))
            rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR) # 보정 끝낸 rgb 화면

            roi = hsv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            rgb_roi = rgb_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            # 흰색 범위 지정
            hsv_upper_white = np.array([180, 40, 255])
            hsv_lower_white = np.array([0, 0, 70])
            hsv_mask = cv2.inRange(roi, hsv_lower_white, hsv_upper_white)

            # 컨투어 찾기
            contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect_image = rgb_roi.copy()
            polygon_image = rgb_roi.copy()  # approxPolyDP 결과 저장

            red_centroids = []  # 무게중심 리스트
            green_centroids = []
            red_tracking = []
            green_tracking = []
            
            for cnt in contours:    
                # -------------------------------------------------------------- 빨간 사각형 -- 근사화를 해서 사각형을 찾음
                epsilon = 0.08 * cv2.arcLength(cnt, True)  # 근사화 정확도 낮춤
                approx = cv2.approxPolyDP(cnt, epsilon, True)  # 윤곽선 근사화

                if len(approx) == 4:  # 꼭짓점이 4개면 사각형으로 간주
                    x, y, w, h = cv2.boundingRect(approx)  # 근사 다각형의 크기 측정
                    if 20 <= w <= 120 and 20 <= h <= 120 and w + h >= 70:  # 기존 minAreaRect와 동일한 크기 조건 적용
                        area = cv2.contourArea(approx)
                        if area > 500:  # 일정 크기 이상인 경우만 선택 << 오목 사각형이 나오기도 함
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
                                cv2.drawContours(polygon_image, [approx], 0, (200, 0, 255), 2)  # 빨간색 사각형 그리기 
                                M = cv2.moments(cnt) # 중심 구하기
                                if M["m00"] != 0:
                                    rcx = int(M["m10"] / M["m00"])  # 중심 x 좌표
                                    rcy = int(M["m01"] / M["m00"])  # 중심 y 좌표
                                    cv2.circle(polygon_image, (rcx, rcy), 3, (255, 0, 0), -1)  # 파란색 점 찍고
                                    red_centroids.append((rcx, rcy))  # 리스트에 추가     
                                
                                red_tracking.append((rcx, rcy, w, h, angle))  # 빨간색 추적에 정보 저장 (중심, 너비 높이, 각도)
                
                # --------------------------------------------------------------초록 사각형 -- 흰색 뭉탱이를 보면 사각형을 만듬  + 중심에는 파란 점
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect  # 기본 저장 값 [0], [1], [2]
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                if 20 <= w <= 120 and 20 <= h <= 120 and w + h >= 70:
                    cv2.drawContours(rect_image, [box], 0, (0, 255, 0), 2)  # 초록색 사각형
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        gcx = int(M["m10"] / M["m00"])  # 중심 x 좌표
                        gcy = int(M["m01"] / M["m00"])  # 중심 y 좌표
                        cv2.circle(rect_image, (gcx, gcy), 3, (255, 0, 0), -1)  # 파란색 점
                        green_centroids.append((gcx, gcy))  # 리스트에 추가    

                    green_tracking.append((gcx, gcy, w, h, angle))                  


            # -------------------------------------------------------------------- 사각형 추적
            red_tracked_rects = self.red_tracker.update(red_tracking)  # 클래스 업뎃
            green_tracked_rects = self.green_tracker.update(green_tracking)

            #ㄷㄷ
            tracked_centroids = [(cx, cy) for (cx,cy,_,_,_,lost) in self.red_tracker.rectangles.values() if lost < self.red_tracker.max_lost]
            
            for id, (rcx, rcy, w, h, angle, lost) in red_tracked_rects.items():  # 추적 실패한 빨간 사각형 (중심, 너비 높이, 각, 놓친 거)
                if lost > 0: 
                    cv2.circle(polygon_image, (rcx,rcy), 3, (0, 255, 255), 2)  # 노란점
                cv2.putText(polygon_image, f"ID: {id}", (rcx, rcy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            for id, (gcx, gcy, w, h, angle, lost) in green_tracked_rects.items():  #추적 실패한 초록 사각형
                if lost > 0:  
                    cv2.circle(rect_image, (gcx,gcy), 3, (0, 255, 255), 2)  # 노란점
                cv2.putText(rect_image, f"ID: {id}", (gcx, gcy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            #----------------------------------------------------------------------------------------------------------------빨간 사각형에 파란 선
            # 중복 제거 (같은 좌표일 경우만 제거하는 간단한 방식)
            all_centroids = red_centroids + [c for c in tracked_centroids if c not in red_centroids]
            red_centroids = sorted(all_centroids, key=lambda c: c[1], reverse=True)[:5]  # 아래부터 정렬, 무게중심 5개만
            line_angle = []
            curve_to_right = 0
            curve_to_left = 0
            horizon = 0

            if len(red_centroids) >= 2:               
                for i in range(len(red_centroids) - 1): # 각도 판단
                    angle = self.get_angle(red_centroids[i], red_centroids[i + 1])
                    line_angle.append(angle) # 각도 저장
                    if angle >= 20: # 20도 이상이면 우회전 + 1
                        curve_to_right += 1
                    if angle <= -20: # -20도 이하면 좌회전 + 1
                        curve_to_left += 1
                    if abs(angle) >= 70: # +-70도 넘으면 수평선 + 1
                        horizon += 1

                # 2. 수평선이면 → 왼쪽부터 정렬  <<< 기울기 오류 안 내려고
                if horizon == len(line_angle):
                    print("Horizon")
                    red_centroids = sorted(red_centroids, key=lambda c: c[0])  # 왼쪽부터 정렬

                # 3. 아니면 아래부터 정렬
                else:
                    red_centroids = sorted(red_centroids, key=lambda c: c[1], reverse=True)  # 아래부터 정렬

                # 4. 정렬 후 선 그리기
                for i in range(len(red_centroids) - 1):
                    angle = self.get_angle(red_centroids[i], red_centroids[i + 1])
                    cv2.line(polygon_image, red_centroids[i], red_centroids[i + 1], (255, 0, 0), 2)

                # 5. 최종 판단 출력
                if curve_to_right >= 2: # 선 4개 중에 20도 이상 기울기가 2개 이상이라면
                    if self.curve_count >= 5: # 5프레임 연속이라면
                        print("Turn Right") # 나중에 상태값이나 함수 등등 넣을 곳     <<<<<<<<<< 커브 움직임
                    else:
                        self.curve_count += 1
                elif curve_to_left >= 2:
                    if self.curve_count >= 5:
                        print("Turn Left")
                    else:
                        self.curve_count += 1
                elif horizon == len(line_angle):
                    print("Horizon")
                    self.curve_count = 0
                else:
                    self.curve_count = 0 # 직선 구간   <<<<<<<<<<<<<<  1, 2 점이 중심에서 벗어마녀 게걸음

                # 6. 평균 기울기 계산 (앞 3개 점 기준)  << 이거로 로봇 회전 판단  <<<< 평소에는 이거로 방향 보정
                if len(red_centroids) >= 3:
                    angle1 = self.get_angle(red_centroids[0], red_centroids[1])
                    angle2 = self.get_angle(red_centroids[1], red_centroids[2])
                    rotate = (angle1 + angle2) / 2
                elif len(red_centroids) == 2: # 점 2개만 탐지시 (그럴 일 거의 없음)
                    rotate = self.get_angle(red_centroids[0], red_centroids[1])
                else:
                    rotate = 0  # 각도를 계산할 수 없을 경우 기본값
                print(f"{rotate:.2f}")
            

            #-------------------------------------------------------------------------------------------------------------------초록 사각형에 파란 선
            green_centroids = sorted(green_centroids, key=lambda c: c[1], reverse=True)[:3]  # 아래부터 정렬, 무게중심 3개 선택

            if len(green_centroids) >= 2:
                # OpenCV fitLine을 이용해 직선 근사 (vx, vy: 방향 벡터)  
                [gvx, gvy, x0, y0] = cv2.fitLine(np.array(green_centroids), cv2.DIST_L2, 0, 0.01, 0.01) # << 무게중심 두 점을 가장 잘 통과하는 직선 찾기
                
                # 기울기를 이용해 가로/세로 판단
                if abs(gvy / (gvx + 1000)) > 1:  # 분모가 0되면 안 되니까
                    green_centroids = sorted(green_centroids, key=lambda c: -c[1])  # 세로 < 아래부터 정렬
                elif gvy / (gvx + 1000) > 0:
                    green_centroids = sorted(green_centroids, key=lambda c: -c[0])  # 우상향 가로 < 오른쪽부터 정렬
                else:
                    green_centroids = sorted(green_centroids, key=lambda c: c[0])  # 좌상향 가로 < 왼쪽부터 정렬
                
                # 점선의 무게중심을 연결하는 파란 선 그리기
                for i in range(len(green_centroids) - 1):
                    cv2.line(rect_image, green_centroids[i], green_centroids[i + 1], (255, 0, 0), 2)

            rgb_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = polygon_image  # 전체화면
            cv2.rectangle(rgb_image, (roi_x_start - 1, roi_y_start - 1), (roi_x_end + 1, roi_y_end), (0, 255, 0), 1)

            # 결과 출력
            cv2.imshow("Only HSV White", hsv_mask)
            cv2.imshow("Contours White", rect_image)  # minAreaRect로 만들어낸 사각형
            cv2.imshow("Contours Rectangle", rgb_image)  # 윤곽선 근사 사각형 - 빨강
            cv2.waitKey(1)



def main():
    rclpy.init()
    node = ImgSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

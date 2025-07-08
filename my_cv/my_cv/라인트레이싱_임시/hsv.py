# 흰색  ->  컨투어 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange
from rcl_interfaces.msg import SetParametersResult

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

    def color_image_callback(self, msg):

            # ROS 이미지 CV 이미지로 받아오기
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            h, w, _ = cv_image.shape

            # ROI 좌표 정의
            roi_y_start = h * 2 // 6  # 위
            roi_y_end = h // 1  # 아래
            roi_x_start = w // 3  # 왼 
            roi_x_end = w * 2 // 3  # 오

            # ROI 설정: 하단 절반에서 가운데 부분만 추출
            roi = cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            # 블러 적용
            blr = cv2.GaussianBlur(roi, (5, 5), 0)

            # BGR -> HSV, CLAHE 적용
            hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            h_, s, v = cv2.split(hsv)
            v = clahe.apply(v)
            hsv_image = cv2.merge((h_, s, v))

            # RGB 변환
            rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

            # 흰색 범위 지정
            hsv_upper_white = np.array([180, 40, 255])
            hsv_lower_white = np.array([0, 0, 70])
            hsv_mask = cv2.inRange(hsv_image, hsv_lower_white, hsv_upper_white)

            # 전체 크기로 변환된 흰색 마스킹 영상
            full_hsv_white = np.zeros_like(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY))
            full_hsv_white[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = hsv_mask

            # 컨투어 찾기
            contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect_image = roi.copy()
            polygon_image = roi.copy()  # approxPolyDP 결과 저장

            centroids = []  # 무게중심 리스트
            for cnt in contours:
                
                # -------------------------------------------------------------- 빨간 사각형 -- 근사화를 해서 사각형을 찾음
                epsilon = 0.08 * cv2.arcLength(cnt, True)  # 근사화 정확도 낮춤
                approx = cv2.approxPolyDP(cnt, epsilon, True)  # 윤곽선 근사화

                if len(approx) == 4:  # 꼭짓점이 4개면 사각형으로 간주
                    x, y, w, h = cv2.boundingRect(approx)  # 근사 다각형의 크기 측정
                    if w >= 20 and w <= 120 and h <= 120 and h >= 20:  # 기존 minAreaRect와 동일한 크기 조건 적용
                        area = cv2.contourArea(approx)
                        if area > 500:  # 일정 크기 이상인 경우만 선택
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
                                cv2.drawContours(polygon_image, [approx], 0, (0, 0, 255), 2)  # 빨간색 사각형
                
                # --------------------------------------------------------------초록 사각형 -- 흰색 뭉탱이를 보면 사각형을 만듬
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                width, height = rect[1]  # 너비, 높이 정보 가져오기 

                if width >= 20 and width <= 120 and height <= 120 and height >= 20:  # 작은 사각형 필터링
                    cv2.drawContours(rect_image, [box], 0, (0, 255, 0), 2)  # 초록색 사각형
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])  # 중심 x 좌표
                        cy = int(M["m01"] / M["m00"])  # 중심 y 좌표
                        cv2.circle(rect_image, (cx, cy), 3, (255, 0, 0), -1)  # 파란색 점

                        centroids.append((cx, cy))  # 리스트에 추가

            # --------------------------------------------------------------------파란 선 -- 파란 점 따라감 / 기울기 구함
            centroids = sorted(centroids, key=lambda c: c[1], reverse=True)[:3]  # 아래부터 정렬, 무게중심 3개 선택

            if len(centroids) > 5:
                centroids = centroids[-5:]  # 리스트에서 최근 5개만 유지 / 화면엔 ㄱㅊ

            if len(centroids) >= 2:
                # OpenCV fitLine을 이용해 직선 근사 (vx, vy: 방향 벡터)  
                [vx, vy, x0, y0] = cv2.fitLine(np.array(centroids), cv2.DIST_L2, 0, 0.01, 0.01) # << 무게중심 두 점을 가장 잘 통과하는 직선 찾기
                
                # 기울기를 이용해 가로/세로 판단
                if abs(vy / (vx + 1e-4)) > 1:  # 분모가 0되면 안 되니까
                    centroids = sorted(centroids, key=lambda c: -c[1])  # 세로 < 아래부터 정렬
                elif vy / (vx + 1e-4) > 0:
                    centroids = sorted(centroids, key=lambda c: -c[0])  # 우상향 가로 < 오른쪽부터 정렬
                else:
                    centroids = sorted(centroids, key=lambda c: c[0])  # 좌상향 가로 < 왼쪽부터 정렬
                
                # 점선의 무게중심을 연결하는 파란 선 그리기
                for i in range(len(centroids) - 1):
                    cv2.line(rect_image, centroids[i], centroids[i + 1], (255, 0, 0), 2)

            # 결과 출력
            cv2.imshow("RealSense RGB Image", rgb_image)
            cv2.imshow("RealSense HSV Image", hsv_image)
            cv2.imshow("Only HSV White", full_hsv_white)
            cv2.imshow("Contours White", rect_image)  # minAreaRect로 만들어낸 사각형
            cv2.imshow("Contours Rectangle", polygon_image)  # 윤곽선 근사 사각형
            cv2.waitKey(1)

    


def main():
    rclpy.init()
    node = ImgSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

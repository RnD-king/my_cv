# white 개선

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge

class RectangleTracker:
    def __init__(self, dist_thresh=50, max_lost=5):
        self.rectangles = {}  # {id: (x, y, w, h, angle, lost)}
        self.next_id = 0
        self.dist_thresh = dist_thresh
        self.max_lost = max_lost

    def iou(self, rect1, rect2):
        """ 두 사각형 간 IoU 계산 """
        x1, y1, w1, h1, _ = rect1
        x2, y2, w2, h2, _ = rect2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area else 0

    def update(self, detected_rects):
        updated_rects = {}
        assigned_ids = set()

        for rect in detected_rects:
            x, y, w, h, angle = rect
            best_id = None
            best_iou = 0

            for id, (px, py, pw, ph, p_angle, lost) in self.rectangles.items():
                iou_score = self.iou(rect, (px, py, pw, ph, p_angle))
                if iou_score > best_iou and iou_score > 0.2 and id not in assigned_ids:
                    best_iou = iou_score
                    best_id = id

            if best_id is not None:
                updated_rects[best_id] = (x, y, w, h, angle, 0)
                assigned_ids.add(best_id)
            else:
                updated_rects[self.next_id] = (x, y, w, h, angle, 0)
                self.next_id += 1

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

        self.subscription_color = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # RealSense에서 제공하는 컬러 이미지 토픽
            self.color_image_callback,
            10)
        
        self.bridge = CvBridge()
        self.tracker = RectangleTracker()  # 사각형 추적 객체 추가

    def color_image_callback(self, msg):

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        h, w, _ = cv_image.shape

        roi_y_start = h * 2 // 6
        roi_y_end = h
        roi_x_start = w // 3
        roi_x_end = w * 2 // 3

        roi = cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        blr = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        h_, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv_image = cv2.merge((h_, s, v))

        hsv_upper_white = np.array([180, 40, 255])
        hsv_lower_white = np.array([0, 0, 70])
        hsv_mask = cv2.inRange(hsv_image, hsv_lower_white, hsv_upper_white)

        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect_image = roi.copy()
        polygon_image = roi.copy()

        detected_rects = []

        for cnt in contours:
            epsilon = 0.08 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if 20 <= w <= 120 and 20 <= h <= 120:
                    area = cv2.contourArea(approx)
                    if area > 500:
                        angles = []
                        for i in range(4):
                            p1, p2, p3 = approx[i - 1][0], approx[i][0], approx[(i + 1) % 4][0]
                            v1 = np.array(p1) - np.array(p2)
                            v2 = np.array(p3) - np.array(p2)
                            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                            angles.append(angle)

                        if all(60 <= angle <= 120 for angle in angles):
                            cv2.drawContours(polygon_image, [approx], 0, (0, 0, 255), 2)
                            detected_rects.append((x, y, w, h, 0))  # 추적을 위해 추가

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            width, height = rect[1]

            if 20 <= width <= 120 and 20 <= height <= 120:
                cv2.drawContours(rect_image, [box], 0, (0, 255, 0), 2)
                detected_rects.append((rect[0][0], rect[0][1], width, height, rect[2]))  # 추적을 위해 추가

        tracked_rects = self.tracker.update(detected_rects)

        for id, (x, y, w, h, angle, lost) in tracked_rects.items():
            if lost == 0:
                cv2.putText(rect_image, f'ID {id}', (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(rect_image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 255, 255), 2)

        cv2.imshow("Only HSV White", hsv_mask)
        cv2.imshow("Contours White", rect_image)
        cv2.imshow("Contours Rectangle", polygon_image)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ImgSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

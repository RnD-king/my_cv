import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import torch
import cv2

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')     
        self.subscription = self.create_subscription( # 화면 받아오고
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        
        self.publisher_ = self.create_publisher(String, '/yolo_result', 10) # 욜로 결과 퍼블리시

        self.model = torch.hub.load( # 욜로 모델
            'ultralytics/yolov5',
            'custom',
            path='/home/noh/my_cv/ultralytics/yolov5/runs/train/line_yolo/weights/best.pt',
            source='local'
        )

        self.model.conf = 0.5  # 신뢰도
        self.bridge = CvBridge()

        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL) # 창 조절 가능

    def image_callback(self, msg):
        bgr_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # CV는 색공간을 BGR로 받아옴
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) # 욜로는 RBG로 보기 때문에 여기서 바꿔준다 (우리도)
        results = self.model(rgb_frame) # 그래서 욜로 모델도 RGB
        
        detections = results.pandas().xyxy[0]
        centers = []

        for _, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']
            if label == 'line': # 욜로 클래스 중 line만 처리
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                centers.append((cx, cy)) # 중심점 배열에 추가

                cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(bgr_frame, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(bgr_frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                self.publisher_.publish(String(data=f"{label} ({conf:.2f})"))

        if centers:
            self.get_logger().info(f"Centers: {centers}")

        cv2.imshow("YOLO Detection", bgr_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

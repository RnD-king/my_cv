import rclpy 
from rclpy.node import Node 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
import cv2


class ImgSubscriber(Node):
    def __init__(self):
        super().__init__('img_subscriber')
        self.subscription_color = self.create_subscription(  # 컬러
            Image,
            '/camera/camera/color/image_raw',  # RealSense에서 제공하는 컬러 이미지 토픽
            self.color_image_callback, 10)        
        self.bridge = CvBridge()

    def color_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
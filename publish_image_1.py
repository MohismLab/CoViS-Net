#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/robomaster_1/camera_0/image_proc', 10)
        timer_period = 0.5  # 發送頻率：0.5秒發一次
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.br = CvBridge()
        
        # 讀取圖片
        self.img_path = '/home/ysy/shiyuan_ws/CoViS-Net/test2.png'  # <-- 這裡換成你的圖片路徑
        self.cv_image = cv2.imread(self.img_path)
        
        if self.cv_image is None:
            self.get_logger().error(f'Failed to load image: {self.img_path}')
            rclpy.shutdown()

    def timer_callback(self):
        # 建立Image消息
        msg = self.br.cv2_to_imgmsg(self.cv_image, encoding="bgr8")
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing image...')

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

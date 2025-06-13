#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge
import os

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/robomaster_0/camera_0/image_proc', 10)
        timer_period = 0.5  
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.br = CvBridge()
        
   
        self.img_path = os.path.join(os.path.dirname(__file__), 'test_img/6-2.png')  
        self.cv_image = cv2.imread(self.img_path)
        
        if self.cv_image is None:
            self.get_logger().error(f'Failed to load image: {self.img_path}')
            rclpy.shutdown()

    def timer_callback(self):
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

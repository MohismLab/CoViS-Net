import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import tty
import termios
import argparse

# Constants
LINEAR_SPEED = 0.2
ANGULAR_SPEED = 2.0

class KeyboardPublisher(Node):
    def __init__(self, topic_name):
        super().__init__('keyboard_cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, topic_name, 10)
        self.get_logger().info(f'Keyboard Publisher Initialized, publishing to {topic_name}')

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def run(self):
        self.get_logger().info('Use keys: [W] forward, [S] back, [A] left, [D] right, [Q/E] rotate, [SPACE] stop, [c] to quit')
        twist = Twist()
        try:
            while rclpy.ok():
                key = self.get_key().lower()
                twist = Twist()  # reset every time

                if key == 'w':
                    twist.linear.x = LINEAR_SPEED
                elif key == 's':
                    twist.linear.x = -LINEAR_SPEED
                elif key == 'a':
                    twist.linear.y = LINEAR_SPEED
                elif key == 'd':
                    twist.linear.y = -LINEAR_SPEED
                elif key == 'q':
                    twist.angular.z = ANGULAR_SPEED
                elif key == 'e':
                    twist.angular.z = -ANGULAR_SPEED
                elif key == ' ':
                    twist = Twist()  # stop
                elif key == 'c':
                    return
                else:
                    continue  # ignore unknown keys

                self.publisher_.publish(twist)

        except KeyboardInterrupt:
            self.get_logger().info('Keyboard control interrupted.')
            stop_twist = Twist()
            self.publisher_.publish(stop_twist)
            raise  

def main():
    parser = argparse.ArgumentParser(description='Keyboard control node for ROS2')
    parser.add_argument('--topic', type=str, default='/s1_0/twist',
                      help='Topic name to publish Twist messages (default: /s1_0/twist)')
    parsed_args = parser.parse_args()
    
    rclpy.init()
    node = KeyboardPublisher(parsed_args.topic)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

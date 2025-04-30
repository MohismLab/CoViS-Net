import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import tty
import termios

class KeyboardPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_cmd_vel_publisher')
        self.publisher_ = self.create_publisher(Twist, '/agent0/cmd_vel', 10)
        self.get_logger().info('Keyboard Publisher Initialized')

    def get_key(self):
        # 獲取鍵盤按鍵
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def run(self):
        self.get_logger().info('Use keys: [W] forward, [S] back, [A] left, [D] right, [Q/E] rotate, [SPACE] stop, [Ctrl+C] to quit')
        twist = Twist()
        try:
            while rclpy.ok():
                key = self.get_key().lower()
                twist = Twist()  # reset every time

                if key == 'w':
                    twist.linear.x = 0.3
                elif key == 's':
                    twist.linear.x = -0.3
                elif key == 'a':
                    twist.linear.y = 0.3
                elif key == 'd':
                    twist.linear.y = -0.3
                elif key == 'q':
                    twist.angular.z = 0.3
                elif key == 'e':
                    twist.angular.z = -0.3
                elif key == ' ':
                    twist = Twist()  # stop
                else:
                    continue  # ignore unknown keys

                self.publisher_.publish(twist)

        except KeyboardInterrupt:
            self.get_logger().info('Keyboard control interrupted.')

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardPublisher()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

from pathlib import Path

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from std_msgs.msg import String

PROJECT_ROOT = Path(__file__).parent.parent


class TestPublisher(Node):
    def __init__(self, topic_name: str = "test_topic", qos_depth: int = 10) -> None:
        super().__init__("test_publisher")
        self._publisher: Publisher = self.create_publisher(
            String, topic_name, qos_depth
        )
        self._counter = 0

    def publish(self, data: str) -> None:
        msg = String()
        msg.data = data
        self._publisher.publish(msg)
        self._counter += 1

    def publish_test_message(self) -> None:
        self.publish("test message")


class TestSubscriber(Node):
    def __init__(self, topic_name: str = "test_topic", qos_depth: int = 10) -> None:
        super().__init__("test_subscriber")
        self._subscription: Subscription = self.create_subscription(
            String, topic_name, self._message_callback, qos_depth
        )
        self.last_msg: str | None = None
        self.received_count = 0

    def _message_callback(self, msg: String) -> None:
        self.last_msg = msg.data
        self.received_count += 1

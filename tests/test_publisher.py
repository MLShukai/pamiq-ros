from pamiq_ros.publisher_member_function import MinimalPublisher


def test_publisher_creation(rclpy_init):
    publisher = MinimalPublisher()
    assert publisher.get_name() == "minimal_publisher"

from pamiq_ros.subscriber_member_function import MinimalSubscriber


def test_subscriber_creation(rclpy_init):
    publisher = MinimalSubscriber()
    assert publisher.get_name() == "minimal_subscriber"

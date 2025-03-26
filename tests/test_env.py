import pytest
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

from pamiq_ros.env import ROS2Environment
from tests.helpers import TestPublisher, TestSubscriber


class TestROS2Environment:
    @pytest.fixture
    def ros2_env(self):
        """ROS2Environment fixture."""
        env = ROS2Environment(
            node_name="test_env_node",
            obs_topic_name="/test_obs_topic",
            obs_msg_type=String,
            action_topic_name="/test_action_topic",
            action_msg_type=String,
            qos=10,
            obs_get_timeout=1.0,
        )
        env.setup()
        try:
            yield env
        finally:
            env.teardown()

    @pytest.fixture
    def executor(self) -> SingleThreadedExecutor:
        """Executor fixture."""
        return SingleThreadedExecutor()

    @pytest.fixture
    def obs_publisher(self, executor: SingleThreadedExecutor) -> TestPublisher:
        """Observation publisher fixture."""
        pub = TestPublisher("/test_obs_topic", 10)
        executor.add_node(pub)
        return pub

    @pytest.fixture
    def action_subscriber(self, executor: SingleThreadedExecutor) -> TestSubscriber:
        """Action subscriber fixture with message tracking."""
        sub = TestSubscriber("/test_action_topic")
        executor.add_node(sub)
        return sub

    def test_observe_without_setup(self):
        """Test that observe raises RuntimeError when called before setup."""

        env = ROS2Environment(
            node_name="test_env_node",
            obs_topic_name="/test_obs_topic",
            obs_msg_type=String,
            action_topic_name="/test_action_topic",
            action_msg_type=String,
            qos=10,
            obs_get_timeout=1.0,
        )
        with pytest.raises(RuntimeError, match="Please call `setup` before."):
            env.observe()

    def test_observe_timeout(self, ros2_env):
        """Test that observe raises TimeoutError when no data is received."""

        with pytest.raises(
            TimeoutError, match="Timed out waiting for observation on topic"
        ):
            ros2_env.observe()

    def test_observe(
        self,
        ros2_env: ROS2Environment[String, String],
        executor: SingleThreadedExecutor,
        obs_publisher: TestPublisher,
        action_subscriber: TestSubscriber,
    ):
        obs_publisher.publish_test_message()
        executor.spin_once(0.1)
        observation = ros2_env.observe()
        assert observation.data == "test message"

    def test_affect(
        self,
        ros2_env: ROS2Environment[String, String],
        executor: SingleThreadedExecutor,
        action_subscriber: TestSubscriber,
    ):
        assert action_subscriber.last_msg is None
        assert action_subscriber.received_count == 0

        msg = String()
        msg.data = "test action"
        ros2_env.affect(msg)
        executor.spin_once(0.1)
        assert action_subscriber.last_msg == "test action"
        assert action_subscriber.received_count == 1

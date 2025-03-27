import threading
from collections.abc import Generator

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from pamiq_ros.env import (
    CachedObsROS2Environment,
    ReactiveROS2Environment,
    ROS2Environment,
)
from tests.helpers import TestPublisher, TestSubscriber


@pytest.fixture
def executor() -> SingleThreadedExecutor:
    """Executor fixture."""
    return SingleThreadedExecutor()


@pytest.fixture
def obs_publisher(executor: SingleThreadedExecutor) -> TestPublisher:
    """Observation publisher fixture."""
    pub = TestPublisher("/test_obs_topic", 10)
    executor.add_node(pub)
    return pub


@pytest.fixture
def action_subscriber(executor: SingleThreadedExecutor) -> TestSubscriber:
    """Action subscriber fixture."""
    sub = TestSubscriber("/test_action_topic")
    executor.add_node(sub)
    return sub


class TestROS2Environment:
    """Tests for ROS2Environment using CachedObsROS2Environment
    implementation."""

    @pytest.fixture
    def env(self) -> Generator[ROS2Environment[String, String], None, None]:
        """CachedObsROS2Environment fixture."""
        env = CachedObsROS2Environment(
            node_name="test_env_node",
            obs_topic_name="/test_obs_topic",
            obs_msg_type=String,
            action_topic_name="/test_action_topic",
            action_msg_type=String,
            qos=10,
            obs_get_timeout=1.0,
        )
        yield env

    @pytest.fixture
    def setup_env(self, env):
        """Setup and teardown for minimal environment."""
        env.setup()
        try:
            yield env
        finally:
            env.teardown()

    def test_properties(self, env: ROS2Environment):
        """Test property accessors."""
        assert env.obs_topic_name == "/test_obs_topic"
        assert env.action_topic_name == "/test_action_topic"
        assert env.obs_msg_type == String
        assert env.action_msg_type == String
        assert isinstance(env.node, Node)

    def test_affect(
        self,
        setup_env: ROS2Environment,
        executor: SingleThreadedExecutor,
        action_subscriber: TestSubscriber,
    ):
        """Test publishing action."""
        assert action_subscriber.last_msg is None
        assert action_subscriber.received_count == 0

        msg = String()
        msg.data = "test action"
        setup_env.affect(msg)
        executor.spin_once(0.1)
        assert action_subscriber.last_msg == "test action"
        assert action_subscriber.received_count == 1


class TestCachedObsROS2Environment:
    """Tests for CachedObsROS2Environment."""

    @pytest.fixture
    def cached_env(self):
        """CachedObsROS2Environment fixture."""
        env = CachedObsROS2Environment(
            node_name="test_cached_env_node",
            obs_topic_name="/test_obs_topic",
            obs_msg_type=String,
            action_topic_name="/test_action_topic",
            action_msg_type=String,
            qos=10,
            obs_get_timeout=0.1,
        )
        yield env

    @pytest.fixture
    def setup_cached_env(self, cached_env: CachedObsROS2Environment):
        """Setup and teardown for cached environment."""
        cached_env.setup()
        try:
            yield cached_env
        finally:
            cached_env.teardown()

    def test_observe_timeout(self, setup_cached_env: CachedObsROS2Environment):
        """Test observation timeout."""
        with pytest.raises(
            TimeoutError, match="Timed out waiting for observation on topic"
        ):
            setup_cached_env.observe()

    def test_observe_with_default(self):
        """Test observe with default observation."""
        default_msg = String()
        default_msg.data = "default message"

        env = CachedObsROS2Environment(
            node_name="test_default_env_node",
            obs_topic_name="/test_default_obs_topic",
            obs_msg_type=String,
            action_topic_name="/test_default_action_topic",
            action_msg_type=String,
            qos=10,
            default_observation=default_msg,
        )
        env.setup()
        try:
            observation = env.observe()
            assert observation.data == "default message"
        finally:
            env.teardown()

    def test_observe_caching(
        self,
        setup_cached_env: CachedObsROS2Environment[String, String],
        executor: SingleThreadedExecutor,
        obs_publisher: TestPublisher,
    ):
        """Test that observations are cached."""
        # Publish first message
        obs_publisher.publish_test_message()
        executor.spin_once(0.1)

        # Get observation, should be cached message
        observation1 = setup_cached_env.observe()
        assert observation1.data == "test message"

        # Create a new message with different data
        node = Node("test_publisher_node")
        publisher = node.create_publisher(String, "/test_obs_topic", 10)
        try:
            msg2 = String()
            msg2.data = "test message 2"
            publisher.publish(msg2)
            executor.add_node(node)
            executor.spin_once(0.1)

            # Get observation again, should be updated message
            observation2 = setup_cached_env.observe()
            assert observation2.data == "test message 2"
        finally:
            node.destroy_node()


class TestReactiveROS2Environment:
    """Tests for ReactiveROS2Environment."""

    @pytest.fixture
    def reactive_env(self) -> ReactiveROS2Environment[String, String]:
        """ReactiveROS2Environment fixture."""
        # Create default timeout observation
        timeout_obs = String()
        timeout_obs.data = "timeout observation"

        env = ReactiveROS2Environment(
            node_name="test_reactive_env_node",
            obs_topic_name="/test_obs_topic",
            obs_msg_type=String,
            action_topic_name="/test_action_topic",
            action_msg_type=String,
            qos=10,
            obs_get_timeout=0.1,  # Short timeout for tests
            timeout_obs=timeout_obs,
        )
        return env

    @pytest.fixture
    def setup_reactive_env(
        self, reactive_env: ReactiveROS2Environment
    ) -> Generator[ReactiveROS2Environment, None, None]:
        """Setup and teardown for reactive environment."""
        reactive_env.setup()
        try:
            yield reactive_env
        finally:
            reactive_env.teardown()

    def test_observe_timeout(
        self, setup_reactive_env: ReactiveROS2Environment[String, String]
    ):
        """Test observation timeout returns timeout_obs."""
        # Should return timeout observation since no message will be published
        observation = setup_reactive_env.observe()
        assert observation.data == "timeout observation"

    def test_observe_with_message(
        self,
        setup_reactive_env: ReactiveROS2Environment[String, String],
        executor: SingleThreadedExecutor,
        obs_publisher: TestPublisher,
    ):
        """Test that observation returns when message arrives."""

        # Set up background thread to publish message after short delay
        def delayed_publish():
            obs_publisher.publish("reactive test message")
            executor.spin_once(0.05)

        # Start background thread and call observe
        timer = threading.Timer(0.05, delayed_publish)
        timer.start()

        # This should block until message is received or timeout
        observation = setup_reactive_env.observe()

        # Wait for background thread to complete
        timer.join()

        # Should receive the published message, not timeout
        assert observation.data == "reactive test message"

    def test_observe_thread_termination(
        self, setup_reactive_env: ReactiveROS2Environment[String, String]
    ):
        """Test behavior when ROS2 thread terminates."""
        # Manually terminate the thread
        rclpy.shutdown()

        # Should return timeout observation
        observation = setup_reactive_env.observe()
        assert observation.data == "timeout observation"

    def test_observe_before_setup(self, reactive_env: ReactiveROS2Environment):
        """Test calling observe before setup raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Environment not set up"):
            reactive_env.observe()

    def test_next_observation_handling(
        self,
        setup_reactive_env: ReactiveROS2Environment[String, String],
        executor: SingleThreadedExecutor,
        obs_publisher: TestPublisher,
    ):
        """Test the handling of _next_observation attribute."""
        # Publish a message
        obs_publisher.publish("first message")
        executor.spin_once(0.1)

        # First observation should get the message
        observation1 = setup_reactive_env.observe()
        assert observation1.data == "first message"

        # _next_observation should be reset to None after observe()
        assert setup_reactive_env.has_new_observation() is False

        # Next observe call should timeout without new messages
        observation2 = setup_reactive_env.observe()
        assert observation2.data == "timeout observation"

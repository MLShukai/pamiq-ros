import logging
import threading
from abc import abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, override

import rclpy
from pamiq_core.interaction import Environment
from pamiq_core.utils.reflection import get_class_module_path
from rclpy.node import Node
from rclpy.qos import QoSProfile

type Kwds = Mapping[str, Any]
DEFAULT_KWDS: Kwds = MappingProxyType({})


class ROS2Environment[Obs, Act](Environment[Obs, Act]):
    """Base class for ROS2 environment implementations.

    This abstract base class provides common functionality for ROS2
    integration with pamiq-core environments. It sets up the ROS2 node,
    publishers, and subscribers, and provides a common interface for
    observations and actions.

    Subclasses implement specific message receiving strategies.
    """

    def __init__(
        self,
        node_name: str,
        obs_topic_name: str,
        obs_msg_type: type[Obs],
        action_topic_name: str,
        action_msg_type: type[Act],
        qos: int | QoSProfile,
        node_kwds: Kwds = DEFAULT_KWDS,
        obs_kwds: Kwds = DEFAULT_KWDS,
        action_kwds: Kwds = DEFAULT_KWDS,
    ) -> None:
        """Initialize ROS2 environment base class.

        Args:
            node_name: Name of the ROS2 node
            obs_topic_name: Topic name for observations
            obs_msg_type: Message type for observations
            action_topic_name: Topic name for actions
            action_msg_type: Message type for actions
            qos: Quality of Service profile or depth
            node_kwds: Additional keyword arguments for node creation
            obs_kwds: Additional keyword arguments for subscription
            action_kwds: Additional keyword arguments for publisher
        """
        super().__init__()
        self._node = rclpy.create_node(node_name, **node_kwds)
        self._obs_subscriber = self._node.create_subscription(  # pyright: ignore[reportUnknownMemberType]
            obs_msg_type,
            obs_topic_name,
            self._obs_callback,
            qos,
            **obs_kwds,
        )
        self._action_publisher = self._node.create_publisher(  # pyright: ignore[reportUnknownMemberType]
            action_msg_type, action_topic_name, qos, **action_kwds
        )

        self._logger = logging.getLogger(get_class_module_path(self.__class__))
        self._obs_thread: threading.Thread | None = None

    @abstractmethod
    def _create_obs_thread(self) -> threading.Thread:
        """Create the observation thread that spins the ROS2 node.

        This method must be implemented by subclasses to define how the
        observation thread should be created and configured.

        Returns:
            A thread that will spin the ROS2 node when started
        """
        ...

    @property
    def node(self) -> Node:
        """Get the ROS2 node.

        Returns:
            The ROS2 node used by this environment
        """
        return self._node

    @property
    def obs_topic_name(self) -> str:
        """Get the observation topic name.

        Returns:
            The name of the observation topic
        """
        return self._obs_subscriber.topic_name  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    @property
    def obs_msg_type(self) -> type[Obs]:
        """Get the observation message type.

        Returns:
            The type of the observation messages
        """
        return self._obs_subscriber.msg_type  # pyright: ignore[reportReturnType]

    @property
    def action_topic_name(self) -> str:
        """Get the action topic name.

        Returns:
            The name of the action topic
        """
        return self._action_publisher.topic_name

    @property
    def action_msg_type(self) -> type[Act]:
        """Get the action message type.

        Returns:
            The type of the action messages
        """
        return self._action_publisher.msg_type  # pyright: ignore[reportReturnType]

    @abstractmethod
    def _obs_callback(self, data: Obs) -> None:
        """Process an observation message from ROS2.

        This method is called whenever a new message is received on the
        observation topic. Subclasses must implement this to define how
        to handle incoming messages.

        Args:
            data: Observation data from ROS2
        """
        ...

    @override
    def affect(self, action: Act) -> None:
        """Publish action to ROS2.

        Args:
            action: Action message to publish
        """
        self._action_publisher.publish(action)

    @override
    def setup(self) -> None:
        """Set up ROS2 node and start spinning it in a separate thread."""
        super().setup()
        self._obs_thread = self._create_obs_thread()
        self._obs_thread.start()

    @override
    def teardown(self) -> None:
        """Clean up ROS2 resources."""
        super().teardown()
        self._node.destroy_node()
        self._obs_thread = None


class CachedObsROS2Environment[Obs, Act](ROS2Environment[Obs, Act]):
    """ROS2 environment that caches the latest observation.

    This implementation maintains the most recent observation received
    from a ROS2 topic. The observe() method returns the latest cached
    message immediately or waits for the first message to arrive.
    """

    def __init__(
        self,
        node_name: str,
        obs_topic_name: str,
        obs_msg_type: type[Obs],
        action_topic_name: str,
        action_msg_type: type[Act],
        qos: int | QoSProfile,
        node_kwds: Kwds = DEFAULT_KWDS,
        obs_kwds: Kwds = DEFAULT_KWDS,
        action_kwds: Kwds = DEFAULT_KWDS,
        obs_get_timeout: float | None = None,
        default_obs: Obs | None = None,
    ) -> None:
        """Initialize cached ROS2 environment.

        Args:
            node_name: Name of the ROS2 node
            obs_topic_name: Topic name for observations
            obs_msg_type: Message type for observations
            action_topic_name: Topic name for actions
            action_msg_type: Message type for actions
            qos: Quality of Service profile or depth
            node_kwds: Additional keyword arguments for node creation
            obs_kwds: Additional keyword arguments for subscription
            action_kwds: Additional keyword arguments for publisher
            obs_get_timeout: Timeout for waiting for observations (None means wait forever)
            default_obs: Default observation to use if no message has been received yet
        """
        super().__init__(
            node_name,
            obs_topic_name,
            obs_msg_type,
            action_topic_name,
            action_msg_type,
            qos,
            node_kwds,
            obs_kwds,
            action_kwds,
        )
        self._obs_condition = threading.Condition()
        self._obs_get_timeout = obs_get_timeout
        self._observation = default_obs

    @property
    def _has_observation(self) -> bool:
        return self._observation is not None

    @override
    def _create_obs_thread(self) -> threading.Thread:
        """Create the observation thread for the cached environment.

        Returns:
            A daemon thread that spins the ROS2 node
        """
        return threading.Thread(target=lambda: rclpy.spin(self.node), daemon=True)

    @override
    def _obs_callback(self, data: Obs) -> None:
        """Process an observation message from ROS2.

        Updates the cached observation and notifies waiting threads.

        Args:
            data: Observation data from ROS2
        """
        with self._obs_condition:
            self._observation = data
            self._obs_condition.notify_all()

    @override
    def observe(self) -> Obs:
        """Get the latest observation from ROS2.

        This method waits for the first observation to arrive if none has been
        received yet and no default was provided. After the first observation,
        it returns the most recent observation immediately.

        Returns:
            Current observation from ROS2

        Raises:
            RuntimeError: If called before setup
            TimeoutError: If observation not received within timeout
            RuntimeError: If observation is None despite being marked as available
        """
        if self._obs_thread is None or not self._obs_thread.is_alive():
            raise RuntimeError(
                "Environment not set up. Call `setup()` before observe()."
            )

        with self._obs_condition:
            # Wait for observation if none has been received yet
            if not self._has_observation:
                if not self._obs_condition.wait(timeout=self._obs_get_timeout):
                    raise TimeoutError(
                        f"Timed out waiting for observation on topic '{self.obs_topic_name}'"
                    )

            # At this point, we should have an observation
            if self._observation is None:
                raise RuntimeError(
                    "Observation is None despite being marked as available"
                )

            return self._observation


class ReactiveROS2Environment[Obs, Act](ROS2Environment[Obs, Act]):
    """ROS2 environment that waits for a new message on each observe call.

    This implementation blocks on each observe() call until a new
    message is received from the ROS2 topic or until the timeout is
    reached.
    """

    def __init__(
        self,
        node_name: str,
        obs_topic_name: str,
        obs_msg_type: type[Obs],
        action_topic_name: str,
        action_msg_type: type[Act],
        qos: int | QoSProfile,
        obs_get_timeout: float,
        timeout_obs: Obs,
        node_kwds: Kwds = DEFAULT_KWDS,
        obs_kwds: Kwds = DEFAULT_KWDS,
        action_kwds: Kwds = DEFAULT_KWDS,
    ) -> None:
        """Initialize reactive ROS2 environment.

        Args:
            node_name: Name of the ROS2 node
            obs_topic_name: Topic name for observations
            obs_msg_type: Message type for observations
            action_topic_name: Topic name for actions
            action_msg_type: Message type for actions
            qos: Quality of Service profile or depth
            obs_get_timeout: Timeout for waiting for a new observation
            timeout_obs: Observation to return if timeout occurs
            node_kwds: Additional keyword arguments for node creation
            obs_kwds: Additional keyword arguments for subscription
            action_kwds: Additional keyword arguments for publisher
        """
        super().__init__(
            node_name,
            obs_topic_name,
            obs_msg_type,
            action_topic_name,
            action_msg_type,
            qos,
            node_kwds,
            obs_kwds,
            action_kwds,
        )
        self._obs_condition = threading.Condition()
        self._obs_get_timeout = obs_get_timeout
        self._timeout_obs = timeout_obs
        self._next_observation: Obs | None = None
        self._thread_running = False

    def has_new_observation(self) -> bool:
        return self._next_observation is not None

    @override
    def _create_obs_thread(self) -> threading.Thread:
        """Create the observation thread with monitoring for thread
        termination.

        Returns:
            A daemon thread that spins the ROS2 node and monitors its state
        """

        def obs_thread_func():
            self._thread_running = True

            try:
                rclpy.spin(self._node)
            finally:
                # Important: Release any waiting observe methods when thread terminates
                with self._obs_condition:
                    self._thread_running = False
                    self._obs_condition.notify_all()
                self._logger.info(
                    f"ROS2 spin thread for node '{self._node.get_name()}' has terminated"
                )

        thread = threading.Thread(target=obs_thread_func)
        thread.daemon = True
        return thread

    @override
    def _obs_callback(self, data: Obs) -> None:
        """Process an observation message from ROS2.

        Stores the new message and notifies waiting threads.

        Args:
            data: Observation data from ROS2
        """
        with self._obs_condition:
            self._next_observation = data
            self._obs_condition.notify_all()

    @override
    def observe(self) -> Obs:
        """Wait for and return the next observation from ROS2.

        This method blocks until a new message is received on the topic
        or the timeout is reached.

        Returns:
            Next observation from ROS2 or timeout observation if timeout occurs

        Raises:
            RuntimeError: If called before setup
            RuntimeError: If received observation is None despite notification
        """
        if self._obs_thread is None or not self._obs_thread.is_alive():
            raise RuntimeError(
                "Environment not set up. Call `setup()` before observe()."
            )

        with self._obs_condition:
            # Timeout immediately if thread is not running
            if not self._thread_running:
                self._logger.error(
                    f"ROS2 spin thread is not running, cannot receive observations from '{self.obs_topic_name}'"
                )
                return self._timeout_obs

            # Wait for new observation or thread termination
            if not self.has_new_observation():
                wait_result = self._obs_condition.wait(timeout=self._obs_get_timeout)

                # Check if thread terminated during wait
                if not self._thread_running:
                    self._logger.info(
                        f"ROS2 spin thread terminated while waiting for observation on '{self.obs_topic_name}'"
                    )
                    return self._timeout_obs

                # Handle timeout case
                if not wait_result and not self.has_new_observation():
                    self._logger.warning(
                        f"Timed out waiting for observation on topic '{self.obs_topic_name}'"
                    )
                    return self._timeout_obs

            # If _has_new_observation is True, _next_observation must not be None
            if self._next_observation is None:
                raise RuntimeError("Received observation is None despite notification")

            observation = self._next_observation
            self._next_observation = None
            return observation

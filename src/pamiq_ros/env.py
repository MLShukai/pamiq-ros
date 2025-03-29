import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, override

import rclpy
from pamiq_core.interaction import Environment
from rclpy.node import Node
from rclpy.qos import QoSProfile

type Kwds = Mapping[str, Any]
DEFAULT_KWDS: Kwds = MappingProxyType({})


class ROS2Environment[Obs, Act](Environment[Obs, Act]):
    """ROS2 environment implementation for pamiq-core.

    This class provides an interface between pamiq-core environment
    abstractions and ROS2 functionality. It manages a ROS2 node with
    subscribers for observations and publishers for actions.

    The environment maintains the latest observation received from the
    specified topic and provides methods to observe the current state
    and affect the environment through action publishing.
    """

    def __init__(
        self,
        node_name: str,
        obs_topic_name: str,
        obs_msg_type: type[Obs],
        action_topic_name: str,
        action_msg_type: type[Act],
        initial_obs: Obs,
        qos: int | QoSProfile,
        node_kwds: Kwds = DEFAULT_KWDS,
        obs_kwds: Kwds = DEFAULT_KWDS,
        action_kwds: Kwds = DEFAULT_KWDS,
        new_obs_timeout: float | None = 0.0,
    ) -> None:
        """Initialize ROS2 environment.

        Args:
            node_name: Name of the ROS2 node
            obs_topic_name: Topic name for observations
            obs_msg_type: Message type for observations
            action_topic_name: Topic name for actions
            action_msg_type: Message type for actions
            initial_obs: Initial observation value
            qos: Quality of Service profile or depth
            node_kwds: Additional keyword arguments for node creation
            obs_kwds: Additional keyword arguments for subscription
            action_kwds: Additional keyword arguments for publisher
            new_obs_timeout: Timeout in seconds for waiting on new observations (None means no timeout)
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
        self._obs = initial_obs
        self._obs_cv = threading.Condition()
        self._new_obs_timeout = new_obs_timeout
        self._has_new_observation = False

        self._obs_thread: threading.Thread | None = None

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

    def _obs_callback(self, data: Obs) -> None:
        """Process an observation message from ROS2.

        Updates the stored observation with thread-safe access and marks
        that a new observation has been received.

        Args:
            data: Observation data from ROS2
        """
        with self._obs_cv:
            self._obs = data
            self._has_new_observation = True
            self._obs_cv.notify_all()

    @override
    def observe(self) -> Obs:
        """Get the current observation from ROS2.

        If a new observation has been received since the last call, it returns
        immediately. Otherwise, if obs_timeout is set, this method will wait
        for a new observation for the specified duration. If no new observation
        arrives within the timeout period, it returns the most recent observation.

        Returns:
            Current observation from ROS2

        Raises:
            RuntimeError: If called before setup
        """
        if self._obs_thread is None or not self._obs_thread.is_alive():
            raise RuntimeError(
                "Environment not set up. Call `setup()` before observe()."
            )

        with self._obs_cv:
            if not self._has_new_observation:
                self._obs_cv.wait(self._new_obs_timeout)
            self._has_new_observation = False
            return self._obs

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

        def obs_thread():
            rclpy.spin(self.node)
            with self._obs_cv:
                self._obs_cv.notify_all()  # release obs condition.

        self._obs_thread = threading.Thread(target=obs_thread)
        self._obs_thread.start()

    @override
    def teardown(self) -> None:
        """Clean up ROS2 resources."""
        super().teardown()
        rclpy.try_shutdown()  # pyright: ignore[reportUnknownMemberType, reportPrivateImportUsage]
        self._node.destroy_node()
        if self._obs_thread is not None:
            self._obs_thread.join()
            self._obs_thread = None

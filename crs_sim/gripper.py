"""
Simulated gripper with the same interface as ``ctu_crs.gripper.Gripper``.
"""
from __future__ import annotations


class Gripper:
    """
    Keeps track of the commanded position only, it is not rendered.
    """
    def __init__(self, bounds=None, axis=None) -> None:
        self.bounds = bounds
        self._axis = axis
        self.position = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the gripper controller."""
        self._initialized = True

    def control_position_relative(self, fraction: float) -> None:
        """Control the gripper by relative position 0 = bounds[0], 1 = bounds[1]"""
        assert self._initialized, "Gripper controller must be initialized first."
        target = int(self.bounds[0] + (self.bounds[1] - self.bounds[0]) * fraction)
        self.control_position(target)

    def control_position(self, position: float) -> None:
        """Control the gripper by absolute position."""
        assert self._initialized, "Gripper controller must be initialized first."
        self.position = position

    def release(self) -> None:
        """Release the gripper."""
        assert self._initialized, "Gripper controller must be initialized first."

    def wait_for_motion_stop(self) -> bool:
        """The simulated gripper moves instantly."""
        assert self._initialized, "Gripper controller must be initialized first."
        return True

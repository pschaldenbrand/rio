# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from rio_hw import time

from ._policy import Policy


class Dummy(Policy):
    """A simple dummy policy that generates actions moving in a small circle."""

    def __init__(
        self,
        *args,
        radius: float = 0.2,
        speed: float = 0.1,
        action_dim: int = 7,
        chunk_size: int = 50,
        **kwargs,
    ):
        """
        Args:
            radius: Radius of the circular motion (in meters)
            speed: Speed of circular motion (radians per step)
            action_dim: Dimension of action space
            chunk_size: Number of actions to return per inference call
        """
        self.radius = radius
        self.speed = speed
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    def construct_policy(self, *args, **kwargs):
        """Initialize the policy. For dummy policy, just reset timestep."""
        pass

    def set_instruction(self, instruction):
        """Set instruction (unused for dummy policy)."""
        pass

    def inference(self, observation, current_plan=None):
        """
        Generate action chunk with circular motion pattern.

        Args:
            observation: Dictionary of observations (unused)
            current_plan: Current plan (unused)

        Returns:
            Action chunk of shape (chunk_size, action_dim)
        """
        actions = []
        current_time = time.now()  # Use current time as timestep base

        for i in range(self.chunk_size):
            # Generate circular motion for x, y coordinates
            angle = (current_time + i / self.chunk_size) * self.speed * 2 * np.pi
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            z = 0.0  # Keep z constant

            # Create action: [x, y, z, roll, pitch, yaw, gripper]
            action = np.zeros(self.action_dim, dtype=np.float32)
            action[0] = x
            action[1] = y
            action[2] = z
            # Rotations and gripper stay at 0

            actions.append(action)

        return np.array(actions, dtype=np.float32)

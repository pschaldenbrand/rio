# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R


def display_frame(name: str, eef_pose: np.ndarray, color_alpha=1.0, axis_length: float = 0.1):
    """Display the pose in Rerun as a coordinate frame."""
    # eef_pose is typically [x, y, z, rx, ry, rz] or [x, y, z, qx, qy, qz, qw]

    position = eef_pose[:3]
    ori = eef_pose[3:]
    # Convert orientation to rotation matrix if needed
    if len(ori) == 3:
        # Assume ori is in axis-angle (rx, ry, rz)
        rotation = R.from_rotvec(ori)
    elif len(ori) == 4:
        # Assume ori is in quaternion (qx, qy, qz, qw)
        rotation = R.from_quat(ori)
    else:
        raise ValueError("Unsupported orientation format in eef_pose.")
    # Get rotation matrix
    coord_frame = rotation.as_matrix()

    rr.log(
        f"{name}",
        rr.Transform3D(
            mat3x3=coord_frame,
            translation=position,
        ),
    )
    rr.log(
        f"{name}/axes",
        rr.Arrows3D(
            vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
            colors=[
                [int(255 * color_alpha), 0, 0],
                [0, int(255 * color_alpha), 0],
                [0, 0, int(255 * color_alpha)],
            ],
        ),
    )


def display_arrows_traj(name: str, traj: np.ndarray, color=None, arrow_length: float = 0.05, radii: float = 0.01):
    """Display a trajectory as arrows in Rerun."""
    if color is None:
        color = [255, 0, 0]
    positions = traj[:, :3]

    arrows = []
    colors = []

    rr.log(name, rr.LineStrips3D(positions, colors=color, radii=radii))

    if traj.shape[1] <= 3:
        return  # No orientation information

    orientations = traj[:, 3:]
    for ori in orientations:
        # Convert orientation to rotation matrix if needed
        if len(ori) == 3:
            rotation = R.from_rotvec(ori)
        elif len(ori) == 4:
            rotation = R.from_quat(ori)
        else:
            raise ValueError("Unsupported orientation format in trajectory.")

        rot_matrix = rotation.as_matrix()
        # Arrow points along the x-axis of the local frame
        arrow_vector = rot_matrix[:, 0] * arrow_length
        arrows.append(arrow_vector)
        colors.append(color)  # Red arrows

    rr.log(
        name,
        rr.Arrows3D(
            origins=positions,
            vectors=arrows,
            colors=colors,
            radii=radii,
        ),
    )


def compress_depth(depth: np.ndarray, max_depth: float = 10.0, downsample_factor: int = 1) -> np.ndarray:
    _depth = depth.copy()
    depth_clipped = np.clip(_depth, 0, max_depth)
    depth_uint16 = (depth_clipped / max_depth * 65535).astype(np.uint16)

    if downsample_factor > 1:
        depth_uint16 = depth_uint16[::downsample_factor, ::downsample_factor]

    return depth_uint16

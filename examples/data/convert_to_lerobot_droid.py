import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import robodm
import tyro
from loguru import logger
from PIL import Image

from rio.cfg.common import DatasetCfg

# DROID camera role names in assignment priority order: [wrist, exterior1, exterior2]
DROID_CAMERA_NAMES = ["wrist_image_left", "exterior_image_1_left", "exterior_image_2_left"]


def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def _load_first_trajectory(dataset_path: str) -> dict:
    """Load and return data from the first .vla file found at dataset_path."""
    path = Path(dataset_path).expanduser()
    vla_file = path if path.is_file() else next(iter(sorted(path.glob("*.vla"))), None)
    if vla_file is None:
        raise FileNotFoundError(f"No .vla files found in {path}")
    logger.info(f"Sampling features from {vla_file.name}...")
    return robodm.Trajectory(path=str(vla_file), mode="r").load()


def infer_action_features(
    data: dict,
    joint_key: str = "observation/proprio_joints",
    gripper_key: str = "observation/gripper_position",
) -> tuple[int, int]:
    """
    Infer num_joints and action_dim from loaded trajectory data.
    """
    joints = data.get(joint_key)
    if joints is None:
        raise KeyError(f"'{joint_key}' not found in trajectory. Keys: {list(data.keys())}")

    gripper = data.get(gripper_key)
    num_joints = joints.shape[-1]

    # gripper_position may be 1D (N,) for scalar gripper or 2D (N, K) for multi-dim
    gripper_dim = gripper.shape[1] if (gripper is not None and gripper.ndim > 1) else 1
    action_dim = num_joints + gripper_dim

    logger.info(f"Inferred: num_joints={num_joints}, action_dim={action_dim}")
    return num_joints, action_dim


def infer_camera_mapping(data: dict) -> dict[str, str]:
    """Infer camera mapping from loaded trajectory data."""
    camera_keys = sorted(k for k in data if k.startswith("observation/cameras/") and k.endswith("/rgb"))
    if not camera_keys:
        raise ValueError(f"No observation/cameras/*/rgb keys found. Available keys: {list(data.keys())}")

    sorted_cams = sorted(camera_keys, key=lambda k: data[k].shape[1] * data[k].shape[2])
    mapping = {cam: DROID_CAMERA_NAMES[i] for i, cam in enumerate(sorted_cams[: len(DROID_CAMERA_NAMES)])}
    logger.info(f"Inferred camera mapping: {mapping}")

    return mapping


def convert_to_lerobot(args: "Args"):
    """
    Convert robodm dataset to DROID-compatible LeRobot format.

    Args:
        args: Parsed configuration. args.camera_mapping, args.num_joints, and
              args.action_dim must already be populated (via inference or explicit CLI values).

    Transformations applied:
        1. Camera resize: all cameras → (image_height, image_width)
        2. Joint position truncation: raw joints → first num_joints dims
        3. Action computation: joint velocities (num_joints) + gripper (1) = action_dim
    """
    from rio.data import LeRobotFormatter

    robodm_path = args.input
    repo_id = args.repo_id or f"your_hf_username/{Path(robodm_path).stem}"
    output_path = str(Path(args.output).expanduser()) if args.output else None

    if output_path is None:
        output_path = str(Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id)
        logger.info(f"Using default LeRobot cache location: {output_path}")

    image_size = (args.image_width, args.image_height)  # PIL uses (W, H)
    camera_mapping = args.camera_mapping  # {robodm_key: droid_name}

    # Build feature_mapping from camera_mapping + fixed non-camera keys
    feature_mapping = {
        **camera_mapping,
        "observation/proprio_joints": "joint_position",
        "action": "actions",
        "instruction": "task",
    }

    # Build feature_transforms: resize+flip for all cameras, truncate joints, cast actions
    image_transform = lambda x: resize_image(x, image_size)
    feature_transforms = dict.fromkeys(camera_mapping.values(), image_transform)
    feature_transforms["joint_position"] = lambda x: x[: args.num_joints].astype(np.float32)
    feature_transforms["actions"] = lambda x: x.astype(np.float32)

    # Build features spec: one image entry per camera + joint_position + actions
    features = {
        droid_name: {
            "dtype": "image",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channel"],
        }
        for droid_name in camera_mapping.values()
    }
    features["joint_position"] = {
        "dtype": "float32",
        "shape": (args.num_joints,),
        "names": ["joint_position"],
    }
    features["actions"] = {
        "dtype": "float32",
        "shape": (args.action_dim,),  # num_joints velocities + 1 gripper
        "names": ["actions"],
    }

    num_joints = args.num_joints

    class DroidLeRobotFormatter(LeRobotFormatter):
        """Compute joint velocities from positions and zero-pad gripper actions."""

        # def _enhance_trajectory_data(self, traj_data: dict) -> dict:
        #     traj_data = super()._enhance_trajectory_data(traj_data)
        #     joint_positions = traj_data["observation/proprio_joints"]  # (N, D)

        #     gripper_positions = np.zeros_like(traj_data["action"][:, -1:])  # (N, 1), zeroed

        #     timesteps = traj_data["timestep"]  # (N,)
        #     dt = np.clip(timesteps[1:] - timesteps[:-1], a_min=1e-4, a_max=None)  # (N-1,)

        #     joint_velocities = (joint_positions[1:, :num_joints] - joint_positions[:-1, :num_joints]) / dt[:, None]
        #     joint_velocities = np.concatenate(
        #         [np.zeros((1, num_joints), dtype=np.float32), joint_velocities], axis=0
        #     )  # (N, num_joints)

        #     traj_data["action"] = np.concatenate([joint_velocities, gripper_positions], axis=-1).astype(
        #         np.float32
        #     )  # (N, action_dim)
        #     return traj_data


        def _enhance_trajectory_data(self, traj_data: dict) -> dict:
            traj_data = super()._enhance_trajectory_data(traj_data)
            raw_actions = traj_data["action"]  # (N, num_joints + 1): absolute joint targets + gripper

            joint_actions = raw_actions[:, :num_joints].astype(np.float32)   # (N, num_joints)
            gripper_actions = raw_actions[:, -1:].astype(np.float32)          # (N, 1), absolute

            traj_data["action"] = np.concatenate([joint_actions, gripper_actions], axis=-1)  # (N, action_dim)
            return traj_data

    formatter = DroidLeRobotFormatter(
        robodm_path=robodm_path,
        output_path=output_path,
        repo_id=repo_id,
        fps=args.fps,
        features=features,
        robot_type=args.robot_type,
        verbose=args.verbose,
        feature_mapping=feature_mapping,
        feature_transforms=feature_transforms,
        only_mapped_keys=True,
        video_keys=list(camera_mapping.values()),
        override_existing=True,
    )

    logger.info(f"Converting {robodm_path} to LeRobot format...")
    formatter.convert()
    logger.info(f"✓ Conversion complete! Output at: {output_path}")


def main(args: "Args"):
    """Main entry point for the dataset conversion script."""
    if args.format != "lerobot":
        logger.error("RLDS format conversion not yet implemented in this script.")
        sys.exit(1)

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Load first trajectory once, infer any missing config fields from it
    if args.num_joints is None or args.action_dim is None or args.camera_mapping is None:
        data = _load_first_trajectory(str(input_path))

        if args.num_joints is None or args.action_dim is None:
            inferred_joints, inferred_action_dim = infer_action_features(data)
            if args.num_joints is None:
                args.num_joints = inferred_joints
            else:
                logger.warning(f"Using provided num_joints={args.num_joints}")
            if args.action_dim is None:
                args.action_dim = inferred_action_dim
            else:
                logger.warning(f"Using provided action_dim={args.action_dim}")

        if args.camera_mapping is None:
            args.camera_mapping = infer_camera_mapping(data)
        else:
            logger.warning(f"Using provided camera_mapping={args.camera_mapping}")

    output_path = Path(args.output).expanduser() if args.output else None
    if args.clean and output_path is not None and output_path.exists():
        logger.warning(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)

    logger.info(f"Processing: {input_path}")
    convert_to_lerobot(args)


@dataclass
class Args(DatasetCfg):
    input: str = "/tmp/dummy_data/"
    output: str | None = None
    format: str = "lerobot"
    verbose: bool = False
    clean: bool = False
    # DROID-specific overrides
    robot_type: str = "xarm"


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

# Quick Start: The Robot Learning Workflow

We will walkthrough setting-up RIO for policy inference.

## Setup Your Config File

All examples are driven by a **station config** — a dataclass that declares every hardware and software node in your setup. See [`docs/setup/station_cfg.md`](station_cfg.md) for a full reference.

Start by copying the closest example config from `examples/cfg/` and editing the fields for your hardware:

```python
# examples/cfg/my_station.py
from dataclasses import dataclass, field
from rio.cfg import Camera, NodeCfg

@dataclass
class MyStation:
    arm: str = "XarmArm"
    arm_cfg: NodeCfg = field(
        default_factory=lambda: NodeCfg(
            robot_ip="192.168.1.205",   # your robot's IP
            robot_model="xarm7",
            robot_controller="joint_pos",
            freq=250,
        )
    )

    gripper: str | None = "RobotiqGripper"
    gripper_cfg: NodeCfg = field(
        default_factory=lambda: NodeCfg(port="/dev/ttyUSB0", model="robotiq_2f140")
    )

    cameras: dict[str, Camera] = field(
        default_factory=lambda: {
            "camera_1": Camera(
                cam_type="Realsense",
                serial="<your-serial>",
                model="D400",
                resolution=(480, 640),
            ),
        }
    )

    mw: str = "Thread"        # "Thread" or "Process"
    mp_method: str = "spawn"
    freq: int = 50
    instruction: str = ""
```

All examples use [tyro](https://github.com/brentyi/tyro) for CLI configuration, so fields in the dataclass can be overridden at the command line:

```bash
uv run -m examples.teleop_eef --freq 30
```

## Collecting

Use a teleoperation example to collect demonstrations. The recorder saves trajectories automatically when configured.

Add a recorder to your station config:

```python
from rio.cfg import NodeCfg

recorder: str | None = "Recorder"
recorder_cfg: NodeCfg = field(
    default_factory=lambda: NodeCfg(
        path="/data/rollouts/my_task/",
        video_codec="libx264",
    )
)
```

Then run teleoperation (see [examples below](#teleoperation)):

```bash
# EEF teleoperation with a Spacemouse
uv run -m examples.teleop_eef

# Joint-space teleoperation with Gello
uv run -m examples.teleop_gello
```

The recorder saves each episode to the configured path. Episodes can be replayed and visualized with `examples/replay_data.py`.


# Exporting data

Convert collected `.vla` trajectories to [LeRobot](https://github.com/huggingface/lerobot) format for training with the DROID pipeline (e.g. [openpi](https://github.com/Physical-Intelligence/openpi)).

The exporter targets the DROID LeRobot schema:
- Three named cameras: `wrist_image_left`, `exterior_image_1_left`, `exterior_image_2_left`
- `joint_position` — first `num_joints` dims of `observation/proprio_joints`
- `actions` — joint velocities (`num_joints`) concatenated with a zeroed gripper channel

## Configuring exporter

The exporter is configured via the `Args` dataclass, which extends `DatasetCfg`. All fields can be overridden at the command line with tyro.

**`DatasetCfg` fields** (shared base config, defined in `rio/cfg/common.py`):

| Field | Default | Description |
|-------|---------|-------------|
| `image_height` | `180` | Output image height in pixels |
| `image_width` | `320` | Output image width in pixels |
| `fps` | `50` | Dataset frames per second |
| `robot_type` | `"panda"` | Robot type label stored in the LeRobot metadata |
| `repo_id` | `"rio"` | HuggingFace repo ID for the dataset |
| `num_joints` | `None` | Number of joints to use; **inferred from data if `None`** |
| `action_dim` | `None` | Action dimensionality (`num_joints + gripper_dim`); **inferred if `None`** |
| `camera_mapping` | `None` | Dict mapping robodm camera keys → DROID names; **inferred if `None`** |

**Script-specific `Args` fields**:

| Field | Default | Description |
|-------|---------|-------------|
| `input` | `"/tmp/dummy_data/"` | Path to a `.vla` file or directory of `.vla` files |
| `output` | `None` | Output directory. Defaults to `~/.cache/huggingface/lerobot/{repo_id}` |
| `robot_type` | `"xarm"` | Overrides `DatasetCfg` default for xarm datasets |
| `verbose` | `False` | Enable verbose formatter logging |
| `clean` | `False` | Delete the output directory before converting |

**Auto-inference** — when `num_joints`, `action_dim`, or `camera_mapping` are left as `None`, the script loads the first trajectory in the dataset and infers them automatically:

- `num_joints` — last dimension of `observation/proprio_joints`
- `action_dim` — `num_joints` + gripper dimensions (1 for scalar gripper)
- `camera_mapping` — discovers all `observation/cameras/*/rgb` keys and assigns DROID names by ascending resolution: smallest pixel count → `wrist_image_left`, larger cameras → `exterior_image_1_left`, `exterior_image_2_left`

To override inference for a specific field, pass it explicitly:

```bash
uv run examples/data/convert_to_lerobot_droid.py \
    --input /data/my_task \
    --num-joints 7 \
    --action-dim 8
```

## Running

**Minimal — auto-infer everything:**

```bash
uv run examples/data/convert_to_lerobot_droid.py --input /data/rollouts/my_task/
```

Output lands in `~/.cache/huggingface/lerobot/rio` by default.

**Specify output path and repo ID:**

```bash
uv run examples/data/convert_to_lerobot_droid.py \
    --input /data/rollouts/my_task/ \
    --output /data/lerobot/my_task/ \
    --repo-id myuser/my_task
```

**Re-run from scratch (wipe existing output):**

```bash
uv run examples/data/convert_to_lerobot_droid.py \
    --input /data/rollouts/my_task/ \
    --output /data/lerobot/my_task/ \
    --clean
```

After conversion, compute normalization statistics required by the openpi training pipeline:

```bash
cd third_party/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_droid_finetune_test
```


## Run Policy Inference

1. **Set your checkpoint path** in `examples/cfg/pi05_xarm_coke_can.py`:
   ```python
   policy_cfg: PolicyConfig = field(
       default_factory=lambda: StationCfg.PolicyConfig(
           policy_path="ckpts/my_task",
           ...
       )
   )
   ```

2. **Run inference**:
   ```bash
   uv run -m examples.policy_inference
   ```

   Press **Enter** when prompted to start the inference loop. The policy runs at the configured frequency (default 50 Hz), requesting new action chunks from the policy server as the current chunk is consumed.

3. **Override the instruction** at runtime:
   ```bash
   uv run -m examples.policy_inference --instruction "Place the cup on the shelf."
   ```

---

# A Walkthrough of the examples

We include handy examples for teleoperation/data collection, camera streaming, data visualization and policy inference.

## Camera Streaming

Stream RGB frames from all configured cameras to a [Rerun](../visualization/rerun.md) visualizer.

```bash
uv run -m examples.stream_cameras
```

Cameras and visualizer are configured via the station config. This is a good first step to verify your camera setup before running teleoperation.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--freq` | `50` | Streaming frequency (Hz) |
| `--mw` | `Thread` | Middleware backend |

## Teleoperation

### End-Effector Teleoperation (`teleop_eef.py`)

Control the robot in Cartesian space using a Spacemouse, gamepad, or keyboard.

```bash
uv run -m examples.teleop_eef
# override teleop device:
uv run -m examples.teleop_eef --teleop Gamepad
uv run -m examples.teleop_eef --teleop Keyboard
```

See [`docs/teleop/gamepad.md`](../teleop/gamepad.md) for gamepad setup instructions.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--teleop` | `Spacemouse` | Teleop device: `Spacemouse`, `Gamepad`, `Keyboard` |
| `--freq` | `50` | Control frequency (Hz) |
| `--arm_latency` | `0.0` | Extra latency offset for arm commands (s) |
| `--instruction` | `""` | Language instruction recorded with the trajectory |

### Gello Teleoperation (`teleop_gello.py`)

Mirror the robot's joint positions using a [Gello](../teleop/gello.md) leader arm.

```bash
uv run -m examples.teleop_gello
# load Gello config from YAML:
uv run -m examples.teleop_gello --teleop_cfg_yaml path/to/gello.yaml
```

On startup, the script checks that the Gello is aligned with the robot's current pose before allowing motion to begin.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--teleop_cfg_yaml` | `None` | Path to a Gello YAML config (output of `generate_config`) |
| `--freq` | `15` | Control frequency (Hz) |
| `--arm_latency` | `0.0` | Extra latency offset for arm commands (s) |

### SO101 Teleoperation (`teleop_so101.py`)

Leader-follower joint mirroring for SO100/SO101 arms connected over serial.

```bash
uv run -m examples.teleop_so101
```

### XLerobot Bimanual Teleoperation (`teleop_xlerobot.py`)

Bimanual teleoperation for XLerobot setups with two leader arms and optional keyboard control for the recorder.

```bash
uv run -m examples.teleop_xlerobot
```

During recording, use keyboard shortcuts:
- `n` — start a new trajectory
- `s` — save the current trajectory

## Data Replay

Visualize a saved trajectory in Rerun without running any hardware.

```bash
uv run -m examples.replay_data --loader_cfg.path /data/rollouts/my_task/traj_0001.vla
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--loader_cfg.path` | (required) | Path to a `.vla` trajectory file |
| `--freq` | `50` | Replay frequency (Hz) |

## Policy Inference

Run a VLA policy on the robot in a closed-loop control at a fixed frequency.

```bash
uv run -m examples.policy_inference
```

The inference loop:
1. Collects an observation (camera images + proprioception)
2. Sends it to the policy server
3. Receives an action chunk and steps through it
4. Requests a new chunk before the current one is exhausted (controlled by `chunk_request_threshold`)

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--freq` | `50` | Control frequency (Hz) |
| `--instruction` | from config | Language instruction passed to the policy |
| `--visualizer` | `None` | Set to `"Rerun"` to enable live visualization |
| `--policy_node_cfg.chunk_size` | `16` | Number of actions per chunk |
| `--policy_node_cfg.chunk_request_threshold` | `0.1` | Fraction consumed before requesting next chunk |

from dataclasses import dataclass, field

from rio.cfg import Camera, VisualizerCfg
from rio.cfg.common import RecorderCfg

TASK = "pick_and_place"


@dataclass
class KassowStation:
    """Kassow KR-series arm teleoperated in end-effector space with a Spacemouse.

    Cartesian streaming is smoothest near 10 Hz with the default TT_TIME of
    0.10s. Raise `arm_cfg.max_pos_speed` and `arm_cfg.max_rot_speed` in steps to
    teleoperate faster, watching the peak commanded speed the arm node logs at
    startup. Set `arm_cfg.log_diagnostics` to see the achieved sync and command
    rates while running.
    """

    @dataclass
    class ArmCfg:
        addr: str = "127.0.0.1:5555"
        robot_ip: str = "192.168.1.44"
        port: int = 7582
        session_id: int = 1
        robot_controller: str = "task_pos"
        max_pos_speed: float = 0.15  # m/s
        max_rot_speed: float = 0.25  # rad/s
        stream_l_mode: str = "time"  # "time" (TT_TIME) | "speed" (TT_WS_TARGET_SPEED)
        stream_l_tt: float = 0.10  # seconds; keep near 1 / station freq
        stream_l_bt: float = 0.07  # blend window, seconds
        stream_l_speed: float = 0.0  # m/s for "speed" mode; 0 derives it
        stream_l_throttle: int = 5
        lowpass_alpha: float | None = 0.35  # None disables; lower = smoother/laggier
        log_diagnostics: bool = False
        cmd_freq: int = 10  # kept in sync with the station freq below
        freq: int = 250

    arm: str = "KassowArm"
    arm_cfg: ArmCfg = field(default_factory=ArmCfg)

    gripper: str | None = None
    gripper_cfg: None = None

    # Add entries here to record video, e.g.
    # "camera_1": Camera(addr="127.0.0.1:5130", cam_type="Realsense", serial="...")
    cameras: dict[str, Camera] = field(default_factory=dict)

    @dataclass
    class TeleopCfg:
        addr: str = "127.0.0.1:5000"

    teleop: str = "Spacemouse"  # Gamepad | Keyboard | Spacemouse
    teleop_cfg: TeleopCfg = field(default_factory=TeleopCfg)

    arm_latency: float = 0.0
    mw: str = "Thread"
    mp_method: str = "spawn"
    freq: int = 10

    action_space: str = "task_pos"
    embodiment_type: str = "SINGLE_ARM"

    instruction: str = ""
    visualizer: str | None = None
    visualizer_cfg: VisualizerCfg = field(default_factory=VisualizerCfg)

    recorder: str | None = "Recorder"
    recorder_cfg: RecorderCfg = field(default_factory=lambda: RecorderCfg(path=f"data/{TASK}/"))

    def __post_init__(self) -> None:
        # The arm node sizes its envelope and step guards against the rate targets
        # actually arrive at, so a mismatch here would silently cap the speed.
        self.arm_cfg.cmd_freq = self.freq

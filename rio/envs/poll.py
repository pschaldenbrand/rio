# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import numpy as np


class TeleopMode(Enum):
    TRANSLATION = 0
    TRANSLATION_ROTATION = 1
    TRANSLATION_2D = 2
    ROTATION = 3


class Interface:
    @staticmethod
    def poll(_teleop, teleop, t_sample, t_last_mode_change, teleop_mode):
        poll_teleop_fn = getattr(Interface, f"poll_{_teleop.lower()}")
        return poll_teleop_fn(teleop, t_sample, t_last_mode_change, teleop_mode)

    @staticmethod
    def poll_gamepad(gp, t_sample, t_last_mode_change, teleop_mode):
        """
        Controls:
        - Left stick: XY translation
        - Right stick: XY rotation
        - LT/RT: Z translation/rotation
        - A/B (South/East) buttons: gripper open/close
        - X (West) button: change mode
        """
        gp_motion = gp.get_motion_state_transformed()
        gp_x = gp.is_button_pressed(2)
        gp_a = gp.is_button_pressed(0)
        gp_b = gp.is_button_pressed(3)
        delta_tcp_pose = gp_motion
        gripper_pose = None
        if gp_x:
            if t_sample - t_last_mode_change > 1.0:  # 1 second delay between mode changes
                teleop_mode = (teleop_mode.value + 1) % len(TeleopMode)
                teleop_mode = TeleopMode(teleop_mode)
                t_last_mode_change = t_sample
        elif gp_a:
            gripper_pose = 1.0  # open
        elif gp_b:
            gripper_pose = 0.0  # close
        return delta_tcp_pose, gripper_pose, t_last_mode_change, teleop_mode

    @staticmethod
    def poll_keyboard(kb, t_sample, t_last_mode_change, teleop_mode):
        """
        Controls:
        - WASD: XY translation
        - QE: Z translation
        - IJKL: XY rotation
        - UO: Z rotation
        - []: gripper open/close
        - 0/1/2/3: teleop mode
        """
        alphanumeric_state = kb.get_state()["alphanumeric_state"]
        kb_motion = np.zeros((6,), dtype=np.float32)
        pos_gripper = None
        keys = []
        for key in alphanumeric_state:
            if key != 0:
                keys.append(chr(key))
        for key in keys:
            # translation
            if key == "w":
                kb_motion[0] = 1.0
            elif key == "s":
                kb_motion[0] = -1.0
            elif key == "a":
                kb_motion[1] = -1.0
            elif key == "d":
                kb_motion[1] = 1.0
            elif key == "q":
                kb_motion[2] = -1.0
            elif key == "e":
                kb_motion[2] = 1.0
            # rotation
            if key == "i":
                kb_motion[3] = 1.0
            elif key == "k":
                kb_motion[3] = -1.0
            elif key == "j":
                kb_motion[4] = 1.0
            elif key == "l":
                kb_motion[4] = -1.0
            elif key == "u":
                kb_motion[5] = 1.0
            elif key == "o":
                kb_motion[5] = -1.0
            # gripper
            if key == "[":
                pos_gripper = 0.0
            elif key == "]":
                pos_gripper = 1.0
            # teleop mode
            if key == "0":
                teleop_mode = TeleopMode.TRANSLATION_2D
            elif key == "1":
                teleop_mode = TeleopMode.TRANSLATION
            elif key == "2":
                teleop_mode = TeleopMode.ROTATION
            elif key == "3":
                teleop_mode = TeleopMode.TRANSLATION_ROTATION
        delta_tcp_pose = kb_motion
        return delta_tcp_pose, pos_gripper, t_last_mode_change, teleop_mode

    @staticmethod
    def poll_spacemouse(sp, t_sample, t_last_mode_change, teleop_mode):
        """
        Controls:
        - controller cap: translation and rotation
        - 0 button: gripper close
        - 1 button: gripper open
        - 0 and 1 buttons: change mode
        """
        sp_motion = sp.get_motion_state_transformed()
        sp_b0 = sp.is_button_pressed(0)
        sp_b1 = sp.is_button_pressed(1)
        delta_tcp_pose = sp_motion
        gripper_pos = None
        if sp_b0 and sp_b1:
            if t_sample - t_last_mode_change > 1.0:  # 1 second delay between mode changes
                teleop_mode = (teleop_mode.value + 1) % len(TeleopMode)
                teleop_mode = TeleopMode(teleop_mode)
                t_last_mode_change = t_sample
        elif sp_b0:
            gripper_pos = 0.0  # close
        elif sp_b1:
            gripper_pos = 1.0  # open
        return delta_tcp_pose, gripper_pos, t_last_mode_change, teleop_mode

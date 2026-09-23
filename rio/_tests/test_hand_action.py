# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tier 1: dexterous hand segment of the SingleArm action vector.
"""

import numpy as np
import pytest

from rio.embodiments.single_arm import SingleArm

pytestmark = pytest.mark.unit

N_HAND_DOF = 6


class StubArm:
    def __init__(self, num_joints=6, state=None):
        self.num_joints = num_joints
        self._state = state or {}
        self.calls = []

    def get_state(self):
        return self._state

    def moveL(self, target_eef_pose, target_time):
        self.calls.append(("moveL", list(target_eef_pose), target_time))

    def moveJ(self, target_joint_q, target_time):
        self.calls.append(("moveJ", list(target_joint_q), target_time))


class RecordingHand:
    """Inspire-hand stand-in: 6 normalized DOF commanded through moveJ."""

    def __init__(self, state=None):
        self._state = state or {"joint_q": np.zeros(N_HAND_DOF)}
        self.calls = []

    def get_state(self):
        return self._state

    def moveJ(self, target_angles, target_time):
        self.calls.append(("moveJ", list(target_angles), target_time))


def make_robot(hand=None, action_space="TASK_POS", arm=None):
    return SingleArm(arm=arm or StubArm(), hand=hand, action_space=action_space)


def test_build_action_appends_hand_segment():
    robot = make_robot(hand=RecordingHand())
    action = robot.build_action(np.arange(6.0), gripper_cmd=0.0, hand_cmd=np.full(N_HAND_DOF, 0.5))
    assert action.shape == (13,)


def test_build_action_without_hand_is_unchanged():
    """Existing stations keep producing 7-D actions."""
    robot = make_robot()
    action = robot.build_action(np.arange(6.0), gripper_cmd=0.25)
    assert action.shape == (7,)


def test_parse_action_splits_arm_gripper_hand():
    robot = make_robot(hand=RecordingHand())
    action = np.arange(13.0)
    parsed = robot.parse_action(action)

    np.testing.assert_array_equal(parsed["arm_cmd"], np.arange(6.0))
    assert parsed["gripper_cmd"] == 6.0
    np.testing.assert_array_equal(parsed["hand_cmd"], np.arange(7.0, 13.0))


def test_parse_build_roundtrip_with_hand():
    robot = make_robot(hand=RecordingHand())
    arm_cmd = np.array([0.4, 0.1, 0.3, 0.0, np.pi, 0.0])
    hand_cmd = np.linspace(0.0, 1.0, N_HAND_DOF)

    parsed = robot.parse_action(robot.build_action(arm_cmd, gripper_cmd=0.0, hand_cmd=hand_cmd))
    np.testing.assert_allclose(parsed["arm_cmd"], arm_cmd)
    np.testing.assert_allclose(parsed["hand_cmd"], hand_cmd)


def test_hand_cmd_is_none_without_a_hand_client():
    """A long action must not be interpreted as hand DOF when no hand is wired."""
    robot = make_robot(hand=None)
    parsed = robot.parse_action(np.arange(13.0))
    assert parsed["hand_cmd"] is None


def test_hand_cmd_is_none_for_short_action():
    """Arm-only bring-up: a 7-D action leaves the hand uncommanded."""
    robot = make_robot(hand=RecordingHand())
    parsed = robot.parse_action(np.arange(7.0))
    assert parsed["hand_cmd"] is None


def test_move_dispatches_arm_and_hand():
    arm = StubArm()
    hand = RecordingHand()
    robot = SingleArm(arm=arm, hand=hand, action_space="TASK_POS")

    arm_cmd = np.array([0.4, 0.1, 0.3, 0.0, np.pi, 0.0])
    hand_cmd = np.full(N_HAND_DOF, 0.5)
    robot.move(robot.build_action(arm_cmd, gripper_cmd=0.0, hand_cmd=hand_cmd), t_cmd_target=1.0)

    # TASK_POS routes the arm through moveL with 6 values, no appended gripper.
    assert [c[0] for c in arm.calls] == ["moveL"]
    np.testing.assert_allclose(arm.calls[0][1], arm_cmd)
    assert len(hand.calls) == 1
    np.testing.assert_allclose(hand.calls[0][1], hand_cmd)
    assert hand.calls[0][2] == 1.0


def test_move_without_hand_segment_leaves_hand_alone():
    hand = RecordingHand()
    robot = make_robot(hand=hand)
    robot.move(robot.build_action(np.zeros(6), gripper_cmd=0.0), t_cmd_target=1.0)
    assert hand.calls == []


def test_get_obs_reports_hand_joints():
    arm = StubArm(state={"eef_pose": np.ones(6), "joint_q": np.zeros(6)})
    hand_state = np.linspace(0.0, 1.0, N_HAND_DOF)
    robot = SingleArm(arm=arm, hand=RecordingHand(state={"joint_q": hand_state}), action_space="TASK_POS")

    obs = robot.get_obs(cams={})
    np.testing.assert_allclose(obs.hand_joints, hand_state)


def test_get_obs_without_hand_leaves_hand_joints_none():
    arm = StubArm(state={"eef_pose": np.ones(6), "joint_q": np.zeros(6)})
    obs = SingleArm(arm=arm, action_space="TASK_POS").get_obs(cams={})
    assert obs.hand_joints is None

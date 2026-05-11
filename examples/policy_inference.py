import multiprocessing as mp
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import tyro
from loguru import logger
from rio_hw import time
from rio_hw.middleware import ServerManager

from rio.envs.env import make_env


# TODO: this should be moved into each individual policy
def create_obs(env):
    formatted_obs = {}
    state = env.get_state()
    for key in state.observation.cameras:
        formatted_obs[key] = state.observation.cameras[key].rgb
    formatted_obs["proprio_joints"] = state.observation.proprio_joints
    formatted_obs["gripper_position"] = state.observation.gripper_position
    return formatted_obs


def policy_loop(args, env, policy, visualizer=None):
    assert env.robot.arm.robot_controller.name == "JOINT_VEL"
    if visualizer:
        visualizer.set_robot_model("world/robot", robot_description=env.robot.urdf_path, variant=None)
    input("Press Enter to start policy inference loop...")
    time.sleep(0.5)
    try:
        # Main loop
        freq = args.freq
        dt = 1.0 / freq
        t_start = time.now()
        it = 0

        # create first action chunk
        action_chunk = []
        action_chunk_idx = args.policy_node_cfg.chunk_size  # force request on first iteration
        action = np.zeros((args.policy_node_cfg.action_dim,), dtype=np.float32)
        processing_obs = False
        env.set_start_time(t_start)
        env.set_instruction(args.instruction)

        recording = False
        chunk_counter = 0

        while True:
            t_cycle_end = t_start + (it + 1) * dt
            t_cmd_target = t_cycle_end + dt

            policy_loaded = policy.policy_loaded

            if policy_loaded:
                # If needed, send observation to request action chunk
                chunk_consumed = action_chunk_idx / args.policy_node_cfg.chunk_size
                if chunk_consumed >= args.policy_node_cfg.chunk_request_threshold:
                    if not processing_obs:
                        logger.warning(
                            f"REQUESTING OBS, action_chunk_idx: {action_chunk_idx}, action_chunk len:{len(action_chunk)}"
                        )
                        obs = create_obs(env)
                        policy.send_observation(obs)
                        processing_obs = True
                        chunk_counter += 1

                    # Get policy response
                    response = policy.get_action_chunk()
                    if response["ready"]:
                        action_chunk = response["actions"]
                        chunk_start = response["timestamp"]
                        action_chunk_idx = int((t_cycle_end - chunk_start) * freq)
                        processing_obs = False

                    # If action within current chunk, send command to environment
                    if action_chunk_idx < len(action_chunk):
                        action = action_chunk[action_chunk_idx]
                        _t_cmd_target = t_cmd_target + args.arm_latency
                        env.move(
                            action,
                            t_cmd_target=_t_cmd_target,
                        )
                        action_chunk_idx += 1

                step = env.get_state(action=action)

                if recording:
                    env.recorder.record_step(step)

                if visualizer:
                    visualizer.log_env_state("env", step)

            time.precise_wait(t_cycle_end)
            it += 1

    except KeyboardInterrupt:
        pass
    finally:
        if env.recorder:
            env.recorder.save(wait=True)


def main(args):
    servers, clients, env = make_env(args)

    with ServerManager(args.mw, [*list(servers.values())]):
        with (
            env,
            clients["policy"]() as policy,
            clients["visualizer"]() if clients["visualizer"] else nullcontext() as visualizer,
        ):
            try:
                policy_loop(args, env, policy, visualizer)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    import examples.policy_cfgs as policy_cfgs
    from examples import get_policy_cfg

    PolicyCfg = get_policy_cfg(policy_cfgs)

    @dataclass
    class Args(PolicyCfg):
        pass

    args = tyro.cli(Args)
    print(args)
    mp.set_start_method(args.mp_method, force=True)
    main(args)

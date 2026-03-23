import argparse
from pathlib import Path
import pickle
import yaml
import gymnasium as gym
import gymnasium_robotics
import numpy as np

from collector import collect_one_episode
from tasks.slide_cabinet import scripted_policy as slide_cabinet_policy
from utils import apply_fixed_reset_noise, get_obs_after_reset_noise, make_fixed_reset_noise_bank, make_fixed_reset_noise_bank_gaussian

gym.register_envs(gymnasium_robotics)


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_name: str, task_name: str, render: bool):
    env = gym.make(
        env_name,
        tasks_to_complete=[task_name],
        render_mode="rgb_array" if render else None,
        width=256, height=256
    )
    return env


def get_policy(task_name: str):
    if task_name == "slide cabinet":
        return slide_cabinet_policy
    raise ValueError(f"Unsupported task: {task_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    task_name = cfg["task_name"]
    env_name = cfg.get("env_name", "FrankaKitchen-v1")
    render = cfg.get("render", True)
    save_path = cfg["save_path"]

    env = make_env(env_name, task_name, render)
    policy_fn = get_policy(task_name)
    noise_bank = make_fixed_reset_noise_bank_gaussian(
        num_demos=200,
        seed=128, #42
        qpos_noise_std=[0.115, 0.055, 0.115, 0.075, 0.095, 0.095, 0.115],
        gripper_noise_std=0.0035,
        truncate_sigma=2.0
    )
    
    all_traj =[]
    for i in range(5):
        print(f"Collecting demo {i+1}/200")

        obs, info = env.reset()

        apply_fixed_reset_noise(env, noise_bank[i])

        obs = get_obs_after_reset_noise(env, obs_before=obs)

        traj = collect_one_episode(env, obs, cfg, policy_fn)

        all_traj.append(traj)

    env.close()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(all_traj, f)

    print(f"saved to {save_path}")


if __name__ == "__main__":
    main()
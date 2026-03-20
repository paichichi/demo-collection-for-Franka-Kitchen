import numpy as np
from utils import get_site_rotmat, get_task_error


def collect_one_episode(env, cfg, policy_fn):
    obs, info = env.reset()

    task_name = cfg["task_name"]
    ee_site_name = cfg["ee_site_name"]
    max_episode_steps = cfg.get("max_episode_steps", 280)
    render = cfg.get("render", True)
    stop_delta_eps = cfg["stop_delta_eps"]
    stop_delta_steps = cfg["stop_delta_steps"]
    verbose = cfg.get("verbose", True)

    observations = []
    actions = []
    images = []

    achieved_goals = []
    episode_task_completions = []

    desired_goal = obs["desired_goal"]

    phase = 0
    state = {
        "phase0_steps": 0,
        "stop_counter": 0,
        "ee_R_ref": get_site_rotmat(env, ee_site_name),
    
        # 通用：基于任务进展的 phase 切换辅助
        "motion_counter": 0,
        "prev_task_error": None,
    }
    
    if cfg["task_name"] == "slide cabinet":
        state["phase1_yz_ref"] = None

    task_errors = []
    phase_history = []
    drawer_deltas = []

    done_by_stop_delta = False
    terminated = False
    truncated = False

    for step in range(max_episode_steps):
        if render:
            img = env.render()
            images.append(np.asarray(img, dtype=np.uint8))

        observations.append(np.asarray(obs["observation"], dtype=np.float64))
        achieved_goals.append(obs["achieved_goal"])

        current_phase = phase
        action, next_phase, policy_info = policy_fn(env, obs, current_phase, state, cfg)

        next_obs, reward, env_terminated, env_truncated, info = env.step(action)

        new_task_error = get_task_error(next_obs, task_name)

        old_ag = np.asarray(obs["achieved_goal"][task_name], dtype=np.float64).reshape(-1)
        new_ag = np.asarray(next_obs["achieved_goal"][task_name], dtype=np.float64).reshape(-1)
        drawer_delta = np.linalg.norm(new_ag - old_ag)

        if current_phase == 1:
            if drawer_delta < stop_delta_eps:
                state["stop_counter"] += 1
            else:
                state["stop_counter"] = 0

            if state["stop_counter"] >= stop_delta_steps:
                done_by_stop_delta = True
        else:
            state["stop_counter"] = 0

        terminated = bool(done_by_stop_delta)
        truncated = bool(env_truncated)

        actions.append(action)
        episode_task_completions.append(list(info.get("episode_task_completions", [])))

        task_errors.append(new_task_error)
        phase_history.append(current_phase)
        drawer_deltas.append(drawer_delta)

        if verbose:
            print(
                f"step={step:03d} | "
                f"phase={current_phase}->{next_phase} | "
                f"approach_error={policy_info.get('approach_error', -1):.4f} | "
                f"handle_center_error={policy_info.get('handle_center_error', -1):.4f} | "
                f"task_progress={policy_info.get('task_progress', 0.0):.4f} | "
                f"motion_counter={policy_info.get('motion_counter', 0)} | "
                f"task_error={new_task_error:.4f} | "
                f"drawer_delta={drawer_delta:.6f} | "
                f"completed={info.get('episode_task_completions', [])}"
            )

        phase = next_phase
        obs = next_obs

        if terminated or truncated:
            break

    final_completed = episode_task_completions[-1] if len(episode_task_completions) > 0 else []
    final_task_error = task_errors[-1] if len(task_errors) > 0 else np.inf

    final_completed = episode_task_completions[-1] if len(episode_task_completions) > 0 else []
    env_success = (task_name in final_completed)
    success = bool(env_success)

    print("\n===== Episode Summary =====")
    print(f"task_name: {task_name}")
    print(f"length: {len(actions)}")
    print(f"first_success_step: {next((i for i, c in enumerate(episode_task_completions) if task_name in c), None)}")
    print(f"final_completed: {final_completed}")
    print(f"env_success: {env_success}")
    print(f"stop_terminated: {done_by_stop_delta}")
    print(f"final_task_error: {final_task_error:.4f}")
    print("===========================\n")

    traj = {
        "task_name": task_name,
        "observations": np.asarray(observations),
        "actions": np.asarray(actions),
        "images": np.asarray(images) if len(images) > 0 else None,
    
        "desired_goal": desired_goal,
        "achieved_goal": achieved_goals,
    
        "episode_task_completions": episode_task_completions,
    
        "task_errors": np.asarray(task_errors),
        "phase_history": np.asarray(phase_history),
        "drawer_deltas": np.asarray(drawer_deltas),
    
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    
        "env_success": bool(env_success),
        "stop_terminated": bool(done_by_stop_delta),
        "success": bool(env_success),
    
        "final_task_error": float(final_task_error),
        "length": len(actions),
    }

    return traj
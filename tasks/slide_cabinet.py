import numpy as np

from utils import (
    get_site_pos,
    get_handle_rel_in_ee_frame,
    get_gripper_width_from_obs,
    get_task_error,
    compute_arm_qvel_from_ee_velocity,
    unit_vector,
)


def scripted_policy(env, obs, phase, state, cfg):
    """
    2-phase scripted policy

    phase:
      0 = approach_and_close
      1 = closed_push
    """

    # =========================
    # 从 cfg 读取参数
    # =========================
    task_name = cfg["task_name"]
    ee_site_name = cfg["ee_site_name"]
    target_site_name = cfg["target_site_name"]

    reach_offset = np.array(cfg["reach_offset"], dtype=np.float32)

    reach_threshold = cfg["reach_threshold"]
    grasp_approach_threshold = cfg["grasp_approach_threshold"]
    handle_center_threshold = cfg["handle_center_threshold"]

    pull_speed = cfg["pull_speed"]
    pull_direction = np.array(cfg["pull_direction"], dtype=np.float32)

    gripper_open_action = cfg.get("gripper_open_action", 1.0)
    gripper_close_action = cfg.get("gripper_close_action", -1.0)

    # 新增：交互触发参数
    task_progress_threshold = cfg.get("task_progress_threshold", 0.002)
    task_progress_steps = cfg.get("task_progress_steps", 3)

    # 新增：phase 1 的 y/z 稳定系数
    yz_hold_k = cfg.get("yz_hold_k", 3.0)
    yz_hold_clip = cfg.get("yz_hold_clip", 0.03)

    # =========================
    # 当前状态
    # =========================
    ee_pos = get_site_pos(env, ee_site_name)
    handle_pos = get_site_pos(env, target_site_name)
    handle_rel_ee = get_handle_rel_in_ee_frame(env, ee_site_name, target_site_name)

    gripper_width = get_gripper_width_from_obs(obs)
    task_error = get_task_error(obs, task_name)

    action = np.zeros(9, dtype=np.float32)

    handle_center_error = np.linalg.norm(handle_rel_ee)

    # 接近目标：把手稍前/稍上方一点
    approach_target = handle_pos + reach_offset
    approach_delta = approach_target - ee_pos
    approach_error = np.linalg.norm(approach_delta)

    grasp_error = np.linalg.norm(ee_pos - handle_pos)

    # =========================
    # 用 task_error 变化近似判断“drawer 是否已经被有效带动”
    # =========================
    prev_task_error = state.get("prev_task_error", None)
    task_progress = 0.0
    if prev_task_error is not None:
        task_progress = float(prev_task_error - task_error)

    if task_progress > task_progress_threshold:
        state["motion_counter"] = state.get("motion_counter", 0) + 1
    else:
        state["motion_counter"] = 0

    # 当前 step 结束前再更新 prev_task_error
    state["prev_task_error"] = task_error

    # =========================
    # Phase 0: approach_and_close
    # =========================
    if phase == 0:
        state["phase0_steps"] += 1

        # ee_v = 4.0 * approach_delta

        # # 远时快，近时慢
        # if approach_error > 0.15:
        #     ee_v = np.clip(ee_v, -0.30, 0.30)
        # elif approach_error > 0.08:
        #     ee_v = np.clip(ee_v, -0.16, 0.16)
        # else:
        #     ee_v = np.clip(ee_v, -0.08, 0.08)

        # ee_v = 6.0 * approach_delta

        # if approach_error > 0.15:
        #     ee_v = np.clip(ee_v, -0.45, 0.45)
        # elif approach_error > 0.08:
        #     ee_v = np.clip(ee_v, -0.25, 0.25)
        # else:
        #     ee_v = np.clip(ee_v, -0.12, 0.12)

        ee_v = 9.0 * approach_delta
        
        if approach_error > 0.18:
            ee_v = np.clip(ee_v, -0.70, 0.70)
        elif approach_error > 0.10:
            ee_v = np.clip(ee_v, -0.40, 0.40)
        else:
            ee_v = np.clip(ee_v, -0.15, 0.15)

        qdot_arm = compute_arm_qvel_from_ee_velocity(env, ee_v, ee_site_name)

        action[:7] = np.clip(qdot_arm, -1.0, 1.0)
        action[7] = gripper_close_action
        action[8] = gripper_close_action

        # -------------------------
        # 切换条件：几何接近 或 已经明显推动 drawer
        # -------------------------
        close_enough = (
            approach_error < reach_threshold
            and grasp_error < grasp_approach_threshold
        )
        center_ready = (handle_center_error < handle_center_threshold)

        interaction_ready = (
            state.get("motion_counter", 0) >= task_progress_steps
        )

        if close_enough or center_ready or interaction_ready:
            next_phase = 1
            state["phase0_steps"] = 0

            # 记录 phase1 刚开始时的 y/z 参考，避免切换时突然下掉
            state["phase1_yz_ref"] = ee_pos[1:].copy()
        else:
            next_phase = 0

    # =========================
    # Phase 1: closed_push
    # =========================
    else:
        # 如果离把手太远，说明真的完全脱手了，才退回 phase 0
        if grasp_error > 0.30:
            next_phase = 0
            action[:7] = 0.0
            action[7] = gripper_close_action
            action[8] = gripper_close_action
        else:
            pull_dir = unit_vector(pull_direction)
    
            # 主方向持续拉
            pull_v = pull_dir * pull_speed
    
            # =========================
            # Phase 1 接触保持：持续朝 handle center 纠偏
            # =========================
            handle_track_k = cfg.get("handle_track_k", 2.5)
            handle_track_clip = cfg.get("handle_track_clip", 0.05)
    
            # handle_rel_ee 是“手把在 ee 坐标系里的相对位置”
            # 用它构造一个朝手把中心贴近的修正速度
            track_v = handle_track_k * handle_rel_ee
            track_v = np.clip(track_v, -handle_track_clip, handle_track_clip)
    
            # 主方向拉动 + 接触保持修正
            # x 方向以 pull 为主，额外叠加少量跟踪
            ee_v = np.array([
                pull_v[0] + 0.2 * track_v[0],
                pull_v[1] + 1.0 * track_v[1],
                pull_v[2] + 1.0 * track_v[2],
            ], dtype=np.float32)
    
            qdot_arm = compute_arm_qvel_from_ee_velocity(env, ee_v, ee_site_name)
    
            action[:7] = np.clip(qdot_arm, -1.0, 1.0)
            action[7] = gripper_close_action
            action[8] = gripper_close_action
    
            next_phase = 1

    info = {
        "phase": phase,
        "next_phase": next_phase,
        "ee_pos": ee_pos,
        "handle_pos": handle_pos,
        "handle_rel_ee": handle_rel_ee,
        "gripper_width": gripper_width,
        "task_error": task_error,
        "task_progress": task_progress,
        "motion_counter": state.get("motion_counter", 0),
        "approach_error": approach_error,
        "grasp_error": grasp_error,
        "handle_center_error": handle_center_error,
    }

    return action, next_phase, info
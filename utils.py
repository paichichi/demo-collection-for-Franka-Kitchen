import mujoco
import numpy as np

DAMPING = 1e-4

def get_site_id(env, site_name: str):
    site_id = mujoco.mj_name2id(
        env.unwrapped.model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found.")
    return site_id

def get_site_pos(env, site_name: str):
    site_id = get_site_id(env, site_name)
    return env.unwrapped.data.site_xpos[site_id].copy()

def get_site_rotmat(env, site_name: str):
    """
    返回 site 旋转矩阵，shape=(3,3)
    MuJoCo site_xmat 是展平后的 9 维
    """
    site_id = get_site_id(env, site_name)
    return env.unwrapped.data.site_xmat[site_id].reshape(3, 3).copy()

def get_handle_rel_in_ee_frame(env, ee_site_name: str, target_site_name: str):
    """
    把 handle 在 EE 局部坐标系中的位置算出来：
        handle_rel = R_ee^T (handle_pos - ee_pos)
    """
    ee_pos = get_site_pos(env, ee_site_name)
    ee_R = get_site_rotmat(env, ee_site_name)
    handle_pos = get_site_pos(env, target_site_name)

    handle_rel = ee_R.T @ (handle_pos - ee_pos)
    return handle_rel.astype(np.float32)

def compute_arm_qvel_from_ee_velocity(env, ee_v_world, site_name, damping=DAMPING):
    """
    ee_v_world: shape=(3,), 末端世界坐标系线速度
    返回 7 维 arm joint velocity
    """
    model = env.unwrapped.model
    data = env.unwrapped.data

    site_id = get_site_id(env, site_name)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    # 前7维对应 Panda arm
    J = jacp[:, :7]   # (3, 7)

    JJt = J @ J.T
    qdot = J.T @ np.linalg.solve(JJt + damping * np.eye(3), ee_v_world)

    return qdot.astype(np.float32)

def get_task_error(obs, task_name):
    ag = np.asarray(obs["achieved_goal"][task_name], dtype=np.float64).reshape(-1)
    dg = np.asarray(obs["desired_goal"][task_name], dtype=np.float64).reshape(-1)
    return np.linalg.norm(ag - dg)

def get_gripper_width_from_obs(obs):
    """
    近似夹爪开口宽度
    observation[7], observation[8] 通常对应左右 finger 位置
    """
    return float(obs["observation"][7] + obs["observation"][8])

def unit_vector(vec, eps=1e-8):
    norm = np.linalg.norm(vec)
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm

def make_fixed_reset_noise_bank(num_demos=200, seed=42, qpos_noise_scale=None, gripper_noise_scale=0.0015,):
    if qpos_noise_scale is None:
        qpos_noise_scale = np.array([
            0.035, 0.000, 0.035, 0.030, 0.030, 0.035, 0.035
        ], dtype=np.float64)
    else:
        qpos_noise_scale = np.asarray(qpos_noise_scale, dtype=np.float64)

    rng = np.random.default_rng(seed)
    bank = []

    for _ in range(num_demos):
        arm_noise = rng.uniform(-qpos_noise_scale, qpos_noise_scale)
        grip_noise = rng.uniform(-gripper_noise_scale, gripper_noise_scale, size=2)

        noise9 = np.concatenate([arm_noise, grip_noise], axis=0)
        bank.append(noise9)

    return np.asarray(bank, dtype=np.float64)  # (num_demos, 9)

def make_fixed_reset_noise_bank_gaussian(
    num_demos=200,
    seed=42,
    qpos_noise_std=None,
    gripper_noise_std=0.0025,
    truncate_sigma=2.0,
):
    """
    生成更接近 old demo 风格的固定 reset 扰动库：
    - 前7维机械臂关节：截断高斯
    - 后2维 gripper：截断高斯
    """
    if qpos_noise_std is None:
        qpos_noise_std = np.array([
            0.060, 0.000, 0.060, 0.050, 0.050, 0.060, 0.060
        ], dtype=np.float64)
    else:
        qpos_noise_std = np.asarray(qpos_noise_std, dtype=np.float64)

    assert qpos_noise_std.shape == (7,)

    rng = np.random.default_rng(seed)
    bank = []

    for _ in range(num_demos):
        arm_noise = rng.normal(loc=0.0, scale=qpos_noise_std, size=7)
        arm_clip = truncate_sigma * qpos_noise_std
        arm_noise = np.clip(arm_noise, -arm_clip, arm_clip)

        grip_noise = rng.normal(loc=0.0, scale=gripper_noise_std, size=2)
        grip_clip = truncate_sigma * gripper_noise_std
        grip_noise = np.clip(grip_noise, -grip_clip, grip_clip)

        noise9 = np.concatenate([arm_noise, grip_noise], axis=0)
        bank.append(noise9)

    return np.asarray(bank, dtype=np.float64)

def apply_fixed_reset_noise(env, noise9, zero_qvel=True, clip_to_joint_range=True):
    """
    把 shape=(9,) 的扰动加到 env 当前 reset 后的 qpos[:9] 上。
    适配没有 set_state() 接口的 KitchenEnv。
    """
    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data

    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    qpos_new = qpos.copy()
    qpos_new[:9] += noise9

    # 可选：clip 到 joint range
    if clip_to_joint_range and hasattr(model, "jnt_range"):
        jnt_range = model.jnt_range
        n_clip = min(9, len(jnt_range))
        for i in range(n_clip):
            lo, hi = jnt_range[i]
            if hi > lo:
                qpos_new[i] = np.clip(qpos_new[i], lo, hi)

    # old demo 风格：qvel 不随机
    if zero_qvel:
        qvel_new = np.zeros_like(qvel)
    else:
        qvel_new = qvel.copy()

    # 关键修改：直接写底层 state
    data.qpos[:] = qpos_new
    data.qvel[:] = qvel_new

    # 如果环境里有 mocap，也保持当前值不变
    if hasattr(data, "mocap_pos"):
        data.mocap_pos[:] = data.mocap_pos
    if hasattr(data, "mocap_quat"):
        data.mocap_quat[:] = data.mocap_quat

    # 前向更新
    mujoco.mj_forward(model, data)

    return {
        "applied_noise": noise9.copy(),
        "qpos_after": data.qpos.copy(),
        "qvel_after": data.qvel.copy(),
    }

import inspect
import numpy as np


def get_obs_after_reset_noise(env, obs_before=None):
    """
    在手动修改 MuJoCo 的 qpos/qvel 之后，重新获取当前 obs。
    尽量兼容不同 KitchenEnv 实现。
    """
    unwrapped = env.unwrapped

    # 1) 最优先：如果环境本身有公开/半公开的 obs dict 接口
    for name in ["get_obs_dict", "_get_obs_dict"]:
        if hasattr(unwrapped, name):
            fn = getattr(unwrapped, name)
            try:
                return fn()
            except Exception:
                pass

    # 2) 如果有 _get_obs，需要看它是否要求 robot_obs
    if hasattr(unwrapped, "_get_obs"):
        fn = unwrapped._get_obs
        try:
            sig = inspect.signature(fn)
            n_params = len(sig.parameters)

            # 绑定方法时，self 不算在 parameters 里
            if n_params == 0:
                return fn()
            elif n_params == 1:
                robot_obs = None

                # 常见路径 A：robot.get_obs(...)
                if hasattr(unwrapped, "robot") and hasattr(unwrapped.robot, "get_obs"):
                    try:
                        robot_obs = unwrapped.robot.get_obs()
                    except Exception:
                        pass

                # 常见路径 B：robot_obs_from_sim(...)
                if robot_obs is None and hasattr(unwrapped, "robot_obs_from_sim"):
                    try:
                        robot_obs = unwrapped.robot_obs_from_sim()
                    except Exception:
                        pass

                # 常见路径 C：直接用 qpos/qvel 拼一个 robot_obs
                if robot_obs is None and hasattr(unwrapped, "data"):
                    try:
                        qpos = unwrapped.data.qpos.copy()
                        qvel = unwrapped.data.qvel.copy()
                        robot_obs = np.concatenate([qpos[:9], qvel[:9]], axis=0)
                    except Exception:
                        pass

                if robot_obs is not None:
                    return fn(robot_obs)
        except Exception:
            pass

    # 3) 最后兜底：直接返回 reset 前 obs 的浅拷贝并只更新 observation 部分
    if obs_before is not None:
        obs = dict(obs_before)

        if "observation" in obs and hasattr(unwrapped, "data"):
            qpos = unwrapped.data.qpos.copy()
            qvel = unwrapped.data.qvel.copy()

            new_obs = np.array(obs["observation"], copy=True)

            # 尽量保守地只替换前半部分
            n = min(len(new_obs), len(qpos) + len(qvel))
            qq = min(len(qpos), n)
            new_obs[:qq] = qpos[:qq]

            remain = n - qq
            if remain > 0:
                new_obs[qq:qq + min(remain, len(qvel))] = qvel[:min(remain, len(qvel))]

            obs["observation"] = new_obs

        return obs

    raise RuntimeError("无法在 reset randomization 后重新获取 obs，请检查环境接口。")
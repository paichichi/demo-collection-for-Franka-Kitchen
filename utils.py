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
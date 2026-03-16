"""
ARM / x86 通用版本
无 msgpack / 无 pickle / 无 jax / 无 flax
仅使用 numpy
"""

import numpy as np
import os
import pdb
import sys

sys.path.append("../")

from dynamics import *
from model_predict_pid import *
from utils.action_deal import *

# ===============================
# 1️⃣ 安全加载 actor npz
# ===============================

def load_actor_npz(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found")
    data = dict(np.load(npz_path, allow_pickle=True))
    return convert_to_float32(data)


def convert_to_float32(obj):
    if isinstance(obj, dict):
        return {k: convert_to_float32(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.ndim == 0:
            return convert_to_float32(obj.item())
        return obj.astype(np.float32)
    else:
        return obj


# ===============================
# 2️⃣ Actor (numpy forward)
# ===============================

class Actor:
    """
    对齐 Flax 结构：

    obs → Dense_0 → ReLU → Dense_1 → ReLU → OutputDenseMean → action(mean)

    log_std 不参与 eval_actions
    """

    def __init__(self, params):
        self.W1 = params["MLP_0_Dense_0_kernel"]
        self.b1 = params["MLP_0_Dense_0_bias"]

        self.W2 = params["MLP_0_Dense_1_kernel"]
        self.b2 = params["MLP_0_Dense_1_bias"]

        self.Wm = params["OutputDenseMean_kernel"]
        self.bm = params["OutputDenseMean_bias"]

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    def __call__(self, obs):
        """
        obs: (obs_dim,) or (B, obs_dim)
        return: (act_dim,) or (B, act_dim)
        """
        x = obs @ self.W1 + self.b1
        x = self.relu(x)

        x = x @ self.W2 + self.b2
        x = self.relu(x)   # activate_final=True

        mean = np.tanh(x @ self.Wm + self.bm)
        return mean
    


# ===============================
# 3️⃣ 用户接口
# ===============================

class ModelPredict:
    def __init__(self, npz_path):
        params = load_actor_npz(npz_path)
        print("[INFO] Loaded keys:", list(params.keys()))
        self.actor = Actor(params)

    def eval_action(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        act = self.actor(obs)
        return act.squeeze(0)

# [-0.35120437  0.30556118 -0.15785918  0.01850896 -0.6067741  -0.9999949
#  -0.05825673 -0.25037003]

npz_path = "./model/Miql_estimateF_MT_data4+6+7_envTTF/20750/actor_arm.npz" #这里注意模型的使用
npz_path = "./model/0225_model0212Sim60k_data4_9_14_seg8_2_off08_sr08__envRanEFD__29125_output_arm_params/actor_arm.npz"
npz_path = "./model/0225_model0212Sim60k_data4_9_14_seg8_2_off08_sr08__envRanEFDALLStep__48750_output_arm_params/actor_arm.npz"
npz_path = "./model/modeldata8+9+10_envTTTrandomM41k__data9+11+12_envTTTrandomM15_offRatio05_sR05__13625_output_arm_params/actor_arm.npz"

# model seed
npz_path = "./model/data16_of05_sr02__envNoEFD_att15__27625_output_arm_params/actor_arm.npz"
# npz_path = "./model/data16_of06_sr08__envNoEFD_att15__47000_output_arm_params/actor_arm.npz"
npz_path = "./model/0310_M27625_data18_0f05sr08__att30EnvV7__13125_output_arm_params/actor_arm.npz"

#0311 model
npz_path = "./model/0311_msiql_data16_ofr05sr05_env7__11250_output_arm_params/actor_arm.npz"
npz_path = "./model/0311_msiql_data16+18_ofr05sr07_env7__10625_output_arm_params/actor_arm.npz"  # 这个感觉还稍微保守一些
#npz_path = "./model/0311_msiql_data16+18_ofr05sr07_env7__14625_output_arm_params/actor_arm.npz"

print(npz_path)
predictor = ModelPredict(npz_path)
dynamics_uav = Dynamics(thrust_to_weight=1/0.5)  # 这个推重比要根据实际调整一下
pid_control_uav  = NonlinearPositionController(dynamics_uav)
filter = ActionSpikeFilter(dim=8)
dt = 0.01


def modelPredict(uav, state_error, vel, rot, rt):
    #pdb.set_trace()
    global dt
    # 注意这里更新的vel可不是世界vel，是机体vel，这里有问题
    # 可以尝试使用
    dynamics_uav.update_state(uav.pos, vel, uav.omega, rot) #这里主要时为了做外力估计
    action = predictor.eval_action(state_error)
    #pdb.set_trace()
    action = filter.filter(action)
    
    force_torque = pid_control_uav.step_force_torque(dynamics=dynamics_uav1, goal=uav.nominal_pos, dt=dt, action=action[:4])
    #print(f"rl cmd force_torque is {force_torque}")
    estimate_force_torque = get_estimate_force_torque(rt, uav, rot, force_torque)
    force_torque_cmd = np.array([force_torque[0], force_torque[1][0], force_torque[1][1], force_torque[1][2]])
    return action, estimate_force_torque , force_torque_cmd



# ===============================
# 4️⃣ Example test
# ===============================

# if __name__ == "__main__":
#     npz_path = "./model/Miql_estimateF_MT_data4+6+7_envTTF/20750/actor_arm.npz"

#     predictor = ModelPredict(npz_path)

#     obs = np.array([
#         5.7519036e-03,  1.7037179e-03, -4.7663195e-04,  6.8771780e-02,
#         3.0535361e-02, -8.7087780e-02,  1.0000000e+00,  1.6238292e-06,
#        -2.4296431e-04, -1.4766680e-06,  9.9999982e-01,  6.0568983e-04,
#         2.4296524e-04, -6.0568942e-04,  9.9999976e-01, -1.2105061e-01,
#        -4.8280157e-02, -5.7204638e-04,  0.0000000e+00,  0.0000000e+00,
#         0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
#         2.2900563e-02, -1.4540896e-01,  3.2798912e-02, -3.0811035e-04,
#        -5.5178846e-03, -8.8616163e-02,  9.9927545e-01,  1.5022551e-02,
#         3.4970339e-02, -1.6788987e-02,  9.9856877e-01,  5.0779380e-02,
#        -3.4157451e-02, -5.1329706e-02,  9.9809748e-01,  1.9582117e-04,
#         1.1403041e-03, -1.2719148e-03
#     ], dtype=np.float32)

#     action = predictor.eval_action(obs)
#     print("Action:", action, "shape:", action.shape)
    
    """
    测试数据对应关系：
    obs = np.array([
        5.7519036e-03,  1.7037179e-03, -4.7663195e-04,  6.8771780e-02,
        3.0535361e-02, -8.7087780e-02,  1.0000000e+00,  1.6238292e-06,
       -2.4296431e-04, -1.4766680e-06,  9.9999982e-01,  6.0568983e-04,
        2.4296524e-04, -6.0568942e-04,  9.9999976e-01, -1.2105061e-01,
       -4.8280157e-02, -5.7204638e-04,  0.0000000e+00,  0.0000000e+00,
        0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
        2.2900563e-02, -1.4540896e-01,  3.2798912e-02, -3.0811035e-04,
       -5.5178846e-03, -8.8616163e-02,  9.9927545e-01,  1.5022551e-02,
        3.4970339e-02, -1.6788987e-02,  9.9856877e-01,  5.0779380e-02,
       -3.4157451e-02, -5.1329706e-02,  9.9809748e-01,  1.9582117e-04,
        1.1403041e-03, -1.2719148e-03
    ], dtype=np.float32)
    action = array([-0.00903396,  0.10073812, -0.03651189, -0.00542336, -0.01681863,
        0.0743802 , -0.08667633,  0.01112348], dtype=float32)
    """

"""
ARM / x86 通用版本
无 msgpack / 无 pickle / 无 jax / 无 flax
仅使用 numpy
"""

import numpy as np
import os
import pdb
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
        self.W1 = params["MLP_0"]["Dense_0"]["kernel"]   # (obs_dim, 512)
        self.b1 = params["MLP_0"]["Dense_0"]["bias"]

        self.W2 = params["MLP_0"]["Dense_1"]["kernel"]   # (512, 256)
        self.b2 = params["MLP_0"]["Dense_1"]["bias"]

        self.Wm = params["OutputDenseMean"]["kernel"]    # (256, act_dim)
        self.bm = params["OutputDenseMean"]["bias"]

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

        mean = x @ self.Wm + self.bm
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


npz_path = "./model/Miql_estimateF_MT_data4+6+7_envTTF/20750/actor_arm.npz"
predictor = ModelPredict(npz_path)

def modelPredictforArm(state_error):
    action = predictor.eval_action(state_error)
    return action





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

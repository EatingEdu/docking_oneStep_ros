"""
ARM / x86 通用版本
无 msgpack / 无 pickle / 无 jax / 无 flax
仅使用 numpy
"""

import numpy as np
import os
import pdb

# ===============================
# 1️⃣ 安全加载 actor npz（ARM/x86 通用）
# ===============================

def load_actor_npz(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found")
    
    data = dict(np.load(npz_path, allow_pickle=True))
    return convert_to_float32(data)

    
def convert_to_float32(obj):
    import numpy as np

    # If obj is a dict, recurse for its values
    if isinstance(obj, dict):
        return {k: convert_to_float32(v) for k, v in obj.items()}
    
    # If obj is a numpy array with numeric dtype, convert
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.ndim == 0:
            # Unpack 0-d object array
            return convert_to_float32(obj.item())
        elif np.issubdtype(obj.dtype, np.floating):
            return obj.astype(np.float32)
        else:
            # Recurse elementwise for object arrays
            return np.array([convert_to_float32(v) for v in obj], dtype=object)
    
    # If obj is a list or tuple, recurse
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_to_float32(v) for v in obj)
    
    # Otherwise, just return
    else:
        return obj


# ===============================
# 2️⃣ MLP Actor (numpy forward)
# ===============================

class Actor:
    def __init__(self, params):
        self.layers = []
        mlp = params["MLP_0"]

        i = 0
        while f"Dense_{i}" in mlp:
            W = mlp[f"Dense_{i}"]["kernel"]
            b = mlp[f"Dense_{i}"]["bias"]
            self.layers.append((W.astype(np.float32), b.astype(np.float32)))
            i += 1

        self.mean_W = params["OutputDenseMean"]["kernel"].astype(np.float32)
        self.mean_b = params["OutputDenseMean"]["bias"].astype(np.float32)

        self.log_std = params["OutpuLogStd"].astype(np.float32)  # 保持原 key 名

        print("[INFO] Actor loaded:")
        for i, (W, b) in enumerate(self.layers):
            print(f"  Dense_{i}: {W.shape} -> {b.shape}")
        print(f"  OutputMean: {self.mean_W.shape} -> {self.mean_b.shape}")
        print(f"  LogStd: {self.log_std.shape}")

    def forward(self, x):
        for W, b in self.layers:
            x = np.maximum(0, x @ W + b)  # ReLU
        # for W, b in self.layers:
        #     x = np.tanh(x @ W + b)
        mean = x @ self.mean_W + self.mean_b
        return mean

    def sample(self, obs, deterministic=True):
        mean = self.forward(obs)
        if deterministic:
            return np.tanh(mean)
        std = np.exp(self.log_std)
        # 保证广播正确
        return np.tanh(mean + np.random.randn(*mean.shape) * std)

# ===============================
# 3️⃣ 用户接口：ModelPredict
# ===============================

class ModelPredict:
    def __init__(self, npz_path):
        params = load_actor_npz(npz_path)
        print("[INFO] Loaded keys:", list(params.keys()))
        self.actor = Actor(params)

    def eval_action(self, obs, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        act = self.actor.sample(obs, deterministic)
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

import numpy as np


class ActionSpikeFilter:

    def __init__(
        self,
        dim=8,
        alpha=0.05,
        spike_k=3.0,
        smooth=0.2,
        min_ratio=0.1,
        max_range=0.1,
        max_step=0.02
    ):

        self.dim = dim
        self.alpha = alpha
        self.spike_k = spike_k
        self.smooth = smooth
        self.min_ratio = min_ratio

        # 新增参数
        self.max_range = max_range
        self.max_step = max_step

        self.prev_action = None
        self.prev_output = None
        self.mean_delta = np.zeros(dim)

    # ----------------------------
    # 自适应阈值
    # ----------------------------
    def compute_threshold(self, delta):

        self.mean_delta = (
            self.alpha * np.abs(delta)
            + (1 - self.alpha) * self.mean_delta
        )

        return self.spike_k * self.mean_delta


    # ----------------------------
    # 尖刺削弱
    # ----------------------------
    def halve_spike(self, action, prev_action, threshold):

        result = action.copy()

        for i in range(self.dim):

            delta = action[i] - prev_action[i]
            original_delta = delta

            while abs(delta) > threshold[i]:
                delta = delta / 2.0

            min_delta = self.min_ratio * abs(original_delta)

            if abs(delta) < min_delta:
                delta = np.sign(original_delta) * min_delta

            result[i] = prev_action[i] + delta

        return result


    # ----------------------------
    # EMA平滑
    # ----------------------------
    def ema_smooth(self, action):

        if self.prev_output is None:
            self.prev_output = action
            return action

        out = (
            self.smooth * action +
            (1 - self.smooth) * self.prev_output
        )

        self.prev_output = out

        return out


    # ----------------------------
    # 丢弃非法值
    # ----------------------------
    def remove_invalid(self, action):

        action = action.copy()

        for i in range(self.dim):
            if action[i] == 1:
                action[i] = self.prev_action[i]

        return action


    # ----------------------------
    # 变化率限制
    # ----------------------------
    def limit_rate(self, action):

        delta = action - self.prev_action
        delta = np.clip(delta, -self.max_step, self.max_step)

        return self.prev_action + delta


    # ----------------------------
    # 自适应缩放
    # ----------------------------
    def scale_to_range(self, action):

        max_val = np.max(np.abs(action))

        if max_val > self.max_range:
            action = action * (self.max_range / max_val)

        return action


    # ----------------------------
    # 主过滤函数
    # ----------------------------
    def filter(self, action):

        action = np.array(action, dtype=np.float32)

        if self.prev_action is None:
            self.prev_action = action
            self.prev_output = action
            return action

        # 1 丢弃值为1的数据
        action = self.remove_invalid(action)

        # 2 计算变化
        delta = action - self.prev_action

        # 3 自适应阈值
        threshold = self.compute_threshold(delta)

        # 4 尖刺削弱
        action = self.halve_spike(action, self.prev_action, threshold)

        # 5 EMA平滑
        action = self.ema_smooth(action)

        # 6 限制变化率
        action = self.limit_rate(action)

        # 7 自适应缩放
        action = self.scale_to_range(action)

        # 8 最终安全限幅
        action = np.clip(action, -self.max_range, self.max_range)

        self.prev_action = action

        return action


# ----------------------------
# 示例
# ----------------------------
if __name__ == "__main__":

    filter = ActionSpikeFilter(dim=8)

    while True:

        action = policy_output()  # RL policy输出

        action_filtered = filter.filter(action)

        send_to_robot(action_filtered)
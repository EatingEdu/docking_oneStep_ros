import numpy as np


class ActionSpikeFilter:

    def __init__(
        self,
        dim=8,
        alpha=0.05,
        spike_k=3.0,
        smooth=0.2,
        min_ratio=0.1
    ):

        self.dim = dim
        self.alpha = alpha
        self.spike_k = spike_k
        self.smooth = smooth
        self.min_ratio = min_ratio

        self.prev_action = None
        self.prev_output = None
        self.mean_delta = np.zeros(dim)

    def compute_threshold(self, delta):

        self.mean_delta = (
            self.alpha * np.abs(delta)
            + (1 - self.alpha) * self.mean_delta
        )

        return self.spike_k * self.mean_delta

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

    def remove_invalid(self, action):
        """
        如果某个维度 == 1，则丢弃该值
        """

        action = action.copy()

        for i in range(self.dim):
            if action[i] == 1:
                action[i] = self.prev_action[i]

        return action

    def filter(self, action):

        action = np.array(action)

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

        self.prev_action = action

        return action
    
if __name__ == "__main__":    
    filter = ActionSpikeFilter(dim=8)

    while True:

        action = policy_output()  # RL policy输出

        action_filtered = filter.filter(action)

        send_to_robot(action_filtered)
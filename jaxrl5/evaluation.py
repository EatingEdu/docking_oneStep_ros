from typing import Dict

import gym
import numpy as np

from jaxrl5.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}


def evaluate_for_norm_env(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    success_rate = 0.
    success_count = 0
    for i in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            # action, agent = agent.eval_actions(observation)
            action, _ = agent.eval_actions(observation)
            observation, _, done, info = env.step(action)
        if env.env.env.env.success_flag:
            success_count += 1
        for k in stats.keys():
            stats[k].append(info['episode'][k])
        #for k in stats.keys():
        # stats['return'].append(info['rewards'])
        # stats['length'].append(env.env.env.env.env.env.env.step_count)
    for k, v in stats.items():
        stats[k] = np.mean(v)
    success_rate = success_count / num_episodes
    print(f"success_rate is {success_rate}")
    return stats, success_rate


def implicit_evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action, agent = agent.sample_implicit_policy(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue)}

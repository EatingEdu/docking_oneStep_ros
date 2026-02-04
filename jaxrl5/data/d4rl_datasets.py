import os
import pdb
import d4rl
import gym
import numpy as np
import mjrl
from collections import defaultdict
from jaxrl5.data.dataset import Dataset


def mergeDataset( batch1: Dataset, batch2: Dataset) -> Dataset:
    batch1 = batch1.dataset_dict
    batch2 = batch2.dataset_dict
    merge = lambda d1, d2: {k: np.concatenate((d1[k], d2[k])) for k in d1.keys()}
    merged_dict = merge(batch1, batch2)  
    return Dataset(merged_dict)

    
def get_dataset_for_norm_env(expert_path,size=None):
    dataset = {
    'env_obs': [],
    'action': [],
    'env_reward': [],
    'mask': [],
    'next_env_obs': []
    }
    try:
        for name in dataset.keys():
            data = np.load(os.path.join(expert_path, name + '.npy'))
            if size:
                dataset[name] = data[:size]
            else:
                dataset[name] = data
        return {
        'observations': dataset["env_obs"],
        'actions': dataset["action"],
        'next_observations': dataset["next_env_obs"],
        'rewards': dataset["env_reward"].reshape(1,-1)[0],
        'terminals': 1- dataset["mask"].reshape(1,-1)[0],
    }
    except:
        dataset = {
        'obs': [],
        'action': [],
        'total_reward': [],
        'mask': [],
        'next_obs': []
        }
        for name in dataset.keys():
            data = np.load(os.path.join(expert_path, name + '.npy'))
            if size:
                dataset[name] = data[:size]
            else:
                dataset[name] = data
        return {
        'observations': dataset["obs"],
        'actions': dataset["action"],
        'next_observations': dataset["next_obs"],
        'rewards': dataset["total_reward"].reshape(1,-1)[0],
        'terminals': 1- dataset["mask"].reshape(1,-1)[0],
        }
    
    
class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, 
                 clip_to_eps: bool = True, 
                 eps: float = 1e-5,
                 env_type: str = "gym",
                 expert_path: str="",
                 fine_tune: bool= False,
                 size: int=64,
                 noforce_flag: bool=False):
        if env_type == "gym":
            dataset_dict = d4rl.qlearning_dataset(env)
        elif env_type == "norm":
            if noforce_flag:
                if fine_tune:
                    dataset_dict = get_dataset_for_norm_env_noForce(expert_path, size=size)
                else:
                    dataset_dict = get_dataset_for_norm_env_noForce(expert_path)
            else:
                if fine_tune:
                    dataset_dict = get_dataset_for_norm_env(expert_path, size=size)
                else:
                    dataset_dict = get_dataset_for_norm_env(expert_path)
        
        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
                or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        dones[-1] = True

        dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        del dataset_dict["terminals"]

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones
        super().__init__(dataset_dict)

# env = gym.make('walker2d-medium-v2')
# ds = D4RLDataset(env)
# sample = ds.sample_jax(128, keys=None)
# print(sample)
# print(sample["observations"].shape)
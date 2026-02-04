from functools import partial
import wandb
import pdb
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import struct
from flax.training.train_state import TrainState
from jax.scipy.special import kl_div

from jaxrl5.types import PRNGKey
import optax
from optax import ScaleByAdamState, EmptyState

@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()


class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions), self.replace(rng=self.rng)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations
        )
        return np.asarray(actions), self.replace(rng=new_rng)
    
    
    # def save(self, save_path: str):
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     with open(save_path, 'wb') as f:
    #         f.write(flax.serialization.to_bytes(self.params))

    # def load(self, load_path: str) -> 'Model':
    #     with open(load_path, 'rb') as f:
    #         params = flax.serialization.from_bytes(self.params, f.read())
    #     return self.replace(params=params)

    def loadUtil(self, original_state):
        #pdb.set_trace()
        # 提取值
        count = original_state['0']['count']
        mu = original_state['0']['mu']
        nu = original_state['0']['nu']

        # 创建新的 ScaleByAdamState 实例
        new_state = ScaleByAdamState(count=count, mu=mu, nu=nu)

        # 如果需要，附加 EmptyState()
        final_state = (new_state, EmptyState())
        return final_state


    #@jit
    def compute_jsd(self, p, q):
        """计算杰森-香农散度（JSD）。"""
        p = jnp.clip(p, 1e-10, 1)
        q = jnp.clip(q, 1e-10, 1)
        m = 0.5 * (p + q)

        kl_pm = kl_div(p, m).sum()
        kl_qm = kl_div(q, m).sum()

        return 0.5 * (kl_pm + kl_qm)

    
    # @jit
    def compute_distribution(self, data, bins=10):
        """计算数据的概率分布（直方图）。"""
        hist, _ = jnp.histogram(data, bins=bins, density=True)
        return hist

    def compute_alpha(self, 
                      offline_states, 
                      online_states, 
                      offline_actions, 
                      online_actions,
                      use_state_alpha = True,
                      use_action_alpha = False):
        alpha_state, alpha_action = self.compute_alpha_two()
        
        if use_state_alpha :
            return alpha_state
        
        elif use_action_alpha:
            return alpha_action
        
        else:
            return coffecient * (alpha_state + alpha_action)
        
    # @jit
    def compute_alpha_two(self,
                      offline_states, 
                      online_states, 
                      offline_actions, 
                      online_actions, 
                      bins=10, 
                      lambda_param=1.0):
        """计算 alpha 值。"""
        # 计算状态分布
        offline_state_dist = self.compute_distribution(offline_states, bins)
        online_state_dist = self.compute_distribution(online_states, bins)
        state_jsd = compute_jsd(offline_state_dist, online_state_dist)

        # 计算动作分布
        offline_action_dist = self.compute_distribution(offline_actions, bins)
        online_action_dist = self.compute_distribution(online_actions, bins)
        action_jsd = self.compute_jsd(offline_action_dist, online_action_dist)

        # 计算 alpha
        alpha_state = 1 / (1 + lambda_param * state_jsd)
        alpha_action = 1 / (1 + lambda_param * action_jsd)

        return alpha_state, alpha_action
    
    def save(self, path_dir):
        with open(path_dir  + "actor", 'wb') as f:
            f.write(flax.serialization.to_bytes(self.actor))
        wandb.save('actor')
        
    def load(self, path_dir):
        agent = self
        with open(path_dir  + "actor", 'rb') as f:
            state_dict = flax.serialization.msgpack_restore(f.read())
        actor = agent.actor.replace(params=state_dict["params"], opt_state=self.loadUtil(state_dict["opt_state"]), step=state_dict["step"])
        agent = agent.replace(actor=actor)
        
        return agent
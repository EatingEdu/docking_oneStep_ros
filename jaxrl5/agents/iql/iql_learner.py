"""Implementations of algorithms for continuous control."""

import math
import pdb
import wandb
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import gym
import jax
import jax.numpy as jnp
import optax
import flax
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import Normal
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IQLLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    discount: float
    tau: float
    expectile: float
    temperature: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 1e-3,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        num_qs: int = 2,
        max_steps: Optional[int] = 1e5,
        opt_decay_schedule = 'cosine',
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        actions = action_space.sample()
        action_dim = action_space.shape[0]
        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)

        # actor_def = Normal(
        #     actor_base_cls,
        #     action_dim,
        #     log_std_min=math.log(0.1),
        #     log_std_max=math.log(0.1),
        #     state_dependent_std=False,
        # )
        
        actor_def = Normal(
            actor_base_cls,
            action_dim,
            log_std_min= -5.0 ,
            log_std_max= 2.0,
            state_dependent_std=False,
        )

        observations = observation_space.sample()
        actor_params = actor_def.init(actor_key, observations)["params"]
        #pdb.set_trace()
        if opt_decay_schedule == "cosine":  # 余弦学习率衰减 
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps) #这里的学习率可能是错的，初始应该是正数数？
            actor_optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            actor_optimiser = optax.adam(learning_rate=actor_lr)
            
        #actor_optimiser = optax.adam(learning_rate=actor_lr)
        actor = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params, tx=actor_optimiser
        )

        critic_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_def = StateValue(base_cls=critic_base_cls)
        value_params = value_def.init(value_key, observations)["params"]

        value_optimiser = optax.adam(learning_rate=value_lr)
        value = TrainState.create(
            apply_fn=value_def.apply, params=value_params, tx=value_optimiser
        )

        return cls(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            value=value,
            tau=tau,
            discount=discount,
            expectile=expectile,
            temperature=temperature,
            rng=rng,
        )

    def update_v(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        q1, q2 = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )

        q = jnp.minimum(q1, q2)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            value_loss = loss(q - v, agent.expectile).mean()
            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)
        agent = agent.replace(value=value)

        return agent, info

    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        next_v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )

        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            q1, q2 = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss, {
                "critic_loss": critic_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
            }

       

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        agent = agent.replace(critic=critic)

        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)

        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        return new_agent, info

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        v = agent.value.apply_fn({"params": agent.value.params}, batch["observations"])

        q1, q2 = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = jnp.minimum(q1, q2)
        exp_a = jnp.exp((q - v) * agent.temperature)
        exp_a = jnp.minimum(exp_a, 100.0)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = agent.actor.apply_fn(
                {"params": actor_params}, batch["observations"], training=True
            )

            log_probs = dist.log_prob(batch["actions"])
            
            #behavior loss
            #bc_loss = -log_probs.mean()
            actor_loss = -(exp_a * log_probs).mean() #+ bc_loss

            return actor_loss, {"actor_loss": actor_loss, "adv": q - v}
        #这里load的时候还是有点bug
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.actor.params)
        actor = agent.actor.apply_gradients(grads=grads)

        agent = agent.replace(actor=actor)

        return agent, info

    @jax.jit
    def update(self, batch: DatasetDict):

        new_agent = self

        # new_agent, critic_info = new_agent.update_v(batch)
        # new_agent, actor_info = new_agent.update_actor(batch)
        # new_agent, value_info = new_agent.update_q(batch)
    
        new_agent, value_info = new_agent.update_v(batch)
        new_agent, actor_info = new_agent.update_actor(batch)
        new_agent, critic_info = new_agent.update_q(batch)
        
        return new_agent, {**actor_info, **critic_info, **value_info}

    def save(self, path_dir):
        # 保存 JAX TrainState
        with open(path_dir  + "actor.msgpack", 'wb') as f:
            f.write(flax.serialization.to_bytes(self.actor))
        wandb.save('actor.msgpack')
        
        with open(path_dir  + "critic.msgpack", 'wb') as f:
            f.write(flax.serialization.to_bytes(self.critic))
        wandb.save('critic.msgpack')
        
        with open(path_dir  + "value.msgpack", 'wb') as f:
            f.write(flax.serialization.to_bytes(self.value))
        wandb.save('value.msgpack')
        
        with open(path_dir  + "target_critic.msgpack", 'wb') as f:
            f.write(flax.serialization.to_bytes(self.target_critic))
        wandb.save('target_critic.msgpack')
        
    def load(self, path_dir):
        #pdb.set_trace()
        #self.rng.load(path_dir  + rng)
        # self.actor = self.actor.load(path_dir  + "actor")
        # self.critic = self.critic.load(path_dir  + "critic")
        # self.value = self.value.load(path_dir  + "value")
        # self.target_critic = self.target_critic.load(path_dir  + "target_critic")
        # self.actor.params["MLP_0"]["Dense_0"]['kernel']
        agent = self
        
        with open(path_dir  + "actor.msgpack", 'rb') as f:
            state_dict = flax.serialization.msgpack_restore(f.read())
        actor = agent.actor.replace(params=state_dict["params"], opt_state=self.loadUtil(state_dict["opt_state"]), step=state_dict["step"])
        
        with open(path_dir  + "critic.msgpack", 'rb') as f:
            state_dict = flax.serialization.msgpack_restore(f.read())
        critic = agent.critic.replace(params=state_dict["params"], opt_state=self.loadUtil(state_dict["opt_state"]), step=state_dict["step"])
        
        with open(path_dir  + "value.msgpack", 'rb') as f:
            state_dict = flax.serialization.msgpack_restore(f.read())
        value = agent.value.replace(params=state_dict["params"], opt_state=self.loadUtil(state_dict["opt_state"]), step=state_dict["step"])
        
        # with open(path_dir  + "target_critic.msgpack", 'rb') as f:
        #     state_dict = flax.serialization.msgpack_restore(f.read())
        # target_critic = agent.target_critic.replace(params=state_dict["params"], opt_state=self.loadUtil(state_dict["opt_state"]), step=state_dict["step"])
        agent = agent.replace(actor=actor, critic=critic, value=value, target_critic=critic)
        return agent

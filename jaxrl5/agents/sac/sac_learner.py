"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import pdb
import jax.numpy as jnp
import optax
import flax
import wandb
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.agents.agent import Agent
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal, Normal
from jaxrl5.networks import MLP, Ensemble, StateActionValue, subsample_ensemble


class SACLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        
        #actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_def = Normal(
            actor_base_cls,
            action_dim,
            log_std_min= -5.0 ,
            log_std_max= 2.0,
            state_dependent_std=False,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=critic_optimiser,
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        # target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        # target_critic = TrainState.create(
        #     apply_fn=target_critic_def.apply,
        #     params=critic_params,
        #     tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        # )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
        )

    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:
        #pdb.set_trace()
        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        #pdb.set_trace()
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}

    # def save(self, path_dir):
    #     #self.rng.save(path_dir  + rng)
    #     self.actor.save(path_dir  + "actor.msgpack")
    #     self.critic.save(path_dir  + "critic.msgpack")
    #     #self.value.save(path_dir  + "value")
    #     self.target_critic.save(path_dir  + "target_critic.msgpack")
        
    # def load(self, path_dir):
    #     #self.rng.load(path_dir  + rng)
    #     self.actor = self.actor.load(path_dir  + "actor.msgpack")
    #     self.critic = self.critic.load(path_dir  + "critic.msgpack")
    #     #self.value = self.value.load(path_dir  + "value")
    #     self.target_critic = self.target_critic.load(path_dir  + "target_critic.msgpack")
        

    def save(self, path_dir):
        # 保存 JAX TrainState
        with open(path_dir  + "actor.msgpack", 'wb') as f:
            f.write(flax.serialization.to_bytes(self.actor))
        wandb.save('actor.msgpack')
        
        with open(path_dir  + "critic.msgpack", 'wb') as f:
            f.write(flax.serialization.to_bytes(self.critic))
        wandb.save('critic.msgpack')
        
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
        
        with open(path_dir  + "target_critic.msgpack", 'rb') as f:
            state_dict = flax.serialization.msgpack_restore(f.read())
        target_critic = agent.target_critic.replace(params=state_dict["params"], opt_state=state_dict["opt_state"], step=state_dict["step"])
        agent = agent.replace(actor=actor, critic=critic, target_critic=target_critic)
        return agent

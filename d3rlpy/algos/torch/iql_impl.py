from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import copy

from ...gpu import Device
from ...models.builders import (
    create_non_squashed_normal_policy,
    create_value_function,
    create_discrete_q_function,
    create_categorical_policy,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import MeanQFunctionFactory, QFunctionFactory
from ...models.torch import NonSquashedNormalPolicy, ValueFunction
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api, soft_sync
from .ddpg_impl import DDPGBaseImpl
from .utility import DiscreteQFunctionMixin
from .base import TorchImplBase


class IQLImpl(DDPGBaseImpl):
    _policy: Optional[NonSquashedNormalPolicy]
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _value_encoder_factory: EncoderFactory
    _value_func: Optional[ValueFunction]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        value_encoder_factory: EncoderFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        expectile: float,
        weight_temp: float,
        max_weight: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=MeanQFunctionFactory(),
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._value_encoder_factory = value_encoder_factory
        self._value_func = None

    def _build_actor(self) -> None:
        self._policy = create_non_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
        )

    def _build_critic(self) -> None:
        super()._build_critic()
        self._value_func = create_value_function(
            self._observation_shape, self._value_encoder_factory
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        assert self._value_func is not None
        q_func_params = list(self._q_func.parameters())
        v_func_params = list(self._value_func.parameters())
        self._critic_optim = self._critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._critic_learning_rate
        )

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._value_func
        with torch.no_grad():
            return self._value_func(batch.next_observations)

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy

        # compute log probability
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)

        return -(weight * log_probs).mean()

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_t = self._value_func(batch.observations)
        adv = q_t - v_t
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_t = self._value_func(batch.observations)
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        # compute Q-function loss
        q_tpn = self.compute_target(batch)
        q_loss = self.compute_critic_loss(batch, q_tpn)

        # compute value function loss
        v_loss = self.compute_value_loss(batch)

        loss = q_loss + v_loss

        loss.backward()
        self._critic_optim.step()

        return q_loss.cpu().detach().numpy(), v_loss.cpu().detach().numpy()


class DiscreteIQLImpl(DiscreteQFunctionMixin, TorchImplBase):
    _policy: Optional[NonSquashedNormalPolicy]
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _value_encoder_factory: EncoderFactory
    _value_func: Optional[ValueFunction]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        value_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        expectile: float,
        weight_temp: float,
        max_weight: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )

        # from ddpg init
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = actor_encoder_factory
        self._critic_encoder_factory = critic_encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._tau = tau
        self._n_critics = n_critics
        self._use_gpu = use_gpu

        self._recurrent_states = {}
        self._value_recurrent_state = {}
        self._next_value_recurrent_state = {}
        self._target_recurrent_states = {}
        self._policy_recurrent_state = {}

        # initialized in build
        self._q_func = None
        self._policy = None
        self._targ_q_func = None
        self._actor_optim = None
        self._critic_optim = None

        # not from super
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._value_encoder_factory = value_encoder_factory
        self._value_func = None

    def build(self) -> None:
        # setup torch models
        self._build_critic()
        self._build_actor()

        # setup target networks
        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()

    def _build_actor_optim(self) -> None:
        assert self._policy is not None
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )

    def _build_actor(self) -> None:
        self._policy = create_categorical_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def _build_critic(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._critic_encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )
        self._value_func = create_value_function(
            self._observation_shape, self._value_encoder_factory
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        assert self._value_func is not None
        q_func_params = list(self._q_func.parameters())
        v_func_params = list(self._value_func.parameters())
        self._critic_optim = self._critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._critic_learning_rate
        )

    def _recurrent_compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        idx = batch.observations["idx"].item()
        if idx not in self._recurrent_states:
            self._recurrent_states[idx] = self._q_func.get_initial_states(
                batch_size=batch.rewards.size(1),
                device=batch.rewards.device,
            )

        q_out, recurrent_states = self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
            recurrent_states=self._recurrent_states[idx],
        )
        self._recurrent_states[idx] = {
            i: (rs[0].detach(), rs[1].detach()) for i, rs in recurrent_states.items()
        }
        return q_out


    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        if "idx" in batch.observations:
            return self._recurrent_compute_critic_loss(batch, q_tpn)
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    def _recurrent_compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        idx = batch.observations["idx"].item()
        if idx not in self._value_recurrent_state:
            BS = batch.rewards.size(1)
            self._value_recurrent_state[idx] = self._value_func.get_initial_state(
                batch_size=BS,
                device=batch.rewards.device,
            )
            _, self._next_value_recurrent_state[idx] = self._value_func.forward(
                batch.observations,
                recurrent_state=self._value_recurrent_state[idx],
            )
        with torch.no_grad():
            v_out, self._next_value_recurrent_state[idx] = self._value_func.forward(
                batch.next_observations,
                recurrent_state=self._next_value_recurrent_state[idx],
            )
            return v_out

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._value_func
        if "idx" in batch.observations:
            return self._recurrent_compute_target(batch)
        with torch.no_grad():
            return self._value_func.forward(batch.next_observations)

    def _recurrent_compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        idx = batch.observations["idx"]
        if idx not in self._policy_recurrent_state:
            self._policy_recurrent_state[idx] = self._policy.get_initial_state(
                batch_size=batch.rewards.size(1),
                device=batch.rewards.device,
            )
        dist, policy_recurrent_state = self._policy.dist(
            batch.observations,
            recurrent_state=self._policy_recurrent_state[idx],
        )
        self._policy_recurrent_state[idx] = (
            policy_recurrent_state[0].detach(),
            policy_recurrent_state[1].detach(),
        )
        actions_reshape = batch.actions.view([-1])
        log_probs = dist.log_prob(actions_reshape).unsqueeze(1)

        with torch.no_grad():
            weight = self._compute_weight(batch)

        return -(weight * log_probs).mean()

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy
        if "idx" in batch.observations:
            return self._recurrent_compute_actor_loss(batch)

        dist = self._policy.dist(batch.observations)
        actions_reshape = batch.actions.view([-1])
        log_probs = dist.log_prob(actions_reshape).unsqueeze(1)

        with torch.no_grad():
            weight = self._compute_weight(batch)

        return -(weight * log_probs).mean()

    def _recurrent_compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        idx = batch.observations["idx"].item()

        q_t, self._target_recurrent_states[idx] = self._targ_q_func(
            batch.observations, reduction="min",
            recurrent_states=self._target_recurrent_states[idx],
        )
        one_hot = F.one_hot(batch.actions.view(-1), num_classes=q_t.size(1))
        q_value = (q_t * one_hot.float()).sum(dim=1, keepdim=True)

        v_t, self._value_recurrent_state[idx] = self._value_func.forward(
            batch.observations,
            recurrent_state=self._value_recurrent_state[idx],
        )
        adv = q_value - v_t
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        if "idx" in batch.observations:
            return self._recurrent_compute_weight(batch)
        q_t = self._targ_q_func(batch.observations, reduction="min")
        one_hot = F.one_hot(batch.actions.view(-1), num_classes=q_t.size(1))
        q_value = (q_t * one_hot.float()).sum(dim=1, keepdim=True)

        v_t = self._value_func.forward(batch.observations)
        adv = q_value - v_t
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def _recurrent_compute_value_loss(self, batch: TorchMiniBatch):
        idx = batch.observations["idx"].item()
        if idx not in self._target_recurrent_states:
            self._target_recurrent_states[idx] = self._targ_q_func.get_initial_states(
                batch_size=batch.rewards.size(1),
                device=batch.rewards.device,
            )
        q_t, _ = self._targ_q_func(
            batch.observations, reduction="min",
            recurrent_states=self._target_recurrent_states[idx],
        )
        v_t, _ = self._value_func.forward(
            batch.observations,
            recurrent_state=self._value_recurrent_state[idx],
        )
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        if "idx" in batch.observations:
            return self._recurrent_compute_value_loss(batch)
        q_t = self._targ_q_func(batch.observations, reduction="min")
        v_t = self._value_func.forward(batch.observations)
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    @train_api
    # @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    @train_api
    # @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        # compute Q-function loss
        q_tpn = self.compute_target(batch)
        q_loss = self.compute_critic_loss(batch, q_tpn)

        # compute value function loss
        v_loss = self.compute_value_loss(batch)

        loss = q_loss + v_loss

        loss.backward()
        self._critic_optim.step()

        return q_loss.cpu().detach().numpy(), v_loss.cpu().detach().numpy()

    def update_critic_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        soft_sync(self._targ_q_func, self._q_func, self._tau)



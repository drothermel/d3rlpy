import copy
from typing import Optional, Sequence

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import create_discrete_q_function
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import EnsembleDiscreteQFunction, EnsembleQFunction
from ...preprocessing import RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, hard_sync, torch_api, train_api
from .base import TorchImplBase
from .utility import DiscreteQFunctionMixin


class DQNImpl(DiscreteQFunctionMixin, TorchImplBase):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _n_critics: int
    _use_gpu: Optional[Device]
    _q_func: Optional[EnsembleDiscreteQFunction]
    _targ_q_func: Optional[EnsembleDiscreteQFunction]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],
        grad_clip: float = 5.0,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=None,
            reward_scaler=reward_scaler,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._n_critics = n_critics
        self._use_gpu = use_gpu
        self._grad_clip = grad_clip
        self._recurrent_states = {}
        self._next_recurrent_states = {} # Only used for DoubleDQN
        self._next_target_recurrent_states = {}

        # initialized in build
        self._q_func = None
        self._targ_q_func = None
        self._optim = None

    def build(self) -> None:
        # setup torch models
        self._build_network()

        # setup target network
        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_optim()

    def _build_network(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_optim(self) -> None:
        assert self._q_func is not None
        self._optim = self._optim_factory.create(
            self._q_func.parameters(), lr=self._learning_rate
        )

    @train_api
    #@torch_api(scaler_targets=["obs_t", "obs_tpn"])
    def update(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._optim is not None
        self._optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_loss(batch, q_tpn)

        loss.backward()
        unclipped_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._q_func.parameters(), self._grad_clip,
        )
        self._optim.step()

        return loss.cpu().detach().numpy(), unclipped_grad_norm.cpu().numpy()

    def _compute_recurrent_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        idx = batch.observations["idx"].item()
        BS = batch.rewards.size(1)
        if idx not in self._recurrent_states:
            self._recurrent_states[idx] = self._q_func.get_initial_states(
                batch_size=BS,
                device=batch.rewards.device,
            )
        q_error, self._recurrent_states[idx] = self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
            recurrent_states=self._recurrent_states[idx],
        )
        return q_error

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        if "idx" in batch.observations:
            return self._compute_recurrent_loss(batch, q_tpn)
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    def _compute_recurrent_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            idx = batch.observations["idx"].item()
            BS = batch.rewards.size(1)
            if idx not in self._next_target_recurrent_states:
                first_target_recurrent_states = self._targ_q_func.get_initial_states(
                    batch_size=BS,
                    device=batch.rewards.device,
                )
                _, self._next_target_recurrent_states[idx]  = self._targ_q_func(
                    batch.observations,
                    reduction="mean",
                    recurrent_state=first_target_recurrent_states,
                )

            next_actions, next_target_recurrent_states = self._targ_q_func(
                batch.next_observations,
                recurrent_states=self._next_target_recurrent_states[idx],
            )
            max_action = next_actions.argmax(dim=1)
            q_out = self._targ_q_func.compute_target(
                batch.next_observations,
                max_action,
                reduction="min",
                recurrent_states=self._next_target_recurrent_states[idx],
            )
            self._next_target_recurrent_states[idx] = next_target_recurrent_states
            return q_out

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        if "idx" in batch.observations:
            return self._compute_recurrent_target(batch)

        with torch.no_grad():
            next_actions = self._targ_q_func(batch.next_observations)
            max_action = next_actions.argmax(dim=1)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                max_action,
                reduction="min",
            )

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func(x).argmax(dim=1)

    def _recurrent_predict_best_action(
        self,
        x: torch.Tensor,
        recurrent_states: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        assert self._q_func is not None
        q_out, next_recurrent_states = self._q_func(x, recurrent_states=recurrent_states)
        action = q_out.argmax(dim=1)
        return action, next_recurrent_states

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)

    def update_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        hard_sync(self._targ_q_func, self._q_func)
        self._next_target_recurrent_states = {
            k: v for k, v in self._next_recurrent_states.items()
        }

    @property
    def q_function(self) -> EnsembleQFunction:
        assert self._q_func
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        assert self._optim
        return self._optim


class DoubleDQNImpl(DQNImpl):
    def _compute_recurrent_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        idx = batch.observations["idx"].item()
        q_error, self._recurrent_states[idx] = self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
            recurrent_states=self._recurrent_states[idx],
        )
        return q_error

    def _compute_recurrent_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            idx = batch.observations["idx"].item()
            BS = batch.rewards.size(1)
            # The target model forward only ever uses the next_observation
            if idx not in self._next_target_recurrent_states:
                first_target_recurrent_states = self._targ_q_func.get_initial_states(
                    batch_size=BS,
                    device=batch.rewards.device,
                )
                _, next_target_recurrent_states  = self._targ_q_func(
                    batch.observations,
                    reduction="mean",
                    recurrent_states=first_target_recurrent_states,
                )
                self._next_target_recurrent_states[idx] = next_target_recurrent_states

            if idx not in self._recurrent_states:
                self._recurrent_states[idx] = self._q_func.get_initial_states(
                    batch_size=BS,
                    device=batch.rewards.device,
                )
                _, self._next_recurrent_states[idx] = self._q_func(
                    batch.observations,
                    recurrent_states=self._recurrent_states[idx],
                )

            # Predict best action uses the learning model
            action, next_next_recurrent_states = self._recurrent_predict_best_action(
                batch.next_observations,
                recurrent_states=self._next_recurrent_states[idx],
            )
            self._recurrent_states[idx] = self._next_recurrent_states[idx]
            self._next_recurrent_states[idx] = next_next_recurrent_states

            # Compute target uses the target model
            q_out = self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
                recurrent_states=self._next_target_recurrent_states[idx],
            )

            _, self._next_target_recurrent_states[idx] = self._targ_q_func(
                batch.next_observations,
                recurrent_states=self._next_target_recurrent_states[idx],
            )
            return q_out

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        if "idx" in batch.observations:
            return self._compute_recurrent_target(batch)
        with torch.no_grad():
            action = self._predict_best_action(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )

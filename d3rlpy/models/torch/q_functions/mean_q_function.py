from typing import Optional, cast, Union, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..encoders import Encoder, EncoderWithAction
from .base import ContinuousQFunction, DiscreteQFunction
from .utility import compute_huber_loss, compute_reduce, pick_value_by_action


class DiscreteMeanQFunction(DiscreteQFunction, nn.Module):  # type: ignore
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if recurrent_state is not None:
            out, recurrent_state = self._encoder(x, recurrent_state)
            return self._fc(out), recurrent_state
        return cast(torch.Tensor, self._fc(self._encoder(x)))

    def compute_error(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        one_hot = F.one_hot(actions.view(-1), num_classes=self.action_size)
        if recurrent_state is not None:
            q_out, new_recurrent_state = self.forward(observations, recurrent_state)
        else:
            q_out = self.forward(observations)
        value = (q_out * one_hot.float()).sum(dim=1, keepdim=True)
        # Necessary for observations with shape [T, B, ...]
        if rewards.size(1) != 1:
            rewards = rewards.reshape(-1, 1)
            terminals = terminals.reshape(-1, 1)

        y = rewards + gamma * target * (1 - terminals)
        loss = compute_huber_loss(value, y)
        return compute_reduce(loss, reduction), new_recurrent_state

    def compute_target(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if recurrent_state is None:
            q_out = self.forward(x)
            if action is None:
                return q_out
            return pick_value_by_action(q_out, action, keepdim=True)
        else:
            q_out, recurrent_state = self.forward(x, recurrent_state)
            if action is None:
                return q_out, recurrent_state
            return pick_value_by_action(q_out, action, keepdim=True), recurrent_state


    def best_action(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if recurrent_state is not None:
            q_out, recurrent_state = self.forward(x, recurrent_state)
        else:
            q_out = self.forward(x)

        action = q_out.argmax(dim=1)
        if recurrent_state is not None:
            return action, recurrent_state
        return action

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousMeanQFunction(ContinuousQFunction, nn.Module):  # type: ignore
    _encoder: EncoderWithAction
    _action_size: int
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction):
        super().__init__()
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), 1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._fc(self._encoder(x, action)))

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        value = self.forward(observations, actions)
        y = rewards + gamma * target * (1 - terminals)
        loss = F.mse_loss(value, y, reduction="none")
        return compute_reduce(loss, reduction)

    def compute_target(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(x, action)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder

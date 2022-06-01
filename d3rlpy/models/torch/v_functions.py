from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import Encoder


class ValueFunction(nn.Module):  # type: ignore
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), 1)

    def get_initial_state(self, batch_size, device):
        initial_state = self._encoder.initial_state(batch_size=batch_size)
        initial_state = (initial_state[0].to(device), initial_state[1].to(device))
        return initial_state

    def forward(
        self, x: torch.Tensor,
        recurrent_state=None,
    ) -> torch.Tensor:
        if recurrent_state is None:
            h = self._encoder(x)
            return cast(torch.Tensor, self._fc(h))
        else:
            h, recurrent_state = self._encoder(x, recurrent_state)
            return self._fc(h), recurrent_state

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))

    def compute_error(
        self, observations: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        v_t = self.forward(observations)
        loss = F.mse_loss(v_t, target)
        return loss

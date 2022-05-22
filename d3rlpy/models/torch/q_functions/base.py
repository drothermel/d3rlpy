from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Dict, Tuple

import torch

from ..encoders import Encoder, EncoderWithAction


class QFunction(metaclass=ABCMeta):
    @abstractmethod
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
        pass

    @property
    def action_size(self) -> int:
        pass


class DiscreteQFunction(QFunction):
    @abstractmethod
    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass

    @abstractmethod
    def compute_target(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: Optional[torch.Tensor],
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pass

    def __call__(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward(x)

    @property
    def encoder(self) -> Encoder:
        pass


class ContinuousQFunction(QFunction):
    @abstractmethod
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(x, action)

    @property
    def encoder(self) -> EncoderWithAction:
        pass

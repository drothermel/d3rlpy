from typing import List, Optional, Union, cast, Dict, Tuple, Sequence

import torch
from torch import nn

from .base import ContinuousQFunction, DiscreteQFunction


def _reduce_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    if reduction == "min":
        return y.min(dim=dim).values
    elif reduction == "max":
        return y.max(dim=dim).values
    elif reduction == "mean":
        return y.mean(dim=dim)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def _gather_quantiles_by_indices(
    y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # TODO: implement this in general case
    if y.dim() == 3:
        # (N, batch, n_quantiles) -> (batch, n_quantiles)
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif y.dim() == 4:
        # (N, batch, action, n_quantiles) -> (batch, action, N, n_quantiles)
        transposed_y = y.transpose(0, 1).transpose(1, 2)
        # (batch, action, N, n_quantiles) -> (batch * action, N, n_quantiles)
        flat_y = transposed_y.reshape(-1, y.shape[0], y.shape[3])
        head_indices = torch.arange(y.shape[1] * y.shape[2])
        # (batch * action, N, n_quantiles) -> (batch * action, n_quantiles)
        gathered_y = flat_y[head_indices, indices.view(-1)]
        # (batch * action, n_quantiles) -> (batch, action, n_quantiles)
        return gathered_y.view(y.shape[1], y.shape[2], -1)
    raise ValueError


def _reduce_quantile_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    # reduction beased on expectation
    mean = y.mean(dim=-1)
    if reduction == "min":
        indices = mean.min(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "max":
        indices = mean.max(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        min_indices = mean.min(dim=dim).indices
        max_indices = mean.max(dim=dim).indices
        min_values = _gather_quantiles_by_indices(y, min_indices)
        max_values = _gather_quantiles_by_indices(y, max_indices)
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


class EnsembleQFunction(nn.Module):  # type: ignore
    _action_size: int
    _q_funcs: nn.ModuleList

    def __init__(
        self,
        q_funcs: Union[List[DiscreteQFunction], List[ContinuousQFunction]],
    ):
        super().__init__()
        self._action_size = q_funcs[0].action_size
        self._q_funcs = nn.ModuleList(q_funcs)

    def get_initial_states(self, batch_size, device):
        initial_states = []
        for q in self._q_funcs:
            initial_state = q._encoder.initial_state(batch_size=batch_size)
            initial_state = (initial_state[0].to(device), initial_state[1].to(device))
            initial_states.append(initial_state)
        return initial_states

    def compute_error(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        recurrent_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        assert target.ndim == 2
        device = (
            observations.device
            if isinstance(observations, torch.Tensor)
            else next(iter(observations.values())).device
        )

        td_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
        if recurrent_states is not None:
            new_recurrent_states = {}
            for i, q_func in enumerate(self._q_funcs):
                loss, new_recurrent_states[i] = q_func.compute_error(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    target=target,
                    terminals=terminals,
                    gamma=gamma,
                    reduction="none",
                    recurrent_state=recurrent_states[i],
                )
                td_sum += loss.mean()
            return td_sum, new_recurrent_states

        for q_func in self._q_funcs:
            loss = q_func.compute_error(
                observations=observations,
                actions=actions,
                rewards=rewards,
                target=target,
                terminals=terminals,
                gamma=gamma,
                reduction="none",
                recurrent_state=recurrent_state,
            )
            td_sum += loss.mean()
        return td_sum

    def _compute_target(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
        recurrent_states: Optional[torch.Tensor] = None,
        return_recurrent: bool = False,
    ) -> torch.Tensor:
        Bin = (
            x.size(0) if isinstance(x, torch.Tensor) else next(iter(x.values())).size(0)
        )

        values_list: List[torch.Tensor] = []
        new_recurrent_states = {}
        for i, q_func in enumerate(self._q_funcs):
            if recurrent_states is not None:
                target, new_recurrent_states[i] = q_func.compute_target(
                    x, action, recurrent_states[i],
                )
            else:
                target = q_func.compute_target(x, action)
            Bout = target.size(0)
            assert Bout % Bin == 0
            values_list.append(target.reshape(1, Bout, -1))

        values = torch.cat(values_list, dim=0)

        if action is None:
            # mean Q function
            if values.shape[2] == self._action_size:
                if return_recurrent:
                    return _reduce_ensemble(values, reduction), new_recurrent_states
                return _reduce_ensemble(values, reduction)
            # distributional Q function
            Bout = values.shape[1]
            n_q_funcs = values.shape[0]
            values = values.view(n_q_funcs, Bout, self._action_size, -1)
            print("BEWARE: THIS MIGHT NOT WORK, this codepath hasn't been tested")
            if return_recurrent:
                return _reduce_quantile_ensemble(values, reduction), new_recurrent_states
            return _reduce_quantile_ensemble(values, reduction)

        if values.shape[2] == 1:
            if return_recurrent:
                return _reduce_ensemble(values, reduction, lam=lam), new_recurrent_states
            return _reduce_ensemble(values, reduction, lam=lam)

        if return_recurrent:
            return _reduce_quantile_ensemble(values, reduction, lam=lam), new_recurrent_states
        return _reduce_quantile_ensemble(values, reduction, lam=lam)

    @property
    def q_funcs(self) -> nn.ModuleList:
        return self._q_funcs


class EnsembleDiscreteQFunction(EnsembleQFunction):
    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        reduction: str = "mean",
        recurrent_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        Bin = (
            x.size(0) if isinstance(x, torch.Tensor) else next(iter(x.values())).size(0)
        )
        values = []
        next_recurrent_states = {}
        for i, q_func in enumerate(self._q_funcs):
            if recurrent_states is not None:
                q_out, next_recurrent_states[i] = q_func(
                    x, recurrent_state=recurrent_states[i],
                )
            else:
                q_out = q_func(x)
            Bout = q_out.size(0)
            assert Bout % Bin == 0
            q_out = q_out.view(1, Bout, self._action_size)
            values.append(q_out)
        reduced_q_out = _reduce_ensemble(torch.cat(values, dim=0), reduction)
        if recurrent_states is not None:
            return reduced_q_out, next_recurrent_states
        return reduced_q_out

    def __call__(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        reduction: str = "mean",
        recurrent_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        return super().__call__(x, reduction=reduction, recurrent_states=recurrent_states)

    def compute_target(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
        recurrent_states: Optional[torch.Tensor] = None,
        return_recurrent: bool = False,
    ) -> torch.Tensor:
        return self._compute_target(
            x,
            action=action,
            reduction=reduction,
            lam=lam,
            recurrent_states=recurrent_states,
            return_recurrent=return_recurrent,
        )


class EnsembleContinuousQFunction(EnsembleQFunction):
    def forward(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x, action).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action, reduction))

    def compute_target(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, action, reduction, lam)

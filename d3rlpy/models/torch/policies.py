import math
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, cast, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from .distributions import GaussianDistribution, SquashedGaussianDistribution
from .encoders import Encoder, EncoderWithAction


def squash_action(
    dist: torch.distributions.Distribution, raw_action: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    squashed_action = torch.tanh(raw_action)
    jacob = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
    log_prob = (dist.log_prob(raw_action) - jacob).sum(dim=-1, keepdims=True)
    return squashed_action, log_prob


class Policy(nn.Module, metaclass=ABCMeta):  # type: ignore
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        return self.sample_with_log_prob(x)[0]

    @abstractmethod
    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def sample_n(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return self.sample_n_with_log_prob(x, n)[0]

    @abstractmethod
    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DeterministicPolicy(Policy):

    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        return torch.tanh(self._fc(h))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("deterministic policy does not support sample")

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("deterministic policy does not support sample_n")

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class DeterministicResidualPolicy(Policy):

    _encoder: EncoderWithAction
    _scale: float
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, scale: float):
        super().__init__()
        self._scale = scale
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), encoder.action_size)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        residual_action = self._scale * torch.tanh(self._fc(h))
        return (action + cast(torch.Tensor, residual_action)).clamp(-1.0, 1.0)

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    def best_residual_action(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(x, action)

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("residual policy does not support best_action")

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("deterministic policy does not support sample")

    def sample_n_with_log_prob(
        self, x: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("deterministic policy does not support sample_n")


class NormalPolicy(Policy):

    _encoder: Encoder
    _action_size: int
    _min_logstd: float
    _max_logstd: float
    _use_std_parameter: bool
    _mu: nn.Linear
    _logstd: Union[nn.Linear, nn.Parameter]

    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
        use_std_parameter: bool,
        squash_distribution: bool,
    ):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._use_std_parameter = use_std_parameter
        self._squash_distribution = squash_distribution
        self._mu = nn.Linear(encoder.get_feature_size(), action_size)
        if use_std_parameter:
            initial_logstd = torch.zeros(1, action_size, dtype=torch.float32)
            self._logstd = nn.Parameter(initial_logstd)
        else:
            self._logstd = nn.Linear(encoder.get_feature_size(), action_size)

    def _compute_logstd(self, h: torch.Tensor) -> torch.Tensor:
        if self._use_std_parameter:
            clipped_logstd = self.get_logstd_parameter()
        else:
            logstd = cast(nn.Linear, self._logstd)(h)
            clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return clipped_logstd

    def dist(
        self, x: torch.Tensor
    ) -> Union[GaussianDistribution, SquashedGaussianDistribution]:
        h = self._encoder(x)
        mu = self._mu(h)
        clipped_logstd = self._compute_logstd(h)
        if self._squash_distribution:
            return SquashedGaussianDistribution(mu, clipped_logstd.exp())
        else:
            return GaussianDistribution(
                torch.tanh(mu),
                clipped_logstd.exp(),
                raw_loc=mu,
            )

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dist = self.dist(x)
        if deterministic:
            action, log_prob = dist.mean_with_log_prob()
        else:
            action, log_prob = dist.sample_with_log_prob()
        return (action, log_prob) if with_log_prob else action

    def sample_with_log_prob(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def sample_n_with_log_prob(
        self,
        x: torch.Tensor,
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x)

        action_T, log_prob_T = dist.sample_n_with_log_prob(n)

        # (n, batch, action) -> (batch, n, action)
        transposed_action = action_T.transpose(0, 1)
        # (n, batch, 1) -> (batch, n, 1)
        log_prob = log_prob_T.transpose(0, 1)

        return transposed_action, log_prob

    def sample_n_without_squash(self, x: torch.Tensor, n: int) -> torch.Tensor:
        dist = self.dist(x)
        action = dist.sample_n_without_squash(n)
        return action.transpose(0, 1)

    def onnx_safe_sample_n(self, x: torch.Tensor, n: int) -> torch.Tensor:
        h = self._encoder(x)
        mean = self._mu(h)
        std = self._compute_logstd(h).exp()

        if not self._squash_distribution:
            mean = torch.tanh(mean)

        # expand shape
        # (batch_size, action_size) -> (batch_size, N, action_size)
        expanded_mean = mean.view(-1, 1, self._action_size).repeat((1, n, 1))
        expanded_std = std.view(-1, 1, self._action_size).repeat((1, n, 1))

        # sample noise from Gaussian distribution
        noise = torch.randn(x.shape[0], n, self._action_size, device=x.device)

        if self._squash_distribution:
            return torch.tanh(expanded_mean + noise * expanded_std)
        else:
            return expanded_mean + noise * expanded_std

    def best_action(self, x: torch.Tensor) -> torch.Tensor:
        action = self.forward(x, deterministic=True, with_log_prob=False)
        return cast(torch.Tensor, action)

    def get_logstd_parameter(self) -> torch.Tensor:
        assert self._use_std_parameter
        logstd = torch.sigmoid(cast(nn.Parameter, self._logstd))
        base_logstd = self._max_logstd - self._min_logstd
        return self._min_logstd + logstd * base_logstd


class SquashedNormalPolicy(NormalPolicy):
    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
        use_std_parameter: bool,
    ):
        super().__init__(
            encoder=encoder,
            action_size=action_size,
            min_logstd=min_logstd,
            max_logstd=max_logstd,
            use_std_parameter=use_std_parameter,
            squash_distribution=True,
        )


class NonSquashedNormalPolicy(NormalPolicy):
    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
        use_std_parameter: bool,
    ):
        super().__init__(
            encoder=encoder,
            action_size=action_size,
            min_logstd=min_logstd,
            max_logstd=max_logstd,
            use_std_parameter=use_std_parameter,
            squash_distribution=False,
        )


class CategoricalPolicy(Policy):

    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def dist(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Union[Categorical, Tuple[Categorical, torch.Tensor]]:
        if recurrent_state is not None:
            h, recurrent_state = self._encoder(x, recurrent_state)
        else:
            h = self._encoder(x)
        h = self._fc(h)
        out = Categorical(torch.softmax(h, dim=1))
        if recurrent_state is not None:
            return out, recurrent_state
        return out

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        deterministic: bool = False,
        with_log_prob: bool = False,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if recurrent_state is not None:
            dist, recurrent_state = self.dist(x, recurrent_state)
        else:
            dist = self.dist(x)

        if deterministic:
            action = cast(torch.Tensor, dist.probs.argmax(dim=1))
        else:
            action = cast(torch.Tensor, dist.sample())

        out = [action]
        if with_log_prob:
            out.append(dist.log_prob(action))

        if recurrent_state is not None:
            out.append(recurrent_state)

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def sample_with_log_prob(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        return self.forward(x, with_log_prob=True, recurrent_state=recurrent_state)

    def sample_n_with_log_prob(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        n: int,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if recurrent_state is not None:
            dist, recurrent_state = self.dist(x, recurrent_state)
        else:
            dist = self.dist(x)

        action_T = cast(torch.Tensor, dist.sample((n,)))
        log_prob_T = dist.log_prob(action_T)

        # (n, batch) -> (batch, n)
        action = action_T.transpose(0, 1)
        # (n, batch) -> (batch, n)
        log_prob = log_prob_T.transpose(0, 1)

        if recurrent_state is not None:
            return action, log_prob, recurrent_state
        return action, log_prob

    def best_action(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward(x, deterministic=True, recurrent_state=recurrent_state)

    def log_probs(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if recurrent_state is None:
            dist = self.dist(x)
            return dist.logits
        else:
            dist, recurrent_state = self.dist(x, recurrent_state)
            return dist.logits, recurrent_state

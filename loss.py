import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianMixtureNLLLoss(nn.Module):
    """Gaussian mixture NLL loss."""

    def __init__(self) -> None:
        """Initialize the module."""

        super().__init__()

    def forward(
        self, 
        location: torch.Tensor,  # Component locations.
        scale: torch.Tensor,     # Component scales.
        weight: torch.Tensor,    # Component weights.
        target: torch.Tensor,    # Targets/labels.
    ) -> torch.Tensor:
        """Forward pass."""

        distributions = Normal(location, scale)
        probabilities = torch.exp(distributions.log_prob(target))
        probabilities = torch.sum(probabilities * weight, dim=-1)

        loss = -torch.log(probabilities + 1e-9).mean()

        return loss

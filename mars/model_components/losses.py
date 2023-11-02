import torch
from jaxtyping import Float
from torch import Tensor

from nerfstudio.model_components.losses import ScaleAndShiftInvariantLoss


def monosdf_depth_loss(
    termination_depth: Float[Tensor, "*batch 1"],
    predicted_depth: Float[Tensor, "*batch 1"],
    directions_norm: Float[Tensor, "*batch 1"],
    is_euclidean: bool,
):
    """MonoSDF depth loss"""
    if not is_euclidean:
        termination_depth = termination_depth * directions_norm
    sift_depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
    mask = torch.ones_like(termination_depth).reshape(1, 32, -1).bool()
    return sift_depth_loss(predicted_depth.reshape(1, 32, -1), (termination_depth * 50 + 0.5).reshape(1, 32, -1), mask)

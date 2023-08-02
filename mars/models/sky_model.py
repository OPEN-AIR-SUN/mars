"""
Sky NeRF Model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import tinycudann as tcnn
import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig


@dataclass
class SkyModelConfig(ModelConfig):
    """Sky Model Config"""

    _target: Type = field(default_factory=lambda: SkyModel)
    hidden_dim: int = 128
    """hidden dimension of the MLP"""
    num_layers: int = 5
    """number of layers of the MLP"""


class SkyModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: SkyModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.field = tcnn.Network(
            n_input_dims=3,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": self.config.hidden_dim,
                "n_hidden_layers": self.config.num_layers - 1,
            },
        )

    def num_sample_points(self) -> int:
        return 1

    def get_param_groups(self):
        param_groups = []
        param_groups += list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        return callbacks

    def inference_without_render(self, ray_bundle: RayBundle):
        rays_d = ray_bundle.directions.requires_grad_()

        outputs = {
            "rgb": self.field(rays_d).float(),
        }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        raise Exception("should not call this method")

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        raise Exception("should not call this method")

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        raise Exception("should not call this method")

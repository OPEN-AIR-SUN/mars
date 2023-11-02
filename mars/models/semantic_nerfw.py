# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Semantic NeRF-W implementation which should be fast enough to view in the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    SemanticRenderer,
    UncertaintyRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model
from mars.fields.nerfacto_field import NerfactoField
from mars.models.nerfacto import NerfactoModelConfig


@dataclass
class SemanticNerfWModelConfig(NerfactoModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: SemanticNerfWModel)
    use_transient_embedding: bool = False
    """Whether to use transient embedding."""
    semantic_loss_weight: float = 1.0
    pass_semantic_gradients: bool = False


class SemanticNerfWModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: SemanticNerfWModelConfig

    def __init__(self, config: SemanticNerfWModelConfig, object_meta: Dict, **kwargs) -> None:
        assert "semantics" in object_meta.keys() and isinstance(object_meta["semantics"], Semantics)
        self.semantics = object_meta["semantics"]
        self.semantic_num = len(object_meta["semantics"].classes)
        super().__init__(config=config, **kwargs)
        self.colormap = self.semantics.colors.clone().detach().to(self.device)
        self.color2label = {tuple(color.tolist()): i for i, color in enumerate(self.colormap)}
        # map Van to Car
        for i, sem_class in enumerate(self.semantics.classes):
            if sem_class == "Car":
                self.color2label[(0, 139, 139)] = i
        self.str2semantic = {label: i for i, label in enumerate(self.semantics.classes)}

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        scene_contraction = SceneContraction(order=float("inf"))

        if self.config.use_transient_embedding:
            raise ValueError("Transient embedding is not fully working for semantic nerf-w.")

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            use_transient_embedding=self.config.use_transient_embedding,
            use_semantics=True,
            num_semantic_classes=len(self.semantics.classes),
            pass_semantic_gradients=self.config.pass_semantic_gradients,
        )

        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
            self.proposal_networks.append(network)
            self.density_fns = [network.density_fn for _ in range(self.config.num_proposal_iterations)]
        else:
            for _ in range(self.config.num_proposal_iterations):
                network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction)
                self.proposal_networks.append(network)
            self.density_fns = [network.density_fn for network in self.proposal_networks]

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # Samplers
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_uncertainty = UncertaintyRenderer()
        self.renderer_semantics = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=self.semantic_num)

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = []
        param_groups += list(self.proposal_networks.parameters())
        param_groups += list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples)

        if self.training and self.config.use_transient_embedding:
            density = field_outputs[FieldHeadNames.DENSITY] + field_outputs[FieldHeadNames.TRANSIENT_DENSITY]
            weights = ray_samples.get_weights(density)
            weights_static = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            rgb_static_component = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            rgb_transient_component = self.renderer_rgb(
                rgb=field_outputs[FieldHeadNames.TRANSIENT_RGB], weights=weights
            )
            rgb = rgb_static_component + rgb_transient_component
        else:
            weights_static = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            weights = weights_static
            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        weights_list.append(weights_static)
        ray_samples_list.append(ray_samples)

        depth = self.renderer_depth(weights=weights_static, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights_static)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}
        outputs["weights_list"] = weights_list
        outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        # transients
        if self.training and self.config.use_transient_embedding:
            weights_transient = ray_samples.get_weights(field_outputs[FieldHeadNames.TRANSIENT_DENSITY])
            uncertainty = self.renderer_uncertainty(field_outputs[FieldHeadNames.UNCERTAINTY], weights_transient)
            outputs["uncertainty"] = uncertainty + 0.03  # NOTE(ethan): this is the uncertainty min
            outputs["density_transient"] = field_outputs[FieldHeadNames.TRANSIENT_DENSITY]

        # semantics
        semantic_weights = weights_static
        if not self.config.pass_semantic_gradients:
            semantic_weights = semantic_weights.detach()
        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=semantic_weights
        )

        # semantics colormaps
        # semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        # outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        raise Exception("should not call this method")

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        raise Exception("should not call this method")

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        raise Exception("should not call this method")

    def inference_without_render(self, ray_bundle: RayBundle):
        # sample
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        # field function
        field_outputs = self.field(ray_samples)
        final_weight = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(final_weight)
        ray_samples_list.append(ray_samples)

        return {
            "ray_samples_list": ray_samples_list,
            "field_outputs": field_outputs,
            "weights_list": weights_list,
            "ray_samples": [ray_samples],
        }

    def num_sample_points(self):
        return self.config.num_nerf_samples_per_ray

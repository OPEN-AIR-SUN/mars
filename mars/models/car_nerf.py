# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Car NeRF for object models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type

import torch
from torch import nn
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from mars.fields.car_nerf_field import CarNeRF_Field
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig


@dataclass
class CarNeRFModelConfig(ModelConfig):
    """CarNeRF Model Config"""

    _target: Type = field(default_factory=lambda: CarNeRF)
    num_coarse_samples: int = 32
    num_fine_samples: int = 97
    background_color: str = "black"
    optimize_latents: bool = False


class CarNeRF(Model):
    """CarNeRF model

    Args:
        config: CarNeRF configuration to instantiate model
    """

    config: CarNeRFModelConfig
    object_meta: Dict
    obj_feat_dim: Optional[int]

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.object_meta = self.kwargs["object_meta"]
        self.obj_feat_dim = self.kwargs["obj_feat_dim"]
        self.car_latents = self.kwargs["car_latents"]

        # whether to optimize the car_latents
        for idx, _ in self.car_latents.items():
            self.car_latents[idx] = nn.Parameter(
                self.car_latents[idx], requires_grad=self.training and self.config.optimize_latents
            )

        self.fields = CarNeRF_Field()
        if "car_nerf_state_dict_path" in self.kwargs:
            self.fields.load_state_dict(torch.load(self.kwargs["car_nerf_state_dict_path"]))

        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_fine_samples, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def num_sample_points(self) -> int:
        return self.config.num_coarse_samples + self.config.num_fine_samples + 1

    def get_param_groups(self):
        param_groups = []
        if self.fields is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups += list(self.fields.parameters())
        if self.config.optimize_latents:
            param_groups += list(self.car_latents.values())
        return param_groups

    # def get_training_callbacks(
    #     self, training_callback_attributes: TrainingCallbackAttributes
    # ) -> List[TrainingCallback]:
    #     callbacks = []
    #     # if self.config.use_proposal_weight_anneal:
    #     #     # anneal the weights of the proposal network before doing PDF sampling
    #     #     N = self.config.proposal_weights_anneal_max_num_iters

    #     # def set_anneal(step):
    #     #     # https://arxiv.org/pdf/2111.12077.pdf eq. 18
    #     #     train_frac = np.clip(step / N, 0, 1)
    #     #     bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
    #     #     anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
    #     #     self.proposal_sampler.set_anneal(anneal)

    #     # callbacks.append(
    #     #     TrainingCallback(
    #     #         where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
    #     #         update_every_num_iters=1,
    #     #         func=set_anneal,
    #     #     )
    #     # )
    #     # callbacks.append(
    #     #     TrainingCallback(
    #     #         where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
    #     #         update_every_num_iters=1,
    #     #         func=self.proposal_sampler.step_cb,
    #     #     )
    #     # )
    #     return callbacks

    # def inference(self, ray_bundle: RayBundle):
    #     ray_samples_uniform = self.sampler_uniform(ray_bundle)

    #     gaussian_samples = ray_samples_uniform.frustums.get_gaussian_blob()
    #     field_outputs_coares = self.fields(
    #         gaussian_samples.mean, latents, viewdirs=ray_samples_uniform.frustums.directions, covs=gaussian_samples.cov
    #     )
    #     rgbs, sigmas = field_outputs_coares[FieldHeadNames.RGB], field_outputs_coares[FieldHeadNames.DENSITY]
    #     weights_coarse = ray_samples_uniform.get_weights(sigmas)

    #     # pdf sampling
    #     ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

    #     # second pass
    #     field_outputs_fine = self.fields(ray_samples_pdf)
    #     rgbs_fine, sigmas_fine = field_outputs_fine[FieldHeadNames.RGB], field_outputs_fine[FieldHeadNames.DENSITY]
    #     weights_fine = ray_samples_pdf.get_weights(sigmas_fine)

    #     return {
    #         "ray_samples": [ray_samples_uniform, ray_samples_pdf],
    #         "weights": [weights_coarse, weights_fine],
    #         "field_outputs": [field_outputs_coares, field_outputs_fine],
    #     }

    def inference_without_render(self, ray_bundle: RayBundle):
        """
        inference without render
        """
        if self.fields is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        obj_ids = ray_bundle.metadata["obj_ids"]
        unique_obj_ids = torch.unique(obj_ids)

        # generate empty ray samples
        weights_coarse_raw = torch.zeros(
            (*ray_bundle.shape, self.config.num_coarse_samples, 1), device=ray_bundle.camera_indices.device
        )

        field_outputs_fine_raw = {
            FieldHeadNames.DENSITY: torch.zeros(
                (*ray_bundle.shape, self.config.num_fine_samples, 1), device=ray_bundle.camera_indices.device
            ),
            FieldHeadNames.RGB: torch.zeros(
                (*ray_bundle.shape, self.config.num_fine_samples, 3), device=ray_bundle.camera_indices.device
            ),
        }

        for obj_id in unique_obj_ids:
            obj_msk = (obj_ids == obj_id).squeeze(-1)
            # uniform sampling
            ray_samples_uniform = self.sampler_uniform(ray_bundle[obj_msk])

            # First pass:
            gaussian_samples = ray_samples_uniform.frustums.get_gaussian_blob()

            field_outputs_coarse = self.fields(
                gaussian_samples.mean.view(1, -1, 3),
                self.car_latents[int(obj_id)].view(1, -1).to(obj_id.device),
                viewdirs=ray_samples_uniform.frustums.directions.reshape(1, -1, 3),
                covs=torch.diagonal(gaussian_samples.cov, dim1=-2, dim2=-1).view(1, -1, 3),
            )

            for it in [FieldHeadNames.DENSITY, FieldHeadNames.RGB]:
                field_outputs_coarse[it] = field_outputs_coarse[it].reshape((*gaussian_samples.mean.shape[:-1], -1))

            weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])

            # pdf sampling
            ray_samples_pdf = self.sampler_pdf(ray_bundle[obj_msk], ray_samples_uniform, weights_coarse)

            # second pass
            gaussian_samples_fine = ray_samples_pdf.frustums.get_gaussian_blob()

            field_outputs_fine = self.fields(
                gaussian_samples_fine.mean.view(1, -1, 3),
                self.car_latents[int(obj_id)].view(1, -1).to(obj_id.device),
                viewdirs=ray_samples_pdf.frustums.directions.reshape(1, -1, 3),
                covs=torch.diagonal(gaussian_samples_fine.cov, dim1=-2, dim2=-1).view(1, -1, 3),
            )

            for it in [FieldHeadNames.DENSITY, FieldHeadNames.RGB]:
                field_outputs_fine[it] = field_outputs_fine[it].reshape((*gaussian_samples_fine.mean.shape[:-1], -1))
                field_outputs_fine_raw[it][obj_msk] = field_outputs_fine[it]

            weights_coarse_raw[obj_msk] = weights_coarse

        # Redundant operations because the TensorDate cannot set by indices
        ray_samples_uniform_raw = self.sampler_uniform(ray_bundle)
        ray_samples_pdf_raw = self.sampler_pdf(ray_bundle, ray_samples_uniform_raw, weights_coarse_raw)

        # the car nerf is learned in BGR mode
        rgb_fine = field_outputs_fine_raw[FieldHeadNames.RGB][..., [2, 1, 0]]

        field_outputs_fine_raw[FieldHeadNames.RGB] = rgb_fine

        outputs = {
            "ray_samples_list": [ray_samples_uniform_raw, ray_samples_pdf_raw],
            "field_outputs": field_outputs_fine_raw,
        }
        return outputs

    def get_outputs(self, ray_bundle: RayBundle):
        if self.fields is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        obj_ids = ray_bundle.metadata["obj_ids"]
        unique_obj_ids = torch.unique(obj_ids)

        rgb_coarse_raw = torch.zeros((*ray_bundle.shape, 3), device=ray_bundle.camera_indices.device)
        rgb_fine_raw = torch.zeros((*ray_bundle.shape, 3), device=ray_bundle.camera_indices.device)
        accumulation_coarse_raw = torch.zeros((*ray_bundle.shape, 1), device=ray_bundle.camera_indices.device)
        accumulation_fine_raw = torch.zeros((*ray_bundle.shape, 1), device=ray_bundle.camera_indices.device)
        depth_coarse_raw = torch.zeros((*ray_bundle.shape, 1), device=ray_bundle.camera_indices.device)
        depth_fine_raw = torch.zeros((*ray_bundle.shape, 1), device=ray_bundle.camera_indices.device)
        ray_samples_uniform_raw = torch.zeros(
            (*ray_bundle.shape, self.config.num_coarse_samples), device=ray_bundle.camera_indices.device
        )
        ray_samples_pdf_raw = torch.zeros(
            (*ray_bundle.shape, self.config.num_fine_samples), device=ray_bundle.camera_indices.device
        )

        for obj_id in unique_obj_ids:
            obj_msk = (obj_ids == obj_id).squeeze(-1)
            # uniform sampling
            ray_samples_uniform = self.sampler_uniform(ray_bundle[obj_msk])

            # First pass:
            gaussian_samples = ray_samples_uniform.frustums.get_gaussian_blob()

            field_outputs_coarse = self.fields(
                gaussian_samples.mean.view(1, -1, 3),
                self.car_latents[int(obj_id)].view(1, -1),
                viewdirs=ray_samples_uniform.frustums.directions.reshape(1, -1, 3),
                covs=torch.diagonal(gaussian_samples.cov, dim1=-2, dim2=-1).view(1, -1, 3),
            )

            for it in [FieldHeadNames.DENSITY, FieldHeadNames.RGB]:
                field_outputs_coarse[it] = field_outputs_coarse[it].reshape((*gaussian_samples.mean.shape[:-1], -1))

            weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])

            rgb_coarse = self.renderer_rgb(rgb=field_outputs_coarse[FieldHeadNames.RGB], weights=weights_coarse)
            accumulation_coarse = self.renderer_accumulation(weights_coarse)
            depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

            # pdf sampling
            ray_samples_pdf = self.sampler_pdf(ray_bundle[obj_msk], ray_samples_uniform, weights_coarse)

            # second pass
            gaussian_samples_fine = ray_samples_pdf.frustums.get_gaussian_blob()

            field_outputs_fine = self.fields(
                gaussian_samples_fine.mean.view(1, -1, 3),
                self.car_latents[int(obj_id)].view(1, -1),
                viewdirs=ray_samples_pdf.frustums.directions.reshape(1, -1, 3),
                covs=torch.diagonal(gaussian_samples_fine.cov, dim1=-2, dim2=-1).view(1, -1, 3),
            )

            for it in [FieldHeadNames.DENSITY, FieldHeadNames.RGB]:
                field_outputs_fine[it] = field_outputs_fine[it].reshape((*gaussian_samples.mean.shape[:-1], -1))

            weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
            rgb_fine = self.renderer_rgb(rgb=field_outputs_fine[FieldHeadNames.RGB], weights=weights_fine)
            accumulation_fine = self.renderer_accumulation(weights_fine)
            depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

            rgb_coarse_raw[obj_msk] = rgb_coarse
            rgb_fine_raw[obj_msk] = rgb_fine
            accumulation_coarse_raw[obj_msk] = accumulation_coarse
            accumulation_fine_raw[obj_msk] = accumulation_fine
            depth_coarse_raw[obj_msk] = depth_coarse
            depth_fine_raw[obj_msk] = depth_fine
            ray_samples_uniform_raw[obj_msk] = ray_samples_uniform
            ray_samples_pdf_raw[obj_msk] = ray_samples_pdf

        outputs = {
            "rgb_coarse": rgb_coarse_raw,
            "rgb_fine": rgb_fine_raw,
            "accumulation_coarse": accumulation_coarse_raw,
            "accumulation_fine": accumulation_fine_raw,
            "depth_coarse": depth_coarse_raw,
            "depth_fine": depth_fine_raw,
            "ray_samples_list": [ray_samples_uniform_raw, ray_samples_pdf_raw],
        }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        # image = batch["image"].to(self.device)
        # metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        # if self.training:
        #     metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        # image = batch["image"].to(self.device)
        # loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        # if self.training:
        #     loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
        #         outputs["weights_list"], outputs["ray_samples_list"]
        #     )
        #     assert metrics_dict is not None and "distortion" in metrics_dict
        #     loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
        #     if self.config.predict_normals:
        #         # orientation loss for computed normals
        #         loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
        #             outputs["rendered_orientation_loss"]
        #         )

        #         # ground truth supervision for normals
        #         loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
        #             outputs["rendered_pred_normal_loss"]
        #         )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        # image = batch["image"].to(self.device)
        # rgb = outputs["rgb"]
        # acc = colormaps.apply_colormap(outputs["accumulation"])
        # depth = colormaps.apply_depth_colormap(
        #     outputs["depth"],
        #     accumulation=outputs["accumulation"],
        # )

        # combined_rgb = torch.cat([image, rgb], dim=1)
        # combined_acc = torch.cat([acc], dim=1)
        # combined_depth = torch.cat([depth], dim=1)

        # # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        # image = torch.moveaxis(image, -1, 0)[None, ...]
        # rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # psnr = self.psnr(image, rgb)
        # ssim = self.ssim(image, rgb)
        # lpips = self.lpips(image, rgb)

        # # all of these metrics will be logged as scalars
        # metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        # metrics_dict["lpips"] = float(lpips)

        # images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        # for i in range(self.config.num_proposal_iterations):
        #     key = f"prop_depth_{i}"
        #     prop_depth_i = colormaps.apply_depth_colormap(
        #         outputs[key],
        #         accumulation=outputs["accumulation"],
        #     )
        #     images_dict[key] = prop_depth_i

        return {}, {}

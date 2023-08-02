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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from rich.console import Console
from torch.nn.functional import binary_cross_entropy
from torch.nn.parameter import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from mars.model_components.losses import monosdf_depth_loss
from mars.models.nerfacto import NerfactoModel, NerfactoModelConfig
from mars.models.semantic_nerfw import SemanticNerfWModel
from mars.models.sky_model import SkyModelConfig
from mars.utils.neural_scene_graph_helper import box_pts, combine_z, world2object
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import DepthLossType, MSELoss
from nerfstudio.model_components.losses import depth_loss as general_depth_loss
from nerfstudio.model_components.losses import (
    interlevel_loss,
    normalized_depth_scale_and_shift,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

CONSOLE = Console()

# warning: correlating with the dict "_sem2label" in "NSG-Studio/mars/data/mars_dataparser.py"
_type2str = ["Car", None, "Truck"]


@dataclass
class SceneGraphModelConfig(ModelConfig):
    """Neural Scene Graph Model Config"""

    _target: Type = field(default_factory=lambda: SceneGraphModel)
    # background_model: ModelConfig = VanillaModelConfig(_target=NeRFModel)
    background_model: ModelConfig = NerfactoModelConfig()
    object_model_template: ModelConfig = NerfactoModelConfig()

    max_num_obj: int = -1
    ray_add_input_rows: int = -1
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black"] = "black"
    """Whether to randomize the background color."""
    latent_size: int = 256
    """Size of the latent vector representing each of object of a class. 
        If 0 no latent vector is applied and a single representation per object is used."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    object_representation: Literal["class-wise", "object-wise"] = "object-wise"
    """Whether to use a single representation for all objects of a class or a different one for each object."""
    object_ray_sample_strategy: Literal["warmup", "remove-bg", "none"] = "remove-bg"
    object_warmup_steps: int = 1000
    """Number of steps to warm up the object models, before starting to train the background networks in the intersection region."""
    depth_loss_mult: float = 1e-2
    """depth loss multiplier"""
    semantic_loss_mult: float = 1.0
    """semantic loss multiplier"""
    mono_depth_loss_mult: float = 0.0
    """Monocular depth consistency loss multiplier."""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    depth_sigma: float = 0.05
    """Uncertainty around depth values in meters (defaults to 5cm)."""
    should_decay_sigma: bool = False
    """Whether to exponentially decay sigma."""
    starting_depth_sigma: float = 4.0
    """Starting uncertainty around depth values in meters (defaults to 4.0m)."""
    sigma_decay_rate: float = 0.99980
    """Rate of exponential decay."""
    depth_loss_type: DepthLossType = DepthLossType.DS_NERF
    """Depth loss type, conflict with monosdf loss. Default DS-NeRF."""
    use_interlevel_loss: bool = True
    """use the interlevel loss from nerfacto"""
    interlevel_loss_mult: float = 1.0
    """Interlevel loss multipler"""
    debug_object_pose: bool = False
    """render object bounding box in writer"""
    use_sky_model: bool = False
    """whether to use sky model"""
    sky_model: Optional[SkyModelConfig] = SkyModelConfig()
    """sky model config"""


class SceneGraphModel(Model):
    """Scene graph model

    Args:
        config: Scene graph configuration to instantiate model
    """

    config: SceneGraphModelConfig
    object_meta: Dict
    obj_feat_dim: Optional[int]
    car_latents: Optional[Dict]
    use_car_latents: Optional[bool]
    semantics_meta: Optional[Semantics]

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.scale_factor = self.kwargs["scale_factor"]
        self.object_meta = self.kwargs["object_meta"]
        self.obj_feat_dim = self.kwargs["obj_feat_dim"]
        self.use_car_latents = self.kwargs["use_car_latents"]
        self.car_latents = self.kwargs["car_latents"]
        self.car_nerf_state_dict_path = self.kwargs["car_nerf_state_dict_path"]
        self.use_object_latent_code = self.config.object_representation == "class-wise"
        if self.config.object_representation == "class-wise":
            object_model_key = [self.get_object_model_name(key) for key in self.object_meta["obj_class"]]
        else:
            object_model_key = [self.get_object_model_name(key) for key in self.object_meta["scene_obj"]]

        self.use_depth_loss = self.kwargs["use_depth"]

        if self.config.should_decay_sigma:
            self.depth_sigma = torch.tensor([self.config.starting_depth_sigma])
        else:
            self.depth_sigma = torch.tensor([self.config.depth_sigma])

        self.use_semantic = self.kwargs["use_semantic"]
        if self.use_semantic:
            self.semantics_meta = self.kwargs["semantics_meta"]

        self.object_model_key = object_model_key

        aabb_scale = 1
        bg_scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if self.config.use_sky_model is True:
            self.use_sky_model = True
            self.sky_model = self.config.sky_model.setup(
                scene_box=self.scene_box,
                num_train_data=self.num_train_data,
            )
        else:
            self.use_sky_model = False

        self.background_model = self.config.background_model.setup(
            scene_box=bg_scene_box,
            num_train_data=self.num_train_data,
            obj_feat_dim=0,
            object_meta=self.object_meta if self.use_semantic else None,
            use_sky_model=self.use_sky_model,
        )

        # TODO(noted by Tianyu LIU): modify various categories of cars
        # TODO (wuzr): unifing all configurations
        object_models = {
            key: self.config.object_model_template.setup(
                scene_box=self.scene_box,
                num_train_data=self.num_train_data,
                object_meta=self.object_meta,
                obj_feat_dim=self.config.latent_size if self.use_object_latent_code else 0,
                car_latents=self.car_latents,
                car_nerf_state_dict_path=self.car_nerf_state_dict_path,
            )
            for key in object_model_key
        }
        self.object_models = torch.nn.ModuleDict(object_models)

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.renderer_semantics = SemanticRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.depth_loss = general_depth_loss
        self.monosdf_depth_loss = monosdf_depth_loss
        if self.use_semantic:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=self.background_model.semantic_num
            )

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.step = 0

    def get_object_model_name(self, type_id):
        type_id = int(type_id)
        if self.config.object_representation == "class-wise":
            return f"object_class_{type_id}"
        return f"object_{type_id}"

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.use_sky_model:
            param_groups["sky_model"] = self.sky_model.get_param_groups()
        param_groups["background_model"] = self.background_model.get_param_groups()
        obj_param_group = []
        for key in self.object_model_key:
            obj_param_group += self.object_models[key].get_param_groups()
        param_groups[f"object_model"] = obj_param_group
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        for model in [self.background_model] + list(self.object_models.values()):
            callbacks += model.get_training_callbacks(training_callback_attributes)

        if self.use_sky_model:
            callbacks += self.sky_model.get_training_callbacks(training_callback_attributes)

        def steps_callback(step):
            self.step = step

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=100,
                func=steps_callback,
            )
        )
        return callbacks

    def get_background_outputs(self, ray_bundle: RayBundle):
        return_keys = [
            "accumulation",
            "weights_list",
            "depth",
            "ray_samples_list",
            "semantics",
            "semantics_colormap",
        ]
        raw_output = self.background_model(ray_bundle)
        output = {key: value for key, value in raw_output.items() if key in return_keys}
        raw_rgb = raw_output["rgb"]
        if self.use_sky_model:
            sky_rgb = self.sky_model.inference_without_render(ray_bundle)["rgb"]
            rgb = raw_rgb + sky_rgb * (1 - raw_output["accumulation"])
            output["sky_rgb"] = sky_rgb
        else:
            rgb = raw_rgb
        output["rgb"] = rgb

        if self.config.use_interlevel_loss and self.training:
            if isinstance(self.background_model, NerfactoModel) or isinstance(
                self.background_model, SemanticNerfWModel
            ):
                output["interlevel_loss"] = interlevel_loss(raw_output["weights_list"], raw_output["ray_samples_list"])
            else:
                output["interlevel_loss"] = 0.0

        if not self.training:  # potential risk for OOM since we store weight_list for eval here
            if self.config.debug_object_pose:
                output["debug_rgb"] = output["rgb"]
            output["background"] = raw_rgb
            output["background_depth"] = output["depth"]
            output["objects_rgb"] = torch.zeros_like(output["rgb"])
            output["objects_depth"] = torch.zeros_like(output["depth"])
        output["directions_norm"] = ray_bundle.metadata["directions_norm"]
        return output

    def get_outputs(self, ray_bundle: RayBundle):
        N_rays = int(ray_bundle.origins.shape[0])
        rays_o, rays_d = ray_bundle.origins, ray_bundle.directions

        # No object pose is provided, use only the background node
        if "object_rays_info" not in ray_bundle.metadata:
            return self.get_background_outputs(ray_bundle)

        obj_pose = self.batchify_object_pose(ray_bundle).to(self.device)
        # [x, y, z, yaw, track_id, length, width, height, class_id]

        # compute intersections of ray and object bounding box.
        # pts_box_o, viewdirs_box_o, z_vals_in_o, z_vals_out_o --> new rays_o, rays_d, near, far in object frames
        # intersection_map: which ray intersects with which object
        (
            pts_box_w,
            viewdirs_box_w,
            z_vals_in_w,
            z_vals_out_w,
            pts_box_o,
            viewdirs_box_o,
            z_vals_in_o,
            z_vals_out_o,
            intersection_map,
            ray_o_o,
        ) = box_pts(
            [rays_o, rays_d],
            obj_pose[..., :3],
            obj_pose[..., 3],
            dim=obj_pose[..., 5:8],
            one_intersec_per_ray=False,
        )

        # No intersection with object bounding boxes, use only the background node
        if intersection_map is None:
            return self.get_background_outputs(ray_bundle)

        ray_o_o = ray_o_o[intersection_map[:, 0], intersection_map[:, 1]]
        # only keep the intersected object poses
        obj_pose = obj_pose[intersection_map[:, 0], intersection_map[:, 1], :]
        # intersected rays
        insec_idx = torch.zeros_like(rays_o[..., 0], dtype=torch.bool)
        insec_idx[intersection_map[..., 0]] = True

        output_background = self.background_model.inference_without_render(ray_bundle)
        interlevel_bg = (
            interlevel_loss(output_background["weights_list"], output_background["ray_samples_list"])
            if (
                isinstance(self.background_model, NerfactoModel)
                or isinstance(self.background_model, SemanticNerfWModel)
            )
            and self.config.use_interlevel_loss
            else 0.0
        )
        interlevels = [interlevel_bg]

        if not self.training:
            background_weights = output_background["ray_samples_list"][-1].get_weights(
                output_background["field_outputs"][FieldHeadNames.DENSITY]
            )
            background_rgb = self.renderer_rgb(
                rgb=output_background["field_outputs"][FieldHeadNames.RGB], weights=background_weights
            )
            background_depth = self.renderer_depth(
                weights=output_background["weights_list"][-1], ray_samples=output_background["ray_samples_list"][-1]
            )

        if self.config.object_ray_sample_strategy == "remove-bg":
            # make density of background sampling points in truncation region with object bounding boxes to be 0
            bg_samples_z_vals = output_background["ray_samples_list"][-1].frustums.starts  # (n_rays, n_samples)

            bg_samples_z_vals = bg_samples_z_vals[intersection_map[..., 0]].squeeze(-1)  # (n_intersects, n_samples)
            output_bg_density = output_background["field_outputs"][FieldHeadNames.DENSITY][..., 0]
            # (n_rays, n_samples)

            output_bg_insec_density = output_bg_density[intersection_map[..., 0]]  # (n_intersects, n_samples)
            mask = (bg_samples_z_vals > z_vals_in_o.unsqueeze(-1)) & (bg_samples_z_vals < z_vals_out_o.unsqueeze(-1))
            output_bg_insec_density[mask] = 0.0

        track_idx = obj_pose[..., 4]  # (n_intersects, )
        n_intersects = track_idx.size(0)
        n_samples = self.background_model.num_sample_points()

        if self.config.object_representation == "class-wise":
            obj_class = obj_pose[..., 8]  # (n_intersects, )
            # type_list: unique class ids, (n_class, ), temp_class_idx: (n_intersects, )
            # note that temp_class_id != type_id
            type_list, temp_class_idx = torch.unique(obj_class.reshape(-1), return_inverse=True)
            ray_class_id = temp_class_idx.reshape(obj_class.shape)  # (n_intersects,)
        else:
            type_list, temp_class_idx = torch.unique(track_idx.reshape(-1), return_inverse=True)
            ray_class_id = temp_class_idx.reshape(-1)  # (n_intersects,)

        output_obj = []
        z_vals_obj_w = torch.zeros((n_intersects, n_samples)).to(self.device)
        for class_id, type_id in enumerate(type_list):
            if type_id == -1:
                continue
            # mask for rays that intersect with the object of this class, (n_intersects, )
            mask = ray_class_id == class_id
            # ray indices that intersect with the object of this class, (n_class_intersects,)
            typed_ray_ids = intersection_map[mask, 0]
            n_class_intersects = typed_ray_ids.shape[0]
            if n_class_intersects == 0:
                output_obj.append(None)
                continue
            ray_obj = RayBundle(
                origins=ray_o_o[mask],
                directions=viewdirs_box_o[mask],
                pixel_area=ray_bundle.pixel_area[typed_ray_ids],
                camera_indices=ray_bundle.camera_indices[typed_ray_ids],
                nears=z_vals_in_o[mask, None],
                fars=z_vals_out_o[mask, None],
                metadata={
                    "directions_norm": ray_bundle.metadata["directions_norm"][typed_ray_ids],
                    "obj_ids": track_idx[mask].unsqueeze(-1),
                    "obj_position": obj_pose[mask][..., :3],
                },
            )

            model = (self.object_models[self.get_object_model_name(type_id)]).to(self.device)
            result = model.inference_without_render(ray_obj)

            interlevel_obj = (
                interlevel_loss(result["weights_list"], result["ray_samples_list"])
                if isinstance(model, NerfactoModel) and self.config.use_interlevel_loss
                else 0.0
            )
            interlevels.append(interlevel_obj)

            # calculate the z_vals in world frame for each ray
            # sampled point coordinate in the object coordinate (n_class_intersects, n_samples, 3)
            pts_box_samples_o = result["ray_samples_list"][-1].frustums.get_positions()
            obj_pose_transform = torch.reshape(
                obj_pose[mask].unsqueeze(-2).repeat_interleave(n_samples, dim=1), [-1, obj_pose.shape[-1]]
            )
            pts_box_samples_w, _ = world2object(
                torch.reshape(pts_box_samples_o, [-1, 3]),
                None,
                obj_pose_transform[..., :3],
                obj_pose_transform[..., 3],
                dim=obj_pose_transform[..., 5:8] if obj_pose.shape[-1] > 5 else None,
                inverse=True,
            )
            pts_box_samples_w = pts_box_samples_w.reshape([-1, n_samples, 3])
            z_vals_obj_w_i = torch.linalg.norm(pts_box_samples_w - rays_o[typed_ray_ids, :].unsqueeze(-2), dim=-1)
            z_vals_obj_w[mask] += z_vals_obj_w_i

            output_obj.append(result)

        z_vals_bckg = output_background["ray_samples_list"][-1].spacing_starts[..., 0]
        z_vals_bckg = output_background["ray_samples_list"][-1].spacing_to_euclidean_fn(z_vals_bckg)
        z_vals, id_z_vals_bckg, id_z_vals_obj = combine_z(
            z_vals_bckg, z_vals_obj_w, intersection_map, N_rays, n_samples, self.config.max_num_obj, n_samples
        )
        delta = torch.cat([z_vals[:, 1:] - z_vals[:, :-1], torch.ones_like(z_vals[:, :1])], dim=-1)

        if self.config.object_ray_sample_strategy == "warmup" and self.step < self.config.object_warmup_steps:
            output_background["field_outputs"][FieldHeadNames.DENSITY][insec_idx, ...] = 0

        # aggregate
        densities = torch.zeros((z_vals.size(0), z_vals.size(1), 1)).to(z_vals.device)
        rgbs = torch.zeros((densities.size(0), densities.size(1), 3)).to(densities.device)
        densities[id_z_vals_bckg[..., 0], id_z_vals_bckg[..., 1], 0] = output_background["field_outputs"][
            FieldHeadNames.DENSITY
        ][..., 0]
        rgbs[id_z_vals_bckg[..., 0], id_z_vals_bckg[..., 1], :] = output_background["field_outputs"][
            FieldHeadNames.RGB
        ][..., :]
        if self.use_semantic:
            num_semantics = len(self.object_meta["semantics"].classes)
            semantics = torch.zeros((densities.size(0), densities.size(1), num_semantics)).to(densities.device)
            semantics[id_z_vals_bckg[..., 0], id_z_vals_bckg[..., 1], :] = output_background["field_outputs"][
                FieldHeadNames.SEMANTICS
            ][..., :]

        # generate debug figure
        if not self.training and self.config.debug_object_pose:
            debug_density = torch.zeros((z_vals.size(0), z_vals.size(1), 1)).to(z_vals.device)
            debug_rgb = torch.zeros((densities.size(0), densities.size(1), 3)).to(densities.device)
            # debug_density[id_z_vals_bckg[..., 0], id_z_vals_bckg[..., 1], 0] = 0
            # debug_rgb[id_z_vals_bckg[..., 0], id_z_vals_bckg[..., 1], :] = 0

        # put object densities and rgbs into the aggregation tensor
        for class_id, type_id in enumerate(type_list):
            type_id = int(type_id)
            if type_id == -1 or output_obj[class_id] is None:
                continue
            mask = ray_class_id == class_id
            index = id_z_vals_obj[intersection_map[mask, 0], intersection_map[mask, 1], :, :]
            densities[index[..., 0], index[..., 1], 0] = output_obj[class_id]["field_outputs"][FieldHeadNames.DENSITY][
                ..., 0
            ]
            rgbs[index[..., 0], index[..., 1], :] = output_obj[class_id]["field_outputs"][FieldHeadNames.RGB][..., :]
            if self.use_semantic:
                semantics[index[..., 0], index[..., 1], self.background_model.str2semantic[_type2str[type_id]]] = 1.0

            if not self.training and self.config.debug_object_pose:
                debug_density[index[..., 0], index[..., 1], 0] = 1
                debug_rgb[index[..., 0], index[..., 1], 0] = 25 * (class_id + 1) / 255.0
                debug_rgb[index[..., 0], index[..., 1], 1] = 25 * (type_id + 1) / 255.0
                debug_rgb[index[..., 0], index[..., 1], 2] = 25 * (class_id + 1) / 255.0

        frustums = Frustums(
            origins=rays_o[:, None, :].expand([z_vals.size(0), z_vals.size(1), 3]),
            directions=rays_d[:, None, :].expand([z_vals.size(0), z_vals.size(1), 3]),
            starts=z_vals[:, :, None],
            ends=(z_vals + delta)[:, :, None],
            pixel_area=ray_bundle.pixel_area[:, None, :].expand([z_vals.size(0), z_vals.size(1), 1]),
        )

        ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=ray_bundle.camera_indices[:, None, :].expand([z_vals.size(0), z_vals.size(1), 1]),
            deltas=delta[:, :, None],
            spacing_starts=torch.clamp_min((z_vals - ray_bundle.nears) / (ray_bundle.fars - ray_bundle.nears), 0)[
                :, :, None
            ],
            spacing_ends=torch.clamp_min((z_vals + delta - ray_bundle.nears) / (ray_bundle.fars - ray_bundle.nears), 0)[
                :, :, None
            ],
        )

        def calc_weights(deltas, density):
            # compute weight
            delta_density = deltas[..., None] * density
            alphas = 1 - torch.exp(-delta_density)

            transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
            transmittance = torch.cat(
                [torch.zeros((*transmittance.shape[:1], 1, 1), device=density.device), transmittance], dim=-2
            )
            transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

            weights = alphas * transmittance  # [..., "num_samples"]
            weights = torch.nan_to_num(weights)
            return weights

        weights = calc_weights(delta, densities)

        outputs = {}

        raw_rgb = self.renderer_rgb(rgb=rgbs, weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        assert ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata
        if self.use_sky_model:
            sky_rgb = self.sky_model.inference_without_render(ray_bundle)["rgb"]
            raw_rgb = self.renderer_rgb(rgb=rgbs, weights=weights)
            rgb = raw_rgb + sky_rgb * (1 - accumulation)
            outputs["sky_rgb"] = sky_rgb
        else:
            rgb = raw_rgb

        outputs.update(
            {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
                "directions_norm": ray_bundle.metadata["directions_norm"],
            }
        )

        if self.use_semantic:
            outputs["semantics"] = self.renderer_semantics(semantics, weights=weights)

        if self.training:
            outputs["weights_list"] = [weights]
            outputs["ray_samples_list"] = [ray_samples]

        if not self.training and self.config.debug_object_pose:
            debug_weights = calc_weights(delta, debug_density)
            debug_rgb_out = self.renderer_rgb(rgb=debug_rgb, weights=debug_weights)
            outputs["debug_rgb"] = debug_rgb_out

        if not self.training:
            outputs["background"] = background_rgb
            outputs["background_depth"] = background_depth
            densities[id_z_vals_bckg[..., 0], id_z_vals_bckg[..., 1], 0] = 0
            rgbs[id_z_vals_bckg[..., 0], id_z_vals_bckg[..., 1], :] = 0
            new_weights = calc_weights(delta, densities)
            outputs["objects_rgb"] = self.renderer_rgb(rgb=rgbs, weights=new_weights)
            outputs["objects_depth"] = self.renderer_depth(weights=new_weights, ray_samples=ray_samples)

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        if self.training and self.config.use_interlevel_loss:
            outputs["interlevel_loss"] = sum(interlevels)

        if ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata:
            outputs["directions_norm"] = ray_bundle.metadata["directions_norm"]

        return outputs

    def batchify_object_pose(self, ray_bundle):
        N_rays = int(ray_bundle.origins.shape[0])
        batch_obj_rays = ray_bundle.metadata["object_rays_info"].reshape(
            N_rays, int(ray_bundle.metadata["object_rays_info"].shape[1] / 3), 3
        )
        # n_rays * n_obj * 6: [x,y,z,yaw,obj_id, 0]
        batch_obj_dyn = batch_obj_rays.view(N_rays, self.config.max_num_obj, self.config.ray_add_input_rows * 3)
        batch_obj = batch_obj_dyn[..., :4]  # n_rays * n_obj * 4: [x,y,z,yaw]
        obj_idx = batch_obj_dyn[..., 4].type(torch.int64)
        # TODO: To run cicai_render.py, add to(self.device) in the following line
        obj_meta_tensor = self.object_meta["obj_metadata"]
        batch_obj_metadata = torch.index_select(obj_meta_tensor, 0, obj_idx.reshape(-1)).reshape(
            -1, obj_idx.shape[1], obj_meta_tensor.shape[1]
        )  # n_rays * n_obj * 5: [track_id, x,y,z, class_id(type)]
        batch_track_id = batch_obj_metadata[..., 0]
        batch_obj = torch.cat([batch_obj, batch_track_id.unsqueeze(-1)], dim=-1)
        batch_dim = batch_obj_metadata[..., 1:4]
        batch_label = batch_obj_metadata[..., 4].unsqueeze(-1)
        batch_obj = torch.cat([batch_obj, batch_dim, batch_label], dim=-1)
        return batch_obj

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        # metrics_dict["depth_mse"] = torch.mean((outputs["depth"] - batch["depth_image"].to(self.device)) ** 2)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        if self.config.use_interlevel_loss and self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * outputs["interlevel_loss"]

        # semantic loss
        if self.use_semantic:
            semantics_gt = batch["semantics"][..., 0]

            semantics_label = [
                self.background_model.color2label.get(tuple(pixel.tolist()), self.background_model.semantic_num)
                for pixel in semantics_gt
            ]
            semantics_label = torch.tensor(semantics_label, device=semantics_gt.device)

            loss_dict["semantics_loss"] = self.config.semantic_loss_mult * self.cross_entropy_loss(
                outputs["semantics"], semantics_label
            )

            if self.use_sky_model:
                occ = outputs["accumulation"].clamp(1e-6, 1 - 1e-6)[:, 0]
                loss_dict["entropy_loss"] = -(occ * torch.log(occ) + (1 - occ) * torch.log(1 - occ)).mean()
                sky_idx = self.semantics_meta.classes.index("Sky")
                sky_mask = semantics_label == sky_idx
                loss_dict["sky_mask_loss"] = binary_cross_entropy(occ, 1 - sky_mask.float()) + occ[sky_mask].mean()
                loss_dict["sky_color_loss"] = self.rgb_loss(outputs["sky_rgb"][sky_mask], image[sky_mask])

        if self.training and self.use_depth_loss:
            assert "depth_image" in batch
            assert "depth_mask" in batch
            depth_gt = batch["depth_image"].to(self.device).float()
            depth_mask = batch["depth_mask"].to(self.device)
            if not self.config.is_euclidean_depth:
                depth_gt = depth_gt * outputs["directions_norm"]
            depth_gt[~depth_mask] = 0.0  # to make it compatible with the automask of the depth loss
            predicted_depth = outputs["depth"].float()
            depth_loss = 0
            sigma = self._get_sigma().to(self.device)

            if self.config.depth_loss_mult > 1e-8:
                for i in range(len(outputs["weights_list"])):
                    depth_loss += self.depth_loss(
                        weights=outputs["weights_list"][i],
                        ray_samples=outputs["ray_samples_list"][i],
                        termination_depth=depth_gt,
                        predicted_depth=predicted_depth,
                        sigma=sigma,
                        directions_norm=outputs["directions_norm"],
                        is_euclidean=self.config.is_euclidean_depth,
                        depth_loss_type=self.config.depth_loss_type,
                    ) / len(outputs["weights_list"])

            mono_depth_loss = monosdf_depth_loss(
                termination_depth=depth_gt,
                predicted_depth=predicted_depth,
                is_euclidean=self.config.is_euclidean_depth,
                directions_norm=outputs["directions_norm"],
            )
            loss_dict["depth_loss"] = (
                self.config.depth_loss_mult * depth_loss + self.config.mono_depth_loss_mult * mono_depth_loss
            )

        if self.training:
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        if self.config.debug_object_pose:
            debug_rgb = outputs["debug_rgb"]
            combined_debug_rgb = torch.cat([debug_rgb], dim=1)
        objects_rgb = outputs["objects_rgb"]
        background_rgb = outputs["background"]

        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = outputs["depth"]

        objects_depth = colormaps.apply_depth_colormap(
            outputs["objects_depth"],
        )

        combined_rgb = torch.cat([image, rgb], dim=0)
        combined_acc = torch.cat([acc], dim=0)
        combined_background_rgb = torch.cat([background_rgb], dim=-1)
        if self.use_depth_loss:
            # align to predicted depth and normalize
            background_depth = outputs["background_depth"]
            depth_gt = batch["depth_image"].to(depth)
            depth_mask = batch["depth_mask"].to(self.device)
            if not self.config.is_euclidean_depth:
                depth_gt = depth_gt * outputs["directions_norm"]
            depth_gt[~depth_mask] = 0.0  # to make it compatible with the automask of the depth loss
            depth[~depth_mask] = 0.0
            max_depth = depth_gt.max()

            if self.config.mono_depth_loss_mult > 1e-8:
                # align to predicted depth and normalize
                scale, shift = normalized_depth_scale_and_shift(
                    depth[None, ...], depth_gt[None, ...], depth_gt[None, ...] > 0.0
                )
                depth = depth * scale + shift

            depth[depth > max_depth] = max_depth  # depth always has some very far value
            background_depth[~depth_mask] = 0.0
            background_depth[background_depth > max_depth] = max_depth
            combined_depth = torch.cat([depth_gt, depth], dim=0)
            combined_depth = colormaps.apply_depth_colormap(combined_depth)
            combined_background_depth = torch.cat([background_depth], dim=0)
            combined_background_depth = colormaps.apply_depth_colormap(combined_background_depth)

        # semantics
        if self.use_semantic:
            semantic_gt = batch["semantics"].to(self.device)[..., 0]
            semantic_labels = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
            semantic_colormap = self.background_model.colormap.to(self.device)[semantic_labels]
            combined_semantic_colormap = torch.cat([semantic_gt, semantic_colormap], dim=0) / 255.0

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...].clamp(0.0, 1.0)

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "background": combined_background_rgb,
            "objects_rgb": objects_rgb,
            "objects_depth": objects_depth,
        }

        if self.use_depth_loss:
            images_dict["bacground_depth"] = combined_background_depth
            images_dict["depth"] = combined_depth

        if self.config.debug_object_pose:
            images_dict["debug_rgb"] = combined_debug_rgb

        if self.use_semantic:
            images_dict["semantics_colormap"] = combined_semantic_colormap

        if self.use_sky_model:
            sky_rgb = outputs["sky_rgb"]
            combined_sky_rgb = torch.cat([sky_rgb], dim=1)
            images_dict["sky_rgb"] = combined_sky_rgb

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle_render(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def _get_sigma(self):
        if not self.config.should_decay_sigma:
            return self.depth_sigma * self.scale_factor

        self.depth_sigma = torch.maximum(  # pylint: disable=attribute-defined-outside-init
            self.config.sigma_decay_rate * self.depth_sigma, torch.tensor([self.config.depth_sigma])
        )
        return self.depth_sigma * self.scale_factor

"""
Nerual scene graph kitti datamanager.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Literal, Optional, Tuple, Type, Union

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from mars.data.mars_dataset import MarsDataset


@dataclass
class MarsDataManagerConfig(VanillaDataManagerConfig):
    """A semantic datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: MarsDataManager)


class MarsDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing semantic data.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def create_train_dataset(self) -> MarsDataset:
        # self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        return MarsDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            use_depth=self.config.dataparser.use_depth,
            use_semantic=self.config.dataparser.use_semantic,
        )

    def create_eval_dataset(self) -> MarsDataset:

        return MarsDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            use_depth=self.config.dataparser.use_depth,
            use_semantic=self.config.dataparser.use_semantic,
        )


    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        object_rays_info = self.train_dataset.metadata["obj_info"][c, y, x]
        object_rays_info = object_rays_info.reshape(object_rays_info.shape[0], -1)
        ray_bundle.metadata["object_rays_info"] = object_rays_info.detach()
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        c = ray_indices[:, 0]  # camera indices
        y = ray_indices[:, 1]  # row indices
        x = ray_indices[:, 2]  # col indices
        object_rays_info = self.eval_dataset.metadata["obj_info"][c, y, x]
        object_rays_info = object_rays_info.reshape(object_rays_info.shape[0], -1)
        ray_bundle.metadata["object_rays_info"] = object_rays_info.detach()
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            object_rays_info = self.eval_dataset.metadata["obj_info"][image_idx]
            camera_ray_bundle.metadata["object_rays_info"] = object_rays_info.reshape(
                camera_ray_bundle.shape[0], camera_ray_bundle.shape[1], -1
            ).detach()
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")
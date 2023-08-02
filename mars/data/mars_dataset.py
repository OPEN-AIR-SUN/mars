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
""""
nerual scene graph base dataset
"""

import os
from pathlib import Path
from typing import Dict

import imageio
import numpy as np
import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import (
    get_depth_image_from_path,
    get_semantics_and_mask_tensors_from_path,
)


class MarsDataset(InputDataset):
    """Dataset that returns nerual scene graph needed: images, pose, render_pose,
    visible_objects, render_objects, objects_meta, hwf, bbox/object_mask.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.

    Returns:
        imgs: [n_frames, h, w, 3]
        instance_segm: [n_frames, h, w]
        poses: [n_frames, 4, 4]
        frame_id: [n_frames]: [frame, cam, 0]
        render_poses: [n_test_frames, 4, 4]
        hwf: [H, W, focal]
        i_split: [[train_split], [validation_split], [test_split]]
        visible_objects: [n_frames, n_obj, 23]
        object_meta: dictionary with metadata for each object with track_id as key
        render_objects: [n_test_frames, n_obj, 23]
        bboxes: 2D bounding boxes in the images stored for each of n_frames
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        use_depth: bool = False,
        use_semantic: bool = False,
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "depth_filenames" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["depth_filenames"] is not None
        ) or not use_depth
        assert (
            "semantics" in dataparser_outputs.metadata.keys() and dataparser_outputs.metadata["semantics"] is not None
        ) or not use_semantic

        self.use_semantic = use_semantic
        if use_semantic:
            self.semantic_filenames = self.metadata["semantics"].filenames
            self.semantic_meta = self.metadata["semantics"]

        self.use_depth = use_depth
        if use_depth:
            self.depth_filenames = self.metadata["depth_filenames"]
            self.depth_unit_scale_factor = 0.01  # VKITTI provide depth maps in centimeters

    def get_scene_images_tracking(self, l_image_filename):
        imgs = imageio.imread(l_image_filename)
        imgs = torch.from_numpy((np.maximum(np.minimum(np.array(imgs), 255), 0) / 255.0).astype(np.float32))
        return imgs

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        image = self.get_scene_images_tracking(image_filename)
        data = {"image_idx": image_idx}
        data["image"] = image
        metadata = self.get_metadata(data)
        data.update(metadata)

        return data

    def get_metadata(self, data: Dict) -> Dict:
        metadata = {}

        if self.use_depth:
            filepath = self.depth_filenames[data["image_idx"]]
            height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
            width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

            # Scale depth images to meter units and also by scaling applied to cameras
            scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
            depth_image = get_depth_image_from_path(
                filepath=Path(filepath), height=height, width=width, scale_factor=scale_factor
            )  # default interpolation cv2.INTER_NEAREST
            depth_mask = torch.abs(depth_image / scale_factor - 65535) > 1e-6  # maskout no-depth input

            metadata["depth_image"] = depth_image
            metadata["depth_mask"] = depth_mask

        # semantic metadata
        if self.use_semantic:
            filepath = self.semantic_filenames[data["image_idx"]]
            semantics, mask = get_semantics_and_mask_tensors_from_path(
                filepath=filepath, mask_indices=[], scale_factor=1.0
            )

            metadata["semantics"] = semantics

        return metadata

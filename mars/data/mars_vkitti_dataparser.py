"""Data# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
# limitations under the License. parser for nerual scene graph kitti dataset"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, List

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.utils.colors import get_color

CONSOLE = Console()

_sem2label = {"Misc": -1, "Car": 0, "Van": 0, "Truck": 2, "Tram": 3, "Pedestrian": 4}
camera_ls = [2, 3]


def extract_object_information(args, visible_objects, objects_meta):
    """Get object and object network properties for the given sequence

    Args:
        args:
            args.object_setting are experimental settings for object networks inputs, set to 0 for current version
        visible_objects: Objects per frame + Pose and other dynamic properties + tracking ID
        objects_meta: Metadata with additional static object information sorted by tracking ID

    Retruns:
        obj_properties [n_input_frames, n_max_objects, n_object_properties, 0]: Object properties per frame
        add_input_rows: 2
        obj_meta_ls: List of object metadata
        scene_objects: List of objects per frame
        scene_classes: List of object classes per frame
    Notes:
        obj_properties: x,y,z,yaw_angle,track_id, 0
    """
    if args.dataset_type == "vkitti":
        # [n_frames, n_max_obj, xyz+track_id+ismoving+0]
        obj_state = visible_objects[:, :, [7, 8, 9, 2, -1]]

        obj_dir = visible_objects[:, :, 10][..., None]
        # [..., width+height+length]
        # obj_dim = visible_objects[:, :, 4:7]
        sh = obj_state.shape
    elif args.dataset_type == "waymo_od":
        obj_state = visible_objects[:, :, [7, 8, 9, 2, -1]]
        obj_dir = visible_objects[:, :, 10][..., None]
        sh = obj_state.shape
    elif args.dataset_type == "kitti":
        obj_state = visible_objects[:, :, [7, 8, 9, 2, 3]]  # [x,y,z,track_id,class_id]
        obj_dir = visible_objects[:, :, 10][..., None]  # yaw_angle
        sh = obj_state.shape
    else:
        raise Exception("Invalid dataset name.")

    # obj_state: [cam, n_obj, [x,y,z,track_id, class_id]]

    # [n_frames, n_max_obj]
    obj_track_id = obj_state[..., 3][..., None]
    obj_class_id = obj_state[..., 4][..., None]
    # Change track_id to row in list(objects_meta)
    obj_meta_ls = list(objects_meta.values())  # object_id, length, height, width, class_id
    # Add first row for no objects
    obj_meta_ls.insert(0, np.zeros_like(obj_meta_ls[0]))
    obj_meta_ls[0][0] = -1
    # Build array describing the relation between metadata IDs and where its located
    row_to_track_id = np.concatenate(
        [
            np.linspace(0, len(objects_meta.values()), len(objects_meta.values()) + 1)[:, None],
            np.array(obj_meta_ls)[:, 0][:, None],
        ],
        axis=1,
    ).astype(np.int32)
    # [n_frames, n_max_obj]
    track_row = np.zeros_like(obj_track_id)

    scene_objects = []
    scene_classes = list(np.unique(np.array(obj_meta_ls)[..., 4]))
    for i, frame_objects in enumerate(obj_track_id):
        for j, camera_objects in enumerate(frame_objects):
            track_row[i, j] = np.argwhere(row_to_track_id[:, 1] == camera_objects)
            if camera_objects >= 0 and not camera_objects in scene_objects:
                scene_objects.append(camera_objects)
    CONSOLE.print(f"{scene_objects} in this scene")

    obj_properties = np.concatenate([obj_state[..., :3], obj_dir, track_row], axis=2)

    if obj_properties.shape[-1] % 3 > 0:
        if obj_properties.shape[-1] % 3 == 1:
            obj_properties = np.concatenate([obj_properties, np.zeros([sh[0], sh[1], 2])], axis=2).astype(np.float32)
        else:
            obj_properties = np.concatenate([obj_properties, np.zeros([sh[0], sh[1], 1])], axis=2).astype(np.float32)

    add_input_rows = int(obj_properties.shape[-1] / 3)

    obj_meta_ls = [
        (obj * np.array([1.0, args.box_scale, 1.0, args.box_scale, 1.0])).astype(np.float32)
        if obj[4] != 4
        else obj * np.array([1.0, 1.2, 1.0, 1.2, 1.0])
        for obj in obj_meta_ls
    ]  # [n_obj, [track_id, length * box_scale/1.2, height, width * box_scale/1.2, class_id]] 1.2 for humans, box_scale for other objects

    return obj_properties, add_input_rows, obj_meta_ls, scene_objects, scene_classes


def _get_objects_by_frame(object_pose, object_meta, max_obj, n_cam, selected_frames, row_id):
    """

    Args:
        object_pose: dynamic information in world and camera space for each object sorted by frames
        object_meta: metadata descirbing object properties like model, label, color, dimensions
        max_obj: Maximum number of objects in a single frame for the whole scene
        n_cam: Amount of cameras
        selected_frames: [first_frame, last_frame]
        row_id: bool

    Returns:
        visible_objects: all objects in the selected sequence of frames
        max_obj: Maximum number of objects in the selected sequence of frames
    """
    visible_objects = []
    frame_objects = []
    max_in_frames = 0
    const_pad = (0, 0)

    #### DEBUG
    ### TODO: Later specify ignored objects as arg
    ignore_objs = [16.0, 17.0, 18.0, 19.0]  # [12.]

    if row_id:
        const_pad = (-1, -1)

    for cam in range(n_cam):
        for obj_pose in object_pose:
            if obj_pose[2] not in ignore_objs:
                if obj_pose[1] == cam:
                    if selected_frames[0] <= obj_pose[0] <= selected_frames[1]:
                        if frame_objects:
                            if not all(frame_objects[-1][:2] == obj_pose[:2]):
                                max_in_frames = (
                                    len(frame_objects) if max_in_frames < len(frame_objects) else max_in_frames
                                )
                                frame_objects = np.pad(
                                    np.array(frame_objects),
                                    ((0, max_obj - len(frame_objects)), (0, 0)),
                                    "constant",
                                    constant_values=const_pad,
                                )
                                visible_objects.append(frame_objects)
                                frame_objects = []

                        label = object_meta[obj_pose[2]][1:4]
                        obj_pose = np.concatenate((obj_pose, label))

                        frame_objects.append(obj_pose)

    max_in_frames = len(frame_objects) if max_in_frames < len(frame_objects) else max_in_frames
    frame_objects = np.pad(
        np.array(frame_objects), ((0, max_obj - len(frame_objects)), (0, 0)), "constant", constant_values=const_pad
    )
    visible_objects.append(frame_objects)

    if max_in_frames < max_obj:
        max_obj = max_in_frames

    # Remove all non existent objects from meta:
    object_meta_seq = {}
    for track_id in np.unique(np.array(visible_objects)[:, :, 2]):
        if track_id in object_meta:
            object_meta_seq[track_id] = object_meta[track_id]

    return visible_objects, object_meta_seq, max_obj


def _get_scene_objects(basedir):
    """

    Args:
        basedir:

    Returns:
        object pose:
            frame cameraID trackID
            alpha width height length
            world_space_X world_space_Y world_space_Z
            rotation_world_space_x rotation_world_space_z rotation_world_space_y
            camera_space_X camera_space_Y camera_space_Z
            rotation_camera_space_y rotation_camera_space_x rotation_camera_space_z
            is_moving
        vehicles_meta:
            trackID
            onehot encoded Label
            onehot encoded vehicle model
            onehot encoded color
            3D bbox dimension (length, height, width)
        max_obj:
            Maximum number of objects in a single frames
        bboxes_by_frame:
            2D bboxes
    """
    object_pose = get_information(os.path.join(basedir, "pose.txt"))
    print("Loading poses from: " + os.path.join(basedir, "pose.txt"))
    bbox = get_information(os.path.join(basedir, "bbox.txt"))
    print("Loading bbox from: " + os.path.join(basedir, "bbox.txt"))
    info = open(os.path.join(basedir, "info.txt")).read()
    print("Loading info from: " + os.path.join(basedir, "info.txt"))
    info = info.splitlines()[1:]

    # Creates a dictionary which label and model for each track_id
    vehicles_meta = {}

    for i, vehicle in enumerate(info):
        vehicle = vehicle.split()  # Ignores colour for now

        label = np.array([_sem2label[vehicle[1]]])

        track_id = np.array([int(vehicle[0])])

        # width height length
        vehicle_dim = object_pose[np.where(object_pose[:, 2] == track_id), :][0, 0, 4:7]
        # For vkitti2 dimensions are defined: width height length
        # To Match vehicle axis xyz swap to length, height, width
        vehicle_dim = vehicle_dim[[2, 1, 0]]

        # vehicle = np.concatenate((np.concatenate((np.concatenate((track_id, label)), model)), color))
        vehicle = np.concatenate([track_id, vehicle_dim])
        vehicle = np.concatenate([vehicle, label]).astype(np.float32)
        vehicles_meta[int(track_id)] = vehicle

    # Get the maximum number of objects in a single frame to define networks
    # input size for the specific scene if objects are used
    max_obj = 0
    f = 0
    c = 0
    count = 0
    for obj in object_pose[:, :2]:
        count += 1
        if not obj[0] == f or obj[1] == c:
            f = obj[0]
            c = obj[1]
            if count > max_obj:
                max_obj = count
            count = 0

    # Add to object_pose if the object is moving between the current and the next frame
    # TODO: Use if moving information to decide if an Object is static or dynamic across the whole scene!!
    object_pose = np.column_stack((object_pose, bbox[:, -1]))

    # Store 2D bounding boxes of frames
    bboxes_by_frame = []
    last_frame = bbox[-1, 0].astype(np.int32)
    for cam in range(2):
        for i in range(last_frame + 1):
            bbox_at_i = np.squeeze(bbox[np.argwhere(bbox[:, 0] == i), :7])
            bboxes_by_frame.append(bbox_at_i[np.argwhere(bbox_at_i[:, 1] == cam), 3:7])

    return object_pose, vehicles_meta, max_obj, bboxes_by_frame


def _convert_to_float(val):
    try:
        v = float(val)
        return v
    except:
        if val == "True":
            return 1
        elif val == "False":
            return 0
        else:
            ValueError("Is neither float nor boolean: " + val)


def get_information(path):
    f = open(path, "r")
    c = f.read()
    c = c.split("\n", 1)[1]

    return np.array([[_convert_to_float(j) for j in i.split(" ")] for i in c.splitlines()])


def get_semantic_information(path):
    colors = pd.read_csv(path, sep=" ", index_col=False)

    return colors


@dataclass
class MarsVKittiDataParserConfig(DataParserConfig):
    """nerual scene graph dataset parser config"""

    _target: Type = field(default_factory=lambda: MarsVKittiParser)
    """target class to instantiate"""
    data: Path = Path("/data1/vkitti/Scene06/clone")
    """Directory specifying location of data."""
    scale_factor: float = 0.1
    """How much to scale the camera origins by."""
    scene_scale: float = 2.0
    """How much to scale the region of interest by."""
    alpha_color: str = "white"
    """alpha color of background"""
    first_frame: int = 0
    """specifies the beginning of a sequence if not the complete scene is taken as Input"""
    last_frame: int = 237
    """specifies the end of a sequence"""
    use_object_properties: bool = True
    """ use pose and properties of visible objects as an input """
    object_setting: int = 0
    """specify wich properties are used"""
    obj_opaque: bool = True
    """Ray does stop after intersecting with the first object bbox if true"""
    box_scale: float = 1.5
    """Maximum scale for bboxes to include shadows"""
    novel_view: str = "left"
    use_obj: bool = True
    render_only: bool = False
    bckg_only: bool = False
    use_object_properties: bool = True
    near_plane: float = 0.5
    """specifies the distance from the last pose to the near plane"""
    far_plane: float = 150.0
    """specifies the distance from the last pose to the far plane"""
    dataset_type: str = "vkitti"
    obj_only: bool = False
    """Train object models on rays close to the objects only"""
    netchunk: int = 1024 * 64
    """number of pts sent through network in parallel, decrease if running out of memory"""
    chunk: int = 1024 * 32
    """number of rays processed in parallel, decrease if running out of memory"""
    max_input_objects: int = -1
    """Max number of object poses considered by the network, will be set automatically"""
    add_input_rows: int = -1
    """reshape tensor, dont change... will be refactor in the future"""
    use_depth: bool = True
    """whether the training loop contains depth"""
    split_setting: str = "reconstruction"
    use_car_latents: bool = False
    car_object_latents_path: Optional[Path] = Path("pretrain/car_nerf/latent_codes.pt")
    """path of car object latent codes"""
    car_nerf_state_dict_path: Optional[Path] = Path("pretrain/car_nerf/car_nerf.ckpt")
    """path of car nerf state dicts"""
    use_semantic: bool = False
    """whether to use semantic information"""
    semantic_path: Optional[Path] = Path("")
    """path of semantic inputs"""
    semantic_mask_classes: List[str] = field(default_factory=lambda: [])
    """semantic classes that do not generate gradient to the background model"""


@dataclass
class MarsVKittiParser(DataParser):
    """nerual scene graph kitti Dataset"""

    config: MarsVKittiDataParserConfig

    def __init__(self, config: MarsVKittiDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.selected_frames = [config.first_frame, config.last_frame]
        self.novel_view = config.novel_view
        self.use_obj = config.use_obj
        self.use_time = False
        self.remove = -1
        self.max_input_objects = -1
        self.render_only = config.render_only
        self.near = config.near_plane
        self.far = config.far_plane
        self.use_object_properties = config.use_object_properties
        self.bckg_only = config.bckg_only
        self.dataset_type = config.dataset_type
        self.time_stamp = None
        self.obj_only = config.obj_only
        self.use_inst_segm = False
        self.netchunk = config.netchunk
        self.chunk = config.chunk
        self.remove_obj = None
        self.debug_local = False
        self.object_setting = config.object_setting
        self.semantic_path = config.semantic_path
        self.use_semantic = config.use_semantic

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        basedir = str(self.data)
        extrinsic = get_information(os.path.join(basedir, "extrinsic.txt"))
        intrinsic = get_information(os.path.join(basedir, "intrinsic.txt"))
        if self.use_semantic:
            semantics = get_semantic_information(os.path.join(basedir, "colors.txt"))
        object_pose, object_meta, max_objects_per_frame, bboxes = _get_scene_objects(basedir)

        if self.object_setting == 0 or self.object_setting == 1:
            row_id = True
        else:
            row_id = False

        count = []
        imgs = []
        poses = []
        extrinsics = []
        frame_id = []
        imgs_name = []
        instance_name = []
        depth_name = []
        semantic_name = []

        rgb_dir = os.path.join(basedir, "frames/rgb")
        instance_dir = os.path.join(basedir, "frames/instanceSegmentation")
        depth_dir = os.path.join(basedir, "frames/depth")
        n_cam = len(os.listdir(rgb_dir))

        if self.selected_frames == -1:
            self.selected_frames = [0, extrinsic.shape[0] - 1]

        # semantic metadata
        semantic_meta = None
        if self.use_semantic:
            semantics = semantics.loc[~semantics["Category"].isin(self.config.semantic_mask_classes)]
            semantics.loc[len(semantics.index)] = ["Undefined", 0, 0, 0]
            semantic_meta = Semantics(
                filenames=[],
                classes=semantics["Category"].tolist(),
                colors=torch.tensor(semantics.iloc[:, 1:].values),
                mask_classes=self.config.semantic_mask_classes,
            )

        for camera in sorted(next(os.walk(rgb_dir))[1]):
            frame_dir = os.path.join(rgb_dir, camera)
            instance_frame_dir = os.path.join(instance_dir, camera)
            depth_frame_dir = os.path.join(depth_dir, camera)
            if self.use_semantic:
                semantic_frame_dir = os.path.join(self.semantic_path, camera)
            cam = int(camera.split("Camera_")[1])

            # TODO: Check mismatching numbers of poses and Images like in loading script for llf
            for frame in sorted(os.listdir(frame_dir)):
                if frame.endswith(".jpg"):
                    frame_num_str = frame.split("rgb_")[1].split(".jpg")[0]
                    frame_num = int(frame_num_str)

                    if self.selected_frames[0] <= frame_num <= self.selected_frames[1]:
                        fname = os.path.join(frame_dir, frame)
                        # imgs.append(imageio.imread(fname))
                        imgs_name.append(fname)

                        inst_frame = "instancegt_" + frame_num_str + ".png"
                        instance_gt_name = os.path.join(instance_frame_dir, inst_frame)
                        instance_name.append(instance_gt_name)
                        depth_frame = "depth_" + frame_num_str + ".png"
                        depth_gt_name = os.path.join(depth_frame_dir, depth_frame)
                        depth_name.append(depth_gt_name)
                        if self.use_semantic:
                            semantic_frame = "classgt_" + frame_num_str + ".png"
                            semantic_gt_name = os.path.join(semantic_frame_dir, semantic_frame)
                            semantic_name.append(semantic_gt_name)

                        ext = extrinsic[frame_num * n_cam : frame_num * n_cam + n_cam, :][cam][2:]
                        ext = np.reshape(ext, (-1, 4))
                        extrinsics.append(ext)

                        # Get camera pose and location from extrinsics
                        pose = np.zeros([4, 4])
                        pose[3, 3] = 1
                        R = np.transpose(ext[:3, :3])
                        t = -ext[:3, -1]

                        # Camera position described in world coordinates
                        pose[:3, -1] = np.matmul(R, t)
                        # Match OpenGL definition of Z
                        pose[:3, :3] = np.matmul(np.eye(3), np.matmul(np.eye(3), R))
                        # Rotate pi around Z
                        pose[:3, 2] = -pose[:3, 2]
                        pose[:3, 1] = -pose[:3, 1]
                        poses.append(pose)
                        frame_id.append([frame_num, cam, 0])

                        count.append(len(imgs) - 1)

        # imgs = (np.array(imgs) / 255.0).astype(np.float32)
        poses = np.array(poses).astype(np.float32)

        origins = poses[..., :3, 3]
        mean_origin = origins.mean(axis=0)
        poses[..., :3, 3] = origins - mean_origin
        object_pose[..., 7:10] = object_pose[..., 7:10] - mean_origin

        visible_objects, object_meta, max_objects_per_frame = _get_objects_by_frame(
            object_pose, object_meta, max_objects_per_frame, n_cam, self.selected_frames, row_id
        )

        visible_objects = np.array(visible_objects)
        # TODO: Undo for final version, now speed up and overfit on less objects
        visible_objects = visible_objects[:, :max_objects_per_frame, :]

        if visible_objects is not None:
            self.config.max_input_objects = visible_objects.shape[1]
        else:
            self.config.max_input_objects = 0

        # count = np.array(range(len(visible_objects)))
        # i_split = [np.sort(count[:]), count[int(0.8 * len(count)) :], count[int(0.8 * len(count)) :]]
        # i_trin, i_val, i_test = i_split

        counts = np.arange(len(visible_objects)).reshape(2, -1)
        i_test = np.array([(idx + 1) % 4 == 0 for idx in counts[0]])
        i_test = np.concatenate((i_test, i_test))
        if self.config.split_setting == "reconstruction":
            i_train = np.ones(len(visible_objects), dtype=bool)
        elif self.config.split_setting == "nvs-75":
            i_train = ~i_test
        elif self.config.split_setting == "nvs-50":
            desired_length = np.shape(counts)[1]
            pattern = np.array([True, True, False, False])
            repetitions = (desired_length + len(pattern) - 1) // len(
                pattern
            )  # Calculate number of necessary repetitions
            repeated_pattern = np.tile(pattern, repetitions)
            i_train = repeated_pattern[:desired_length]  # Slice to the desired length
            i_train = np.concatenate((i_train, i_train))
        elif self.config.split_setting == "nvs-25":
            i_train = np.array([idx % 4 == 0 for idx in counts[0]])
            i_train = np.concatenate((i_train, i_train))
        else:
            raise ValueError("No such split method")

        counts = counts.reshape(-1)
        i_train = counts[i_train]
        i_test = counts[i_test]

        if visible_objects is not None:
            self.max_input_objects = visible_objects.shape[1]
        else:
            self.max_input_objects = 0

        test_load_image = imageio.imread(imgs_name[0])
        image_height, image_width = test_load_image.shape[:2]
        cx, cy = image_width / 2.0, image_height / 2.0

        # Extract objects positions and labels
        if self.use_object_properties or self.bckg_only:
            obj_nodes, add_input_rows, obj_meta_ls, scene_objects, scene_classes = extract_object_information(
                self.config, visible_objects, object_meta
            )
            # obj_nodes: [n_frames, n_max_objects, [x,y,z,yaw_angle,track_id, 0]]
            n_input_frames = obj_nodes.shape[0]
            obj_nodes[..., :3] = obj_nodes[..., :3] * self.scale_factor
            obj_nodes = np.reshape(obj_nodes, [n_input_frames, self.max_input_objects * add_input_rows, 3])
        obj_meta_tensor = torch.from_numpy(np.array(obj_meta_ls, dtype="float32"))  # TODO
        poses[..., :3, 3] *= self.scale_factor

        obj_meta_tensor[..., 1:4] = obj_meta_tensor[..., 1:4] * self.scale_factor

        self.config.add_input_rows = add_input_rows
        if split == "train":
            indices = i_train
        elif split == "val":
            indices = i_test
            # indices = i_val
        elif split == "test":
            indices = i_test
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        input_size = 0

        obj_nodes_tensor = torch.from_numpy(obj_nodes)
        # obj_nodes_tensor = torch.from_numpy(obj_nodes).cuda()
        obj_nodes_tensor = obj_nodes_tensor[:, :, None, ...].repeat_interleave(image_width, dim=2)
        obj_nodes_tensor = obj_nodes_tensor[:, :, None, ...].repeat_interleave(image_height, dim=2)

        obj_size = self.max_input_objects * add_input_rows
        input_size += obj_size
        # [N, ro+rd+rgb+obj_nodes, H, W, 3]
        # rays_rgb_env = np.concatenate([rays_rgb_env, obj_nodes], 1)

        # [N, H, W, ro+rd+rgb+obj_nodes*max_obj, 3]
        # with obj_nodes [(x+y+z)*max_obj + (obj_id+is_training+0)*max_obj]
        obj_nodes_tensor = obj_nodes_tensor.permute([0, 2, 3, 1, 4]).cpu()
        # obj_nodes = np.stack([obj_nodes[i] for i in i_train], axis=0)  # train images only
        obj_info = torch.cat([obj_nodes_tensor[i : i + 1] for i in indices], dim=0)

        image_filenames = [imgs_name[i] for i in indices]
        instance_filenames = [instance_name[i] for i in indices]
        depth_filenames = [depth_name[i] for i in indices]
        if self.use_semantic:
            semantic_meta.filenames = [semantic_name[i] for i in indices]
        poses = poses[indices]

        if self.config.use_car_latents:
            if not self.config.car_object_latents_path.exists():
                CONSOLE.print("[yello]Error: latents not exist")
                exit()
            car_latents = torch.load(str(self.config.car_object_latents_path))
            track_car_latents = {}
            track_car_latents_mean = {}
            for k, idx in enumerate(car_latents["indices"]):
                if self.selected_frames[0] <= idx["fid"] <= self.selected_frames[1]:
                    if idx["oid"] in track_car_latents.keys():
                        track_car_latents[idx["oid"]] = torch.cat(
                            [track_car_latents[idx["oid"]], car_latents["latents"][k].unsqueeze(-1)], dim=-1
                        )
                    else:
                        track_car_latents[idx["oid"]] = car_latents["latents"][k].unsqueeze(-1)
            for k in track_car_latents.keys():
                track_car_latents_mean[k] = track_car_latents[k][..., -1]

        else:
            car_latents = None

        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        focal_X = focal_Y = intrinsic[0, 2]

        cameras = Cameras(
            camera_to_worlds=torch.from_numpy(poses[:, :3, :4]),
            fx=focal_X,
            fy=focal_Y,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
            height=image_height,
            width=image_width,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            mask_filenames=instance_filenames,
            dataparser_scale=self.scale_factor,
            metadata={
                "depth_filenames": depth_filenames,
                "obj_metadata": obj_meta_tensor if len(obj_meta_tensor) > 0 else None,
                "obj_class": scene_classes if len(scene_classes) > 0 else None,
                "scene_obj": scene_objects if len(scene_objects) > 0 else None,
                "obj_info": obj_info if len(obj_info) > 0 else None,
                "scale_factor": self.scale_factor,
                "semantics": semantic_meta,
            },
        )

        if self.config.use_car_latents:
            dataparser_outputs.metadata.update(
                {
                    "car_latents": track_car_latents_mean,
                    "car_nerf_state_dict_path": self.config.car_nerf_state_dict_path,
                }
            )

        print("finished data parsing")
        return dataparser_outputs


VKittiParserSpec = DataParserSpecification(config=MarsVKittiDataParserConfig)

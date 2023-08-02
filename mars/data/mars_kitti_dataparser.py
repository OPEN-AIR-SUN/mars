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
import torch
from cv2 import sort
from rich.console import Console
import pandas as pd

from nerfstudio.cameras import camera_utils
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
from nerfstudio.utils.io import load_from_json
from mars.utils.neural_scene_graph_helper import box_pts

CONSOLE = Console(width=120)
_sem2label = {"Misc": -1, "Car": 0, "Van": 1, "Truck": 2, "Tram": 3, "Pedestrian": 4}
camera_ls = [2, 3]


def kitti_string_to_float(str):
    return float(str.split("e")[0]) * 10 ** int(str.split("e")[1])


def get_rotation(roll, pitch, heading):
    s_heading = np.sin(heading)
    c_heading = np.cos(heading)
    rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])

    s_pitch = np.sin(pitch)
    c_pitch = np.cos(pitch)
    rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

    s_roll = np.sin(roll)
    c_roll = np.cos(roll)
    rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

    rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))

    return rot


def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0.0, 0.0, 0.0, 1.0]])])


def calib_from_txt(calibration_path):
    """
    Read the calibration files and extract the required transformation matrices and focal length.

    Args:
        calibration_path (str): The path to the directory containing the calibration files.

    Returns:
        tuple: A tuple containing the following elements:
            traimu2v (np.array): 4x4 transformation matrix from IMU to Velodyne coordinates.
            v2c (np.array): 4x4 transformation matrix from Velodyne to left camera coordinates.
            c2leftRGB (np.array): 4x4 transformation matrix from left camera to rectified left camera coordinates.
            c2rightRGB (np.array): 4x4 transformation matrix from right camera to rectified right camera coordinates.
            focal (float): Focal length of the left camera.
    """
    c2c = []

    # Read and parse the camera-to-camera calibration file
    f = open(os.path.join(calibration_path, "calib_cam_to_cam.txt"), "r")
    cam_to_cam_str = f.read()
    [left_cam, right_cam] = cam_to_cam_str.split("S_02: ")[1].split("S_03: ")
    cam_to_cam_ls = [left_cam, right_cam]

    # Extract the transformation matrices for left and right cameras
    for i, cam_str in enumerate(cam_to_cam_ls):
        r_str, t_str = cam_str.split("R_0" + str(i + 2) + ": ")[1].split("\nT_0" + str(i + 2) + ": ")
        t_str = t_str.split("\n")[0]
        R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
        R = np.reshape(R, [3, 3])
        t = np.array([kitti_string_to_float(t) for t in t_str.split(" ")])
        Tr = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

        t_str_rect, s_rect_part = cam_str.split("\nT_0" + str(i + 2) + ": ")[1].split("\nS_rect_0" + str(i + 2) + ": ")
        s_rect_str, r_rect_part = s_rect_part.split("\nR_rect_0" + str(i + 2) + ": ")
        r_rect_str = r_rect_part.split("\nP_rect_0" + str(i + 2) + ": ")[0]
        R_rect = np.array([kitti_string_to_float(r) for r in r_rect_str.split(" ")])
        R_rect = np.reshape(R_rect, [3, 3])
        t_rect = np.array([kitti_string_to_float(t) for t in t_str_rect.split(" ")])
        Tr_rect = np.concatenate(
            [np.concatenate([R_rect, t_rect[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]]
        )

        c2c.append(Tr_rect)

    c2leftRGB = c2c[0]
    c2rightRGB = c2c[1]

    # Read and parse the Velodyne-to-camera calibration file
    f = open(os.path.join(calibration_path, "calib_velo_to_cam.txt"), "r")
    velo_to_cam_str = f.read()
    r_str, t_str = velo_to_cam_str.split("R: ")[1].split("\nT: ")
    t_str = t_str.split("\n")[0]
    R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(" ")])
    v2c = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

    # Read and parse the IMU-to-Velodyne calibration file
    f = open(os.path.join(calibration_path, "calib_imu_to_velo.txt"), "r")
    imu_to_velo_str = f.read()
    r_str, t_str = imu_to_velo_str.split("R: ")[1].split("\nT: ")
    R = np.array([kitti_string_to_float(r) for r in r_str.split(" ")])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(" ")])
    imu2v = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0.0, 0.0, 0.0, 1.0])[None, :]])

    # Extract the focal length of the left camera
    focal = kitti_string_to_float(left_cam.split("P_rect_02: ")[1].split()[0])

    return imu2v, v2c, c2leftRGB, c2rightRGB, focal


def tracking_calib_from_txt(calibration_path):
    """
    Extract tracking calibration information from a KITTI tracking calibration file.

    This function reads a KITTI tracking calibration file and extracts the relevant
    calibration information, including projection matrices and transformation matrices
    for camera, LiDAR, and IMU coordinate systems.

    Args:
        calibration_path (str): Path to the KITTI tracking calibration file.

    Returns:
        dict: A dictionary containing the following calibration information:
            P0, P1, P2, P3 (np.array): 3x4 projection matrices for the cameras.
            Tr_cam2camrect (np.array): 4x4 transformation matrix from camera to rectified camera coordinates.
            Tr_velo2cam (np.array): 4x4 transformation matrix from LiDAR to camera coordinates.
            Tr_imu2velo (np.array): 4x4 transformation matrix from IMU to LiDAR coordinates.
    """
    # Read the calibration file
    f = open(calibration_path)
    calib_str = f.read().splitlines()

    # Process the calibration data
    calibs = []
    for calibration in calib_str:
        calibs.append(np.array([kitti_string_to_float(val) for val in calibration.split()[1:]]))

    # Extract the projection matrices
    P0 = np.reshape(calibs[0], [3, 4])
    P1 = np.reshape(calibs[1], [3, 4])
    P2 = np.reshape(calibs[2], [3, 4])
    P3 = np.reshape(calibs[3], [3, 4])

    # Extract the transformation matrix for camera to rectified camera coordinates
    Tr_cam2camrect = np.eye(4)
    R_rect = np.reshape(calibs[4], [3, 3])
    Tr_cam2camrect[:3, :3] = R_rect

    # Extract the transformation matrices for LiDAR to camera and IMU to LiDAR coordinates
    Tr_velo2cam = np.concatenate([np.reshape(calibs[5], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
    Tr_imu2velo = np.concatenate([np.reshape(calibs[6], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    return {
        "P0": P0,
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "Tr_cam2camrect": Tr_cam2camrect,
        "Tr_velo2cam": Tr_velo2cam,
        "Tr_imu2velo": Tr_imu2velo,
    }


def get_poses_calibration(basedir, oxts_path_tracking=None, selected_frames=None):
    """
    Extract poses and calibration information from the KITTI dataset.

    This function processes the OXTS data (GPS/IMU) and extracts the
    pose information (translation and rotation) for each frame. It also
    retrieves the calibration information (transformation matrices and focal length)
    required for further processing.

    Args:
        basedir (str): The base directory containing the KITTI dataset.
        oxts_path_tracking (str, optional): Path to the OXTS data file for tracking sequences.
            If not provided, the function will look for OXTS data in the basedir.
        selected_frames (list, optional): A list of frame indices to process.
            If not provided, all frames in the dataset will be processed.

    Returns:
        tuple: A tuple containing the following elements:
            poses (np.array): An array of 4x4 pose matrices representing the vehicle's
                position and orientation for each frame (IMU pose).
            calibrations (dict): A dictionary containing the transformation matrices
                and focal length obtained from the calibration files.
            focal (float): The focal length of the left camera.
    """

    def oxts_to_pose(oxts):
        """
        OXTS (Oxford Technical Solutions) data typically refers to the data generated by an Inertial and GPS Navigation System (INS/GPS) that is used to provide accurate position, orientation, and velocity information for a moving platform, such as a vehicle. In the context of the KITTI dataset, OXTS data is used to provide the ground truth for the vehicle's trajectory and 6 degrees of freedom (6-DoF) motion, which is essential for evaluating and benchmarking various computer vision and robotics algorithms, such as visual odometry, SLAM, and object detection.

        The OXTS data contains several important measurements:

        1. Latitude, longitude, and altitude: These are the global coordinates of the moving platform.
        2. Roll, pitch, and yaw (heading): These are the orientation angles of the platform, usually given in Euler angles.
        3. Velocity (north, east, and down): These are the linear velocities of the platform in the local navigation frame.
        4. Accelerations (ax, ay, az): These are the linear accelerations in the platform's body frame.
        5. Angular rates (wx, wy, wz): These are the angular rates (also known as angular velocities) of the platform in its body frame.

        In the KITTI dataset, the OXTS data is stored as plain text files with each line corresponding to a timestamp. Each line in the file contains the aforementioned measurements, which are used to compute the ground truth trajectory and 6-DoF motion of the vehicle. This information can be further used for calibration, data synchronization, and performance evaluation of various algorithms.
        """
        poses = []

        def latlon_to_mercator(lat, lon, s):
            """
            Converts latitude and longitude coordinates to Mercator coordinates (x, y) using the given scale factor.

            The Mercator projection is a widely used cylindrical map projection that represents the Earth's surface
            as a flat, rectangular grid, distorting the size of geographical features in higher latitudes.
            This function uses the scale factor 's' to control the amount of distortion in the projection.

            Args:
                lat (float): Latitude in degrees, range: -90 to 90.
                lon (float): Longitude in degrees, range: -180 to 180.
                s (float): Scale factor, typically the cosine of the reference latitude.

            Returns:
                list: A list containing the Mercator coordinates [x, y] in meters.
            """
            r = 6378137.0  # the Earth's equatorial radius in meters
            x = s * r * ((np.pi * lon) / 180)
            y = s * r * np.log(np.tan((np.pi * (90 + lat)) / 360))
            return [x, y]

        # Compute the initial scale and pose based on the selected frames
        if selected_frames is None:
            lat0 = oxts[0][0]
            scale = np.cos(lat0 * np.pi / 180)
            pose_0_inv = None
        else:
            oxts0 = oxts[selected_frames[0][0]]
            lat0 = oxts0[0]
            scale = np.cos(lat0 * np.pi / 180)

            pose_i = np.eye(4)

            [x, y] = latlon_to_mercator(oxts0[0], oxts0[1], scale)
            z = oxts0[2]
            translation = np.array([x, y, z])
            rotation = get_rotation(oxts0[3], oxts0[4], oxts0[5])
            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
            pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

        # Iterate through the OXTS data and compute the corresponding pose matrices
        for oxts_val in oxts:
            pose_i = np.zeros([4, 4])
            pose_i[3, 3] = 1

            [x, y] = latlon_to_mercator(oxts_val[0], oxts_val[1], scale)
            z = oxts_val[2]
            translation = np.array([x, y, z])

            roll = oxts_val[3]
            pitch = oxts_val[4]
            heading = oxts_val[5]
            rotation = get_rotation(roll, pitch, heading)  # (3,3)

            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)  # (4, 4)
            if pose_0_inv is None:
                pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

            pose_i = np.matmul(pose_0_inv, pose_i)
            poses.append(pose_i)

        return np.array(poses)

    # If there is no tracking path specified, use the default path
    if oxts_path_tracking is None:
        oxts_path = os.path.join(basedir, "oxts/data")
        oxts = np.array([np.loadtxt(os.path.join(oxts_path, file)) for file in sorted(os.listdir(oxts_path))])
        calibration_path = os.path.dirname(basedir)

        calibrations = calib_from_txt(calibration_path)

        focal = calibrations[4]

        poses = oxts_to_pose(oxts)

    # If a tracking path is specified, use it to load OXTS data and compute the poses
    else:
        oxts_tracking = np.loadtxt(oxts_path_tracking)
        poses = oxts_to_pose(oxts_tracking)  # (n_frames, 4, 4)
        calibrations = None
        focal = None
        # Set velodyne close to z = 0
        # poses[:, 2, 3] -= 0.8

    # Return the poses, calibrations, and focal length
    return poses, calibrations, focal


def get_camera_poses_tracking(poses_velo_w_tracking, tracking_calibration, selected_frames, scene_no=None):
    exp = False
    camera_poses = []

    opengl2kitti = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    start_frame = selected_frames[0]
    end_frame = selected_frames[1]

    #####################
    # Debug Camera offset
    if scene_no == 2:
        yaw = np.deg2rad(0.7)  ## Affects camera rig roll: High --> counterclockwise
        pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(0.9)  ## Affects camera rig pitch: High -->  up
        # roll = np.deg2rad(1.2)
    elif scene_no == 1:
        if exp:
            yaw = np.deg2rad(0.3)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.6)  ## Affects camera rig yaw: High --> Turn Right
            # pitch = np.deg2rad(-0.97)
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
            # roll = np.deg2rad(1.2)
        else:
            yaw = np.deg2rad(0.5)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
    else:
        yaw = np.deg2rad(0.05)
        pitch = np.deg2rad(-0.75)
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(1.05)
        # roll = np.deg2rad(1.2)

    cam_debug = np.eye(4)
    cam_debug[:3, :3] = get_rotation(roll, pitch, yaw)

    Tr_cam2camrect = tracking_calibration["Tr_cam2camrect"]
    Tr_cam2camrect = np.matmul(Tr_cam2camrect, cam_debug)
    Tr_camrect2cam = invert_transformation(Tr_cam2camrect[:3, :3], Tr_cam2camrect[:3, 3])
    Tr_velo2cam = tracking_calibration["Tr_velo2cam"]
    Tr_cam2velo = invert_transformation(Tr_velo2cam[:3, :3], Tr_velo2cam[:3, 3])

    camera_poses_imu = []
    for cam in camera_ls:
        Tr_camrect2cam_i = tracking_calibration["Tr_camrect2cam0" + str(cam)]
        Tr_cam_i2camrect = invert_transformation(Tr_camrect2cam_i[:3, :3], Tr_camrect2cam_i[:3, 3])
        # transform camera axis from kitti to opengl for nerf:
        cam_i_camrect = np.matmul(Tr_cam_i2camrect, opengl2kitti)
        cam_i_cam0 = np.matmul(Tr_camrect2cam, cam_i_camrect)
        cam_i_velo = np.matmul(Tr_cam2velo, cam_i_cam0)

        cam_i_w = np.matmul(poses_velo_w_tracking, cam_i_velo)
        camera_poses_imu.append(cam_i_w)

    for i, cam in enumerate(camera_ls):
        for frame_no in range(start_frame, end_frame + 1):
            camera_poses.append(camera_poses_imu[i][frame_no])

    return np.array(camera_poses)


def get_obj_pose_tracking(tracklet_path, poses_imu_tracking, calibrations, selected_frames, transform_matrix):
    """
    Extracts object pose information from the KITTI motion tracking dataset for the specified frames.

    Parameters
    ----------
    tracklet_path : str
        Path to the text file containing tracklet information.  A tracklet is a small sequence of object positions and orientations over time, often used in the context of object tracking and motion estimation in computer vision. In a dataset, a tracklet usually represents a single object's pose information across multiple consecutive frames. This information includes the object's position, orientation (usually as rotation around the vertical axis, i.e., yaw angle), and other attributes like object type, dimensions, etc.  In the KITTI dataset, tracklets are used to store and provide ground truth information about dynamic objects in the scene, such as cars, pedestrians, and cyclists.

    poses_imu_tracking : list of numpy arrays
        A list of 4x4 transformation matrices representing the poses (positions and orientations) of the ego vehicle A(the main vehicle equipped with sensors) in Inertial Measurement Unit (IMU) coordinates at different time instances (frames). Each matrix in the list corresponds to a single frame in the dataset.

    calibrations : dict
        Dictionary containing calibration information:
            - "Tr_velo2cam": 3x4 transformation matrix from Velodyne coordinates to camera coordinates.
            - "Tr_imu2velo": 3x4 transformation matrix from IMU coordinates to Velodyne coordinates.

    selected_frames : list of int
        List of two integers specifying the start and end frames to process.

    Returns
    -------
    visible_objects : numpy array
        Array of visible objects with dimensions [2*(end_frame-start_frame+1), max_obj_per_frame, 14] which stores information about the visible objects in each frame for both stereo cameras.
        Contains information about frame number, camera number, object ID, object type, dimensions, 3D pose, and moving status. (explained later in Notes)

    objects_meta : dict
        Dictionary containing metadata for objects in the scene, with object IDs as keys and metadata as values.
        Metadata includes object ID(float), length, height, width, and object type(float).

    Notes
    -----
    The visible_objects array contains the following information for each object:
        0: frame number
        1: camera number
        2: object ID
        3: object type
        4: object length
        5: object height
        6: object width
        7: x coordinate of the object in world coordinates
        8: y coordinate of the object in world coordinates
        9: z coordinate of the object in world coordinates
        10: yaw angle of the object in world coordinates
        11: unused
        12: unused
        13: is_moving flag (1.0 for moving objects, -1.0 for non-moving objects)

        The objects_meta dictionary has the following structure:
            key: object ID (integer)
            value: numpy array containing the following information:
                0: object ID (as a float)
                1: object length
                2: object height
                3: object width
                4: object type (as a float)
    """

    # Helper function to generate a rotation matrix around the y-axis
    def roty_matrix(roty):
        c = np.cos(roty)
        s = np.sin(roty)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    # Extract calibration data
    velo2cam = calibrations["Tr_velo2cam"]
    imu2velo = calibrations["Tr_imu2velo"]
    cam2velo = invert_transformation(velo2cam[:3, :3], velo2cam[:3, 3])
    velo2imu = invert_transformation(imu2velo[:3, :3], imu2velo[:3, 3])

    # Initialize dictionaries and lists to store object metadata and tracklets
    objects_meta_kitti = {}
    objects_meta = {}
    tracklets_ls = []

    start_frame = selected_frames[0]
    end_frame = selected_frames[1]

    # Read tracklets from file
    f = open(tracklet_path)
    tracklets_str = f.read().splitlines()

    # Total number of frames in the scene
    n_scene_frames = len(poses_imu_tracking)
    # Initialize an array to count the number of objects in each frame
    n_obj_in_frame = np.zeros(n_scene_frames)

    # Extract metadata for all objects in the scene
    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        if float(tracklet[1]) < 0:
            continue
        id = tracklet[1]
        if tracklet[2] in _sem2label:
            type = _sem2label[tracklet[2]]
            if not int(id) in objects_meta_kitti:
                length = float(tracklet[12])
                height = float(tracklet[10])
                width = float(tracklet[11])
                # length = float(tracklet[12])
                # height = float(tracklet[10])
                # width = float(tracklet[11])
                objects_meta_kitti[int(id)] = np.array([float(id), type, length, height, width])

            """
            The first two elements (frame number and object ID) as float64.
            The object type (converted from the semantic label) as a float.
            The remaining elements of the tracklet (3D position, rotation, and dimensions) as float64.
            """
            tr_array = np.concatenate(
                [np.array(tracklet[:2]).astype(np.float64), np.array([type]), np.array(tracklet[3:]).astype(np.float64)]
            )
            tracklets_ls.append(tr_array)
            n_obj_in_frame[int(tracklet[0])] += 1

    # Convert tracklets to a numpy array
    tracklets_array = np.array(tracklets_ls)

    # Find the maximum number of objects in a frame for the selected frames
    max_obj_per_frame = int(n_obj_in_frame[start_frame : end_frame + 1].max())
    # Initialize an array to store visible objects with dimensions [2*(end_frame-start_frame+1), max_obj_per_frame, 14]
    visible_objects = np.ones([(end_frame - start_frame + 1) * 2, max_obj_per_frame, 14]) * -1.0

    # Iterate through the tracklets and process object data
    for tracklet in tracklets_array:
        frame_no = tracklet[0]
        if start_frame <= frame_no <= end_frame:
            obj_id = tracklet[1]
            frame_id = np.array([frame_no])
            id_int = int(obj_id)
            obj_type = np.array([objects_meta_kitti[id_int][1]])
            dim = objects_meta_kitti[id_int][-3:].astype(np.float32)

            if id_int not in objects_meta:
                objects_meta[id_int] = np.concatenate(
                    [
                        np.array([id_int]).astype(np.float32),
                        objects_meta_kitti[id_int][2:].astype(np.float64),
                        np.array([objects_meta_kitti[id_int][1]]).astype(np.float64),
                    ]
                )

            # Extract object pose data from tracklet
            pose = tracklet[-4:]

            # Initialize a 4x4 identity matrix for object pose in camera coordinates
            obj_pose_c = np.eye(4)
            obj_pose_c[:3, 3] = pose[:3]
            roty = pose[3]
            obj_pose_c[:3, :3] = roty_matrix(roty)

            # Transform object pose from camera coordinates to IMU coordinates
            obj_pose_imu = np.matmul(velo2imu, np.matmul(cam2velo, obj_pose_c))

            # Get the IMU pose for the corresponding frame
            pose_imu_w_frame_i = poses_imu_tracking[int(frame_id)]

            # Calculate the world pose of the object
            pose_obj_w_i = np.matmul(pose_imu_w_frame_i, obj_pose_imu)
            pose_obj_w_i = np.matmul(transform_matrix, pose_obj_w_i)
            # pose_obj_w_i[:, 3] *= scale_factor

            # Calculate the approximate yaw angle of the object in the world frame
            yaw_aprox = -np.arctan2(pose_obj_w_i[1, 0], pose_obj_w_i[0, 0])

            # TODO: Change if necessary
            is_moving = 1.0

            # Create a 7-element array representing the 3D pose of the object
            pose_3d = np.array([pose_obj_w_i[0, 3], pose_obj_w_i[1, 3], pose_obj_w_i[2, 3], yaw_aprox, 0, 0, is_moving])

            # Iterate through the available cameras
            for j, cam in enumerate(camera_ls):
                cam = np.array(cam).astype(np.float32)[None]
                # Create an array representing the object data for this camera view
                obj = np.concatenate([frame_id, cam, np.array([obj_id]), obj_type, dim, pose_3d])
                frame_cam_id = (int(frame_no) - start_frame) + j * (end_frame + 1 - start_frame)
                obj_column = np.argwhere(visible_objects[frame_cam_id, :, 0] < 0).min()
                visible_objects[frame_cam_id, obj_column] = obj

    # # Remove not moving objects
    # print("Removing non moving objects")
    # obj_to_del = []
    # for key, values in objects_meta.items():
    #     all_obj_poses = np.where(visible_objects[:, :, 2] == key)
    #     if len(all_obj_poses[0]) > 0 and values[4] != 4.0:
    #         frame_intervall = all_obj_poses[0][[0, -1]]
    #         y = all_obj_poses[1][[0, -1]]
    #         obj_poses = visible_objects[frame_intervall, y][:, 7:10]
    #         distance = np.linalg.norm(obj_poses[1] - obj_poses[0])
    #         print(distance)
    #         if distance < 0.5 * scale_factor:
    #             print("Removed:", key)
    #             obj_to_del.append(key)
    #             visible_objects[all_obj_poses] = np.ones(14) * -1.0

    # # Remove metadata for the non-moving objects
    # for key in obj_to_del:
    #     del objects_meta[key]

    return visible_objects, objects_meta


def get_scene_images_tracking(
    tracking_path, sequence, selected_frames, use_depth=False, use_semantic=False, semantic_path=None
):
    [start_frame, end_frame] = selected_frames
    # imgs = []
    img_name = []
    depth_name = []
    semantic_name = []

    left_img_path = os.path.join(os.path.join(tracking_path, "image_02"), sequence)
    right_img_path = os.path.join(os.path.join(tracking_path, "image_03"), sequence)

    if use_depth:
        left_depth_path = os.path.join(os.path.join(tracking_path, "completion_02"), sequence)
        right_depth_path = os.path.join(os.path.join(tracking_path, "completion_03"), sequence)

    for frame_dir in [left_img_path, right_img_path]:
        for frame_no in range(len(os.listdir(left_img_path))):
            if start_frame <= frame_no <= end_frame:
                frame = sorted(os.listdir(frame_dir))[frame_no]
                fname = os.path.join(frame_dir, frame)
                # imgs.append(imageio.imread(fname))
                img_name.append(fname)

    if use_depth:
        for frame_dir in [left_depth_path, right_depth_path]:
            for frame_no in range(len(os.listdir(left_depth_path))):
                if start_frame <= frame_no <= end_frame:
                    frame = sorted(os.listdir(frame_dir))[frame_no]
                    fname = os.path.join(frame_dir, frame)
                    depth_name.append(fname)

    if use_semantic:
        frame_dir = os.path.join(semantic_path, "train", sequence)
        for _ in range(2):
            for frame_no in range(len(os.listdir(frame_dir))):
                if start_frame <= frame_no <= end_frame:
                    frame = sorted(os.listdir(frame_dir))[frame_no]
                    fname = os.path.join(frame_dir, frame)
                    semantic_name.append(fname)

    # imgs = (np.maximum(np.minimum(np.array(imgs), 255), 0) / 255.0).astype(np.float32)
    return img_name, depth_name, semantic_name


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    # Numpy Version
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def rotate_yaw(p, yaw):
    """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards

    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle

    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    # p of size [batch_rays, n_obj, samples, xyz]
    if len(p.shape) < 4:
        # p = p[..., tf.newaxis, :]
        p = p.unsqueeze(-2)

    c_y = torch.cos(yaw)
    s_y = torch.sin(yaw)

    if len(c_y.shape) < 3:
        c_y = c_y.unsqueeze(-1)
        s_y = s_y.unsqueeze(-1)

    # c_y = tf.cos(yaw)[..., tf.newaxis]
    # s_y = tf.sin(yaw)[..., tf.newaxis]

    p_x = c_y * p[..., 0] - s_y * p[..., 2]
    p_y = p[..., 1]
    p_z = s_y * p[..., 0] + c_y * p[..., 2]

    # return tf.concat([p_x[..., tf.newaxis], p_y[..., tf.newaxis], p_z[..., tf.newaxis]], axis=-1)
    return torch.cat([p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)], dim=-1)


def scale_frames(p, sc_factor, inverse=False):
    """Scales points given in N_frames in each dimension [xyz] for each frame or rescales for inverse==True

    Args:
        p: Points given in N_frames frames [N_points, N_frames, N_samples, 3]
        sc_factor: Scaling factor for new frame [N_points, N_frames, 3]
        inverse: Inverse scaling if true, bool

    Returns:
        p_scaled: Points given in N_frames rescaled frames [N_points, N_frames, N_samples, 3]
    """
    # Take 150% of bbox to include shadows etc.
    dim = torch.tensor([1.0, 1.0, 1.0]) * sc_factor
    # dim = tf.constant([0.1, 0.1, 0.1]) * sc_factor

    half_dim = dim / 2
    scaling_factor = (1 / (half_dim + 1e-9)).unsqueeze(-2)
    # scaling_factor = (1 / (half_dim + 1e-9))[:, :, tf.newaxis, :]

    if not inverse:
        p_scaled = scaling_factor * p
    else:
        p_scaled = (1.0 / scaling_factor) * p

    return p_scaled


def get_all_ray_3dbox_intersection(rays_rgb, obj_meta_tensor, chunk, local=False, obj_to_remove=-100):
    """get all rays hitting an oject given 3D multi-object-tracking results of a sequence

    Args:
        rays_rgb: All rays
        obj_meta_tensor: Metadata of all objects
        chunk: No. of rays processed at the same time
        local: Limit used memory if processed on a local machine with limited CPU/GPU resources
        obj_to_remove: If object should be removed from the set of rays

    Returns:
        rays_on_obj: Set of all rays hitting at least one object
        rays_to_remove: Set of all rays hitting an object, that should not be trained
    """

    print("Removing object ", obj_to_remove)
    rays_on_obj = np.array([])
    rays_to_remove = np.array([])
    _batch_sz_inter = chunk if not local else 5000  # args.chunk
    _only_intersect_rays_rgb = rays_rgb[0][None]
    _n_rays = rays_rgb.shape[0]
    _n_obj = (rays_rgb.shape[1] - 3) // 2
    _n_bt = np.ceil(_n_rays / _batch_sz_inter).astype(np.int32)

    for i in range(_n_bt):
        _tf_rays_rgb = torch.from_numpy(rays_rgb[i * _batch_sz_inter : (i + 1) * _batch_sz_inter]).to(torch.float32)
        # _tf_rays_rgb = tf.cast(rays_rgb[i * _batch_sz_inter:(i + 1) * _batch_sz_inter], tf.float32)
        _n_bt_i = _tf_rays_rgb.shape[0]
        _rays_bt = [_tf_rays_rgb[:, 0, :], _tf_rays_rgb[:, 1, :]]
        _objs = torch.reshape(_tf_rays_rgb[:, 3:, :], (_n_bt_i, _n_obj, 6))
        # _objs = tf.reshape(_tf_rays_rgb[:, 3:, :], [_n_bt_i, _n_obj, 6])
        _obj_pose = _objs[..., :3]
        _obj_theta = _objs[..., 3]
        _obj_id = _objs[..., 4].to(torch.int64)
        # _obj_id = tf.cast(_objs[..., 4], tf.int32)
        _obj_meta = torch.index_select(obj_meta_tensor, 0, _obj_id.reshape(-1)).reshape(
            -1, _obj_id.shape[1], obj_meta_tensor.shape[1]
        )
        # _obj_meta = tf.gather(obj_meta_tensor, _obj_id, axis=0)
        _obj_track_id = _obj_meta[..., 0].unsqueeze(-1)
        _obj_dim = _obj_meta[..., 1:4]

        box_points_insters = box_pts(_rays_bt, _obj_pose, _obj_theta, _obj_dim, one_intersec_per_ray=False)
        _mask = box_points_insters[8]
        if _mask is not None:
            if rays_on_obj.any():
                rays_on_obj = np.concatenate([rays_on_obj, np.array(i * _batch_sz_inter + (_mask[:, 0]).cpu().numpy())])
            else:
                rays_on_obj = np.array(i * _batch_sz_inter + _mask[:, 0].cpu().numpy())
            if obj_to_remove is not None:
                _hit_id = _obj_track_id[_mask]
                import pdb

                pdb.set_trace()
                # _hit_id = tf.gather_nd(_obj_track_id, _mask)
                # bool_remove = tf.equal(_hit_id, obj_to_remove)
                bool_remove = np.equal(_hit_id, obj_to_remove)
                if any(bool_remove):
                    # _remove_mask = tf.gather_nd(_mask, tf.where(bool_remove))
                    _remove_mask = np.array(_mask[:, 0])[np.where(np.equal(_hit_id, obj_to_remove))[0]]
                    if rays_to_remove.any():
                        rays_to_remove = np.concatenate([rays_to_remove, np.array(i * _batch_sz_inter + _remove_mask)])
                    else:
                        rays_to_remove = np.array(i * _batch_sz_inter + _remove_mask)

    return rays_on_obj, rays_to_remove, box_points_insters


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
                # print(camera_objects, "in this scene")
                scene_objects.append(camera_objects)
    CONSOLE.log(f"{scene_objects} in this scene.")

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


@dataclass
class MarsKittiDataParserConfig(DataParserConfig):
    """nerual scene graph dataset parser config"""

    _target: Type = field(default_factory=lambda: MarsKittiParser)
    """target class to instantiate"""
    data: Path = Path("data/kitti/training/image_02/0006")
    """Directory specifying location of data."""
    scale_factor: float = 1
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    alpha_color: str = "white"
    """alpha color of background"""
    first_frame: int = 5
    """specifies the beginning of a sequence if not the complete scene is taken as Input"""
    last_frame: int = 260
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
    dataset_type: str = "kitti"
    obj_only: bool = False
    """Train object models on rays close to the objects only"""
    netchunk: int = 1024 * 64
    """number of pts sent through network in parallel, decrease if running out of memory"""
    chunk: int = 1024 * 32
    """number of rays processed in parallel, decrease if running out of memory"""
    max_input_objects: int = -1
    """Max number of object poses considered by the network, will be set automatically"""
    add_input_rows: int = -1
    use_car_latents: bool = False
    car_object_latents_path: Optional[Path] = Path("pretrain/car_nerf/latent_codes.pt")
    """path of car object latent codes"""
    car_nerf_state_dict_path: Optional[Path] = Path("pretrain/car_nerf/car_nerf.ckpt")
    """path of car nerf state dicts"""
    use_depth: bool = True
    """whether the training loop contains depth"""
    split_setting: str = "reconstruction"
    use_semantic: bool = False
    """whether to use semantic information"""
    semantic_path: Optional[Path] = Path("")
    """path of semantic inputs"""
    semantic_mask_classes: List[str] = field(default_factory=lambda: [])
    """semantic classes that do not generate gradient to the background model"""


@dataclass
class MarsKittiParser(DataParser):
    """nerual scene graph kitti Dataset"""

    config: MarsKittiDataParserConfig

    def __init__(self, config: MarsKittiDataParserConfig):
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
        self.use_semantic = config.use_semantic
        self.semantic_path = config.semantic_path

    def _generate_dataparser_outputs(self, split="train"):
        visible_objects_ls = []
        objects_meta_ls = []
        semantic_meta = []
        kitti2vkitti = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        basedir = str(self.data)
        scene_id = basedir[-4:]  # check
        kitti_scene_no = int(scene_id)
        tracking_path = basedir[:-13]  # check
        calibration_path = os.path.join(os.path.join(tracking_path, "calib"), scene_id + ".txt")
        oxts_path_tracking = os.path.join(os.path.join(tracking_path, "oxts"), scene_id + ".txt")
        tracklet_path = os.path.join(os.path.join(tracking_path, "label_02"), scene_id + ".txt")

        tracking_calibration = tracking_calib_from_txt(calibration_path)
        focal_X = tracking_calibration["P2"][0, 0]
        focal_Y = tracking_calibration["P2"][1, 1]
        poses_imu_w_tracking, _, _ = get_poses_calibration(basedir, oxts_path_tracking)  # (n_frames, 4, 4) imu pose

        tr_imu2velo = tracking_calibration["Tr_imu2velo"]
        tr_velo2imu = invert_transformation(tr_imu2velo[:3, :3], tr_imu2velo[:3, 3])
        poses_velo_w_tracking = np.matmul(poses_imu_w_tracking, tr_velo2imu)  # (n_frames, 4, 4) velodyne pose
        if self.use_semantic:
            semantics = pd.read_csv(
                "/data22/DISCOVER_summer2023/chenjt2305/datasets/kitti-step/panoptic_maps/mapping/colors/0006.txt",
                sep=" ",
                index_col=False,
            )

        if self.use_semantic:
            semantics = pd.read_csv(
                os.path.join(self.semantic_path, "colors", scene_id + ".txt"),
                sep=" ",
                index_col=False,
            )

        if self.use_semantic:
            semantics = semantics.loc[~semantics["Category"].isin(self.config.semantic_mask_classes)]
            semantic_meta = Semantics(
                filenames=[],
                classes=semantics["Category"].tolist(),
                colors=torch.tensor(semantics.iloc[:, 1:].values),
                mask_classes=self.config.semantic_mask_classes,
            )
        # Get camera Poses   camare id: 02, 03
        for cam_i in range(2):
            transformation = np.eye(4)
            projection = tracking_calibration["P" + str(cam_i + 2)]  # rectified camera coordinate system -> image
            K_inv = np.linalg.inv(projection[:3, :3])
            R_t = projection[:3, 3]

            t_crect2c = np.matmul(K_inv, R_t)
            # t_crect2c = 1./projection[[0, 1, 2],[0, 1, 2]] * projection[:, 3]
            transformation[:3, 3] = t_crect2c
            tracking_calibration["Tr_camrect2cam0" + str(cam_i + 2)] = transformation

        sequ_frames = self.selected_frames

        cam_poses_tracking = get_camera_poses_tracking(
            poses_velo_w_tracking, tracking_calibration, sequ_frames, kitti_scene_no
        )
        # cam_poses_tracking[..., :3, 3] *= self.scale_factor

        # Orients and centers the poses
        oriented = torch.from_numpy(np.array(cam_poses_tracking).astype(np.float32))  # (n_frames, 3, 4)
        oriented, transform_matrix = camera_utils.auto_orient_and_center_poses(
            oriented
        )  # oriented (n_frames, 3, 4), transform_matrix (3, 4)
        row = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        zeros = torch.zeros(oriented.shape[0], 1, 4)
        oriented = torch.cat([oriented, zeros], dim=1)
        oriented[:, -1] = row  # (n_frames, 4, 4)
        transform_matrix = torch.cat([transform_matrix, row[None, :]], dim=0)  # (4, 4)
        cam_poses_tracking = oriented.numpy()
        transform_matrix = transform_matrix.numpy()
        image_filenames, depth_name, semantic_name = get_scene_images_tracking(
            tracking_path, scene_id, sequ_frames, self.config.use_depth, self.use_semantic, self.semantic_path
        )

        # Get Object poses
        visible_objects_, objects_meta_ = get_obj_pose_tracking(
            tracklet_path, poses_imu_w_tracking, tracking_calibration, sequ_frames, transform_matrix
        )

        # # Align Axis with vkitti axis
        poses = np.matmul(kitti2vkitti, cam_poses_tracking).astype(np.float32)
        visible_objects_[:, :, [9]] *= -1
        visible_objects_[:, :, [7, 8, 9]] = visible_objects_[:, :, [7, 9, 8]]

        # oriented = torch.from_numpy(np.array(poses).astype(np.float32))
        # oriented, transform_matrix = camera_utils.auto_orient_and_center_poses(
        #     oriented, method="none", center_poses=True
        # )
        # poses[..., :3, :] = oriented.numpy()
        # visible_objects_[:, :, 7:10] = (
        #     transform_matrix
        #     @ np.concatenate(
        #         [
        #             visible_objects_[:, :, 7:10, None],
        #             np.ones([visible_objects_.shape[0], visible_objects_.shape[1], 1, 1]),
        #         ],
        #         axis=-2,
        #     )
        # )[:, :, :3, 0]

        visible_objects_ls.append(visible_objects_)
        objects_meta_ls.append(objects_meta_)

        objects_meta = objects_meta_ls[0]
        N_obj = np.array([len(seq_objs[0]) for seq_objs in visible_objects_ls]).max()
        for seq_i, visible_objects in enumerate(visible_objects_ls):
            diff = N_obj - len(visible_objects[0])
            if diff > 0:
                fill = np.ones([np.shape(visible_objects)[0], diff, np.shape(visible_objects)[2]]) * -1
                visible_objects = np.concatenate([visible_objects, fill], axis=1)
                visible_objects_ls[seq_i] = visible_objects

            if seq_i != 0:
                objects_meta.update(objects_meta_ls[seq_i])

        visible_objects = np.concatenate(visible_objects_ls)

        if visible_objects is not None:
            self.config.max_input_objects = visible_objects.shape[1]
        else:
            self.config.max_input_objects = 0

        # count = np.array(range(len(visible_objects)))
        # i_split = [np.sort(count[:]), count[int(0.8 * len(count)) :], count[int(0.8 * len(count)) :]]
        # i_train, i_val, i_test = i_split

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

        novel_view = self.novel_view
        shift_frame = None
        n_oneside = int(poses.shape[0] / 2)
        render_poses = poses[:1]
        # Novel view middle between both cameras:
        if novel_view == "mid":
            new_poses_o = ((poses[n_oneside:, :, -1] - poses[:n_oneside, :, -1]) / 2) + poses[:n_oneside, :, -1]
            new_poses = np.concatenate([poses[:n_oneside, :, :-1], new_poses_o[..., None]], axis=2)
            render_poses = new_poses

        elif novel_view == "shift":
            render_poses = np.repeat(np.eye(4)[None], n_oneside, axis=0)
            l_poses = poses[:n_oneside, ...]
            r_poses = poses[n_oneside:, ...]
            render_poses[:, :3, :3] = (l_poses[:, :3, :3] + r_poses[:, :3, :3]) / 2.0
            render_poses[:, :3, 3] = (
                l_poses[:, :3, 3] + (r_poses[:, :3, 3] - l_poses[:, :3, 3]) * np.linspace(0, 1, n_oneside)[:, None]
            )
            if shift_frame is not None:
                visible_objects = np.repeat(visible_objects[shift_frame][None], len(visible_objects), axis=0)

        elif novel_view == "left":
            render_poses = None
            start_i = 0
            # Render at trained left camera pose
            sequ_frames = self.selected_frames
            l_sequ = sequ_frames[1] - sequ_frames[0] + 1
            render_poses = (
                poses[start_i : start_i + l_sequ, ...]
                if render_poses is None
                else np.concatenate([render_poses, poses[start_i : start_i + l_sequ, ...]])
            )
        elif novel_view == "right":
            # Render at trained left camera pose
            render_poses = poses[n_oneside:, ...]

        render_objects = None

        if self.use_obj:
            start_i = 0
            # Render at trained left camera pose
            sequ_frames = self.selected_frames
            l_sequ = sequ_frames[1] - sequ_frames[0] + 1
            render_objects = (
                visible_objects[start_i : start_i + l_sequ, ...]
                if render_objects is None
                else np.concatenate([render_objects, visible_objects[start_i : start_i + l_sequ, ...]])
            )

        if self.use_time:
            time_stamp = np.zeros([len(poses), 3])
            print("TIME ONLY WORKS FOR SINGLE SEQUENCES")
            time_stamp[:, 0] = np.repeat(
                np.linspace(self.selected_frames[0], self.selected_frames[1], len(poses) // 2)[None], 2, axis=0
            ).flatten()
            render_time_stamp = time_stamp
        else:
            time_stamp = None
            render_time_stamp = None

        if visible_objects is not None:
            self.max_input_objects = visible_objects.shape[1]
        else:
            self.max_input_objects = 0

        if self.render_only:
            visible_objects = render_objects

        test_load_image = imageio.imread(image_filenames[0])
        image_height, image_width = test_load_image.shape[:2]
        cx, cy = image_width / 2.0, image_height / 2.0

        # Extract objects positions and labels
        if self.use_object_properties or self.bckg_only:
            obj_nodes, add_input_rows, obj_meta_ls, scene_objects, scene_classes = extract_object_information(
                self.config, visible_objects, objects_meta
            )
            # obj_nodes: [n_frames, n_max_objects, [x,y,z,yaw_angle,track_id, 0]]
            n_input_frames = obj_nodes.shape[0]
            obj_nodes[..., :3] *= self.scale_factor
            obj_nodes = np.reshape(obj_nodes, [n_input_frames, self.max_input_objects * add_input_rows, 3])
        obj_meta_tensor = torch.from_numpy(np.array(obj_meta_ls, dtype="float32"))  # TODO

        obj_meta_tensor[..., 1:4] *= self.scale_factor
        poses[..., :3, 3] *= self.scale_factor

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

        ###############TODO############### Maybe change here, change generate rays
        # print("get rays")
        # # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # # for each pixel in the image. This stack() adds a new dimension.
        # rays = [get_rays_np(image_height, image_width, focal_X, p) for p in poses[:, :3, :4]]
        # rays = np.stack(rays, axis=0)  # [N, 2:ro+rd, H, W, 3]
        # print("done, concats")
        # # [N, ro+rd+rgb, H, W, 3]
        # rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)

        # print("adding object nodes to each ray")
        # rays_rgb_env = rays_rgb
        input_size = 0

        obj_nodes_tensor = torch.from_numpy(obj_nodes)
        # if self.config.fast_loading:
        #     obj_nodes_tensor = obj_nodes_tensor.cuda()
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

        # """
        # obj_info: n_images * image height * image width * (rays_o, rays_d, rgb, add_input_rows * n_max_obj) * 3
        # add_input_rows = 2 for kitti:
        #     the object info is represented as a 6-dim vector (~2*3, add_input_rows=2):
        #     0~2. x, y, z position of the object
        #     3. yaw angle of the object
        #     4. object id: not track id. track_id = obj_meta[object_id][0]
        #     5. 0 (no use, empty digit)
        # """
        # obj_info = torch.from_numpy(
        #     np.reshape(rays_rgb, [image_n, image_height, image_width, 3 + input_size, 3])[:, :, :, 3:, :]
        # )

        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = [depth_name[i] for i in indices] if self.config.use_depth else None
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
                if sequ_frames[0] <= idx["fid"] <= sequ_frames[1]:
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
            mask_filenames=None,
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


KittiParserSpec = DataParserSpecification(config=MarsKittiDataParserConfig)

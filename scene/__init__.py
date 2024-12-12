#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json

import numpy as np
import torch

from utils.sh_utils import RGB2SH
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from simple_knn._C import distCUDA2

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            pcd = scene_info.point_cloud
            mask = self.process_pointcloud(pcd)

            # Basic processing of the pointcloud using 1nn neighbors with maximum distance threshold
            # This should result in sparser pointcloud representation, leaving clusters with densely packed neighbors.
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()[mask]
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())[mask]
            fused_normals = torch.tensor(np.ones_like(np.asarray(pcd.points))).float().cuda()[mask]

            # Next, we perform clustering using persistent homology, where we let the cluster evolve till the
            # pointcloud is resulted in 2 to 1 connected components, each holds a cluster of 3d points.
            # We see how much time it took the last 2 connected components to die. In case it took the last 2 connected components a significant time until they collapsed into a single cluster, we say that they are valid.

            self.gaussians.create_from_pcd(pcd=pcd, spatial_lr_scale=self.cameras_extent)
            self.gaussians.init_RT_seq(self.train_cameras)


    def save(self, iteration):
        # point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(self.model_path, "scene_dense.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def process_pointcloud(self, pts3d, upper_quantile=.7, lower_quantile=0.):
        max_dist = torch.quantile(distCUDA2(pts3d).float(), q=upper_quantile).item()
        min_dist = torch.quantile(distCUDA2(pts3d).float(), q=lower_quantile).item()
        mask = (distCUDA2(pts3d).float() >= min_dist) & (distCUDA2(pts3d).float() <= max_dist)
        return mask.cuda()

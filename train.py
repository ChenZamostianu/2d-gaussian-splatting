# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import wandb
import os
import numpy as np
import torch
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    wandb.login()
    wandb.init(project="Medida-dev", entity='cryptoguys')
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, depthmap = render_pkg["render"], render_pkg["viewspace_points"], \
render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["surf_depth"] 

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            if viewpoint_cam.uid == 1 or viewpoint_cam.uid == 10:
                psnr_val = psnr(image, gt_image).mean().double()
                ssim_val = ssim(image, gt_image).mean().double()
                wandb_logger(image,
                     rend_normal, depthmap, iteration,
                     gaussians.get_xyz.shape[0], loss.item(), psnr_val.item(), ssim_val.item(), viewpoint_cam.uid)
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            # TODO: Saving iteration intervals should be increased - 30_000, 20_000, 10_000 (?)
            if (iteration in saving_iterations):
                extract_dmaps(background, dataset, gaussians, pipe, scene)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def create_intrinsic_matrix(camera) :
    """
    Creates an intrinsic matrix (K) from the Camera() object.

    Args:
        camera: The Camera() object containing FoVx, image_width, and image_height.

    Returns:
        np.ndarray: The intrinsic matrix K of shape (3, 3).
    """
    if not hasattr(camera, 'FoVx') or not hasattr(camera, 'image_width') or not hasattr(camera, 'image_height'):
        raise ValueError("Camera object must have 'FoVx', 'image_width', and 'image_height' attributes.")

    # Extract necessary attributes
    FoVx = camera.FoVx  # Field of view in the x direction
    image_width = camera.image_width
    image_height = camera.image_height

    # Compute the focal length in pixels (f_x and f_y)
    f_x = (image_width / 2) / math.tan(FoVx / 2)
    f_y = f_x * (image_height / image_width)  # Assuming square pixels

    # Principal point (cx, cy)
    c_x = image_width / 2
    c_y = image_height / 2

    # Construct the intrinsic matrix
    K = torch.tensor([
        [f_x, 0,   c_x],
        [0,   f_y, c_y],
        [0,   0,   1]
    ], dtype=torch.float32)

    return K.detach().cpu().numpy().astype(np.float64)

def saveDMAP(data: dict, dmap_path: str):
    assert "depth_map" in data, "depth_map is required"
    assert "image_width" in data and data["image_width"] > 0, "image_width is required"
    assert (
        "image_height" in data and data["image_height"] > 0
    ), "image_height is required"
    assert "depth_width" in data and data["depth_width"] > 0, "depth_width is required"
    assert (
        "depth_height" in data and data["depth_height"] > 0
    ), "depth_height is required"

    assert "depth_min" in data, "depth_min is required"
    assert "depth_max" in data, "depth_max is required"

    assert "file_name" in data, "file_name is required"
    assert "reference_view_id" in data, "reference_view_id is required"
    assert "neighbor_view_ids" in data, "neighbor_view_ids is required"

    assert "K" in data, "K is required"
    assert "R" in data, "R is required"
    assert "C" in data, "C is required"

    content_type = 1
    if "normal_map" in data:
        content_type += 2
    if "confidence_map" in data:
        content_type += 4
    if "views_map" in data:
        content_type += 8

    with open(dmap_path, "wb") as dmap:
        dmap.write("DR".encode())

        dmap.write(np.array([content_type], dtype=np.dtype("B")))
        dmap.write(np.array([0], dtype=np.dtype("B")))

        dmap.write(
            np.array([data["image_width"], data["image_height"]], dtype=np.dtype("I"))
        )
        dmap.write(
            np.array([data["depth_width"], data["depth_height"]], dtype=np.dtype("I"))
        )

        dmap.write(
            np.array([data["depth_min"], data["depth_max"]], dtype=np.dtype("f"))
        )

        file_name = data["file_name"]
        dmap.write(np.array([len(file_name)], dtype=np.dtype("H")))
        dmap.write(file_name.encode())

        view_ids = [data["reference_view_id"]] + data["neighbor_view_ids"]
        dmap.write(np.array([len(view_ids)], dtype=np.dtype("I")))
        dmap.write(np.array(view_ids, dtype=np.dtype("I")))

        K = data["K"]
        R = data["R"]
        C = data["C"]
        dmap.write(K.tobytes())
        dmap.write(R.tobytes())
        dmap.write(C.tobytes())

        depth_map = data["depth_map"]
        dmap.write(depth_map.tobytes())

        if "normal_map" in data:
            normal_map = data["normal_map"]
            dmap.write(normal_map.tobytes())
        if "confidence_map" in data:
            confidence_map = data["confidence_map"]
            dmap.write(confidence_map.tobytes())
        if "views_map" in data:
            views_map = data["views_map"]
            dmap.write(views_map.tobytes())

def cameras_2_dmaps(cameras, path_to_mvs):
    """
    Converts a list of Camera() objects into depth maps and saves them as.dmap files.
    cam is assumed to have the following attributes:
        1) cam.R, cam.C, cam.K, corresponding to Rotation, translation and intrinsic matrix accordingly.
        2) cam.depth_map, which corresponds to the depth of a given viewing point
        NOTE: All of these attributes should be numpy arrays as float64, not tensors!!!
        In addition, make sure that it is not arranged as tensor image format, but standard one (for instance, not [C, H, W] but [H, W, C])
        3) cam.image_width and cam.image_height, representing the width and height of the image

    Args:
        cameras: List of Camera() objects.
        path_to_mvs: Path to the directory where the.dmap files will be saved.

    Returns:
        None
    """
    def normalize_depth(depth_map):
        min_depth = depth_map.min()
        max_depth = depth_map.max()
        range_depth = max_depth - min_depth
        if range_depth == 0:
            return torch.zeros_like(depth_map)
        normalized_depth = (depth_map - min_depth) / range_depth
        return normalized_depth


    for cam in cameras:

        dmap_path = os.path.join(path_to_mvs, f"depth_{cam.uid:04d}.dmap")
        depth_map = normalize_depth(cam.depth_map)

        valid_mask = ~((np.isnan(depth_map)) | (np.isinf(depth_map)))

        # check if np.array depthmap is nan or inf:
        depth_map[np.isnan(depth_map)] = np.inf
        depth_map[np.isinf(depth_map)] = 0


        depth_map_min = depth_map[valid_mask].min()
        depth_map_max = depth_map[valid_mask].max()
        if depth_map.ndim == 3:
            depth_map = depth_map[..., 0].astype(np.float64)

        confidence_map = np.ones_like(depth_map, dtype=np.float64) * 10
        data = {
            "depth_map": depth_map,
            "image_width": cam.true_width,
            "image_height": cam.true_height,
            "depth_width": depth_map.shape[1],
            "depth_height": depth_map.shape[0],
            "depth_min": depth_map_min,
            "depth_max": depth_map_max,
            "file_name": f"{cam.image_full_name}",
            "reference_view_id": cam.uid,
            "neighbor_view_ids": list(range(len(cameras))),
            "K": create_intrinsic_matrix(cam).astype(np.float64),
            "R": cam.R.astype(np.float64),
            "C": cam.C.astype(np.float64),
            # Optional fields (include if available)
            "confidence_map": confidence_map,
            # "normal_map": normal_map,
        }

        saveDMAP(data, dmap_path=dmap_path)
def extract_dmaps(background, dataset, gaussians, pipe, scene):
    import torchvision.transforms as F

    def normalize_depth(depth_map):
        min_depth = depth_map.min()
        max_depth = depth_map.max()
        range_depth = max_depth - min_depth
        if range_depth == 0:
            return torch.zeros_like(depth_map)
        normalized_depth = (depth_map - min_depth) / range_depth
        return normalized_depth


    vp = scene.getTrainCameras().copy()


    # base_mvs = os.path.join(dataset.model_path, f"mvs_{iteration}")
    base_mvs = dataset.model_path
    for cam in vp:

        dmap_path = os.path.join(base_mvs, f"depth_{cam.uid:04d}.dmap")

        rend_pkg = render(cam, gaussians, pipe, background)

        depth_map = rend_pkg['surf_depth']
        depth_map = normalize_depth(depth_map)

        valid_mask = ~((depth_map.isinf()) | (depth_map.isnan()))
        depth_map_min = depth_map[valid_mask].min().item()
        depth_map_max = depth_map[valid_mask].max().item()
        depth_map = depth_map.detach().cpu().permute(1, 2, 0).numpy()
        depth_map = depth_map[..., 0].astype(np.float64)
        normal_map = rend_pkg['rend_normal'].detach().cpu().permute(1, 2, 0).numpy()
        confidence_map = np.ones_like(depth_map, dtype=np.float64) * 10
        print(f"image_width {cam.image_width}")
        print(f"image_hight {cam.image_height}")
        print(f"image_width {cam.true_width}")
        print(f"image_hight {cam.true_height}")
        print(f"depth_width {depth_map.shape[1]}")
        print(f"depth_hight {depth_map.shape[0]}\n\n")

        data = {
            "depth_map": depth_map,
            "image_width": cam.true_width,
            "image_height": cam.true_height,
            "depth_width": depth_map.shape[1],
            "depth_height": depth_map.shape[0],
            "depth_min": depth_map_min,
            "depth_max": depth_map_max,
            "file_name": f"{cam.image_full_name}",
            "reference_view_id": cam.uid,
            "neighbor_view_ids": list(range(len(vp))),
            "K": create_intrinsic_matrix(cam).astype(np.float64),
            "R": cam.R.astype(np.float64),
            "C": cam.T.astype(np.float64),
            # Optional fields (include if available)
            "normal_map": normal_map,
            "confidence_map": confidence_map,  ##
        }

        saveDMAP(data, dmap_path=dmap_path)
def wandb_logger(predicted_image, normal_map, depth_map, iteration, num_patches, loss, psnr_score, ssim_score, uid):

    """
    Log images and metrics to Weights & Biases with side-by-side comparisons and video support.
    """
    # Ensure proper normalization for images
    def normalize_image(img, n=255):
        if img.dtype != np.uint8:
            img = np.clip(img * n, 0, n).astype(np.uint8)
        return img

    def normalize_depth(depth):
        depth = depth.squeeze()
        depth_min = np.min(depth)
        depth_max = np.max(depth)
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)
        return (normalized_depth * 255).astype(np.uint8)

    def normalize_normals(normals):
        return ((normals + 1) * 127.5).astype(np.uint8)

    # Process images
    depth_viz = depth_map.detach().cpu().permute(1, 2, 0).numpy()
    pred_img = normalize_image(predicted_image.detach().cpu().permute(1, 2, 0).numpy())
    normal_viz = normalize_normals(normal_map.detach().cpu().permute(1, 2, 0).numpy())

    log_dict = {
    #     Images panel 1: Reconstructed Image
        f"View_{uid}/Reconstructed": wandb.Image(
            pred_img,
            caption=f"Reconstructed (PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.2f})"
        ),

        # Images panel 2: Depth vs Normal maps
        f"View_{uid}/Depth_Map": wandb.Image(
            depth_viz,
            caption="Depth Map",
            mode="L"
        ),

        f"View_{uid}/Normal_Map": wandb.Image(
            normal_viz,
            caption="Normal Map"
        ),

        # Metrics
        "num_patches": num_patches,
        "Loss": loss,
        "PSNR": psnr_score,
        "SSIM": ssim_score,
        "iteration": iteration,
    }

    wandb.log(log_dict, step=iteration)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10*(i+1) for i in range(2)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000*(i+1) for i in range(30)])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
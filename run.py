import sys
sys.path.append("FLARE")
sys.path.append("FLARE/dust3r")

import os
import torch
from mast3r.model import AsymmetricMASt3R
import matplotlib.pyplot as pl
from dust3r.utils.image import load_images
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from dust3r.utils.geometry import xy_grid
import numpy as np
import cv2
from dust3r.utils.device import to_numpy
import imageio
import matplotlib
from PIL import Image
import argparse
from pathlib import Path
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


def colorize(
    depth: np.ndarray,
    mask: np.ndarray = None,
    normalize: bool = True,
    cmap: str = "Spectral",
) -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.999)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp), 0)
    colored = (colored.clip(0, 1) * 255).astype(np.uint8)[:, :, :3]
    return colored


def pad_to_square(reshaped_image):
    B, C, H, W = reshaped_image.shape
    max_dim = max(H, W)
    pad_height = max_dim - H
    pad_width = max_dim - W
    padding = (
        pad_width // 2,
        pad_width - pad_width // 2,
        pad_height // 2,
        pad_height - pad_height // 2,
    )
    padded_image = F.pad(reshaped_image, padding, mode="constant", value=0)
    return padded_image


def generate_rank_by_dino(reshaped_image, backbone, query_frame_num, image_size=336):
    # Downsample image to image_size x image_size
    # because we found it is unnecessary to use high resolution
    rgbs = pad_to_square(reshaped_image)
    rgbs = F.interpolate(
        reshaped_image,
        (image_size, image_size),
        mode="bilinear",
        align_corners=True,
    )
    rgbs = _resnet_normalize_image(rgbs.cuda())

    # Get the image features (patch level)
    frame_feat = backbone(rgbs, is_training=True)
    frame_feat = frame_feat["x_norm_patchtokens"]
    frame_feat_norm = F.normalize(frame_feat, p=2, dim=1)

    # Compute the similiarty matrix
    frame_feat_norm = frame_feat_norm.permute(1, 0, 2)
    similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
    similarity_matrix = similarity_matrix.mean(dim=0)
    distance_matrix = 100 - similarity_matrix.clone()

    # Ignore self-pairing
    similarity_matrix.fill_diagonal_(-100)

    similarity_sum = similarity_matrix.sum(dim=1)

    # Find the most common frame
    most_common_frame_index = torch.argmax(similarity_sum).item()
    return most_common_frame_index


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
_resnet_mean = torch.tensor(_RESNET_MEAN).view(1, 3, 1, 1).cuda()
_resnet_std = torch.tensor(_RESNET_STD).view(1, 3, 1, 1).cuda()


def _resnet_normalize_image(img: torch.Tensor) -> torch.Tensor:
    return (img - _resnet_mean) / _resnet_std


def calculate_index_mappings(query_index, S, device=None):
    """
    Construct an order that we can switch [query_index] and [0]
    so that the content of query_index would be placed at [0]
    """
    new_order = torch.arange(S)
    new_order[0] = query_index
    new_order[query_index] = 0
    if device is not None:
        new_order = new_order.to(device)
    return new_order


def local_get_reconstructed_scene(
    model: AsymmetricMASt3R,
    backbone,
    inputfiles,
    image_size,
    min_conf_thr,
):
    batch = load_images(inputfiles, size=image_size, verbose=False)
    images = [gt["img"] for gt in batch]
    images = torch.cat(images, dim=0)
    images = images / 2 + 0.5
    index = generate_rank_by_dino(images, backbone, query_frame_num=1)
    sorted_order = calculate_index_mappings(index, len(images), device=device)
    sorted_batch = []
    for i in range(len(batch)):
        sorted_batch.append(batch[sorted_order[i]])
    batch = sorted_batch
    ignore_keys = set(["depthmap", "dataset", "label", "instance", "idx", "rng", "vid"])
    ignore_dtype_keys = set(
        [
            "true_shape",
            "camera_pose",
            "pts3d",
            "fxfycxcy",
            "img_org",
            "camera_intrinsics",
            "depthmap",
            "depth_anything",
            "fxfycxcy_unorm",
        ]
    )
    dtype = torch.bfloat16
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], torch.Tensor):
                view[name] = view[name].to(device, non_blocking=True)
            else:
                view[name] = torch.tensor(view[name]).to(device, non_blocking=True)
            if view[name].dtype == torch.float32 and name not in ignore_dtype_keys:
                view[name] = view[name].to(dtype)
    view1 = batch[:1]
    view2 = batch[1:]
    with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
        pred1, pred2, pred_cameras = model(view1, view2, True, dtype)
    pts3d = pred2["pts3d"]
    conf = pred2["conf"]
    pts3d = pts3d.detach().cpu()
    B, N, H, W, _ = pts3d.shape
    thres = torch.quantile(conf.flatten(2, 3), min_conf_thr, dim=-1)[0]
    masks_conf = conf > thres[None, :, None, None]
    masks_conf = masks_conf.cpu()

    images = [view["img"] for view in view1 + view2]
    shape = (
        torch.stack([view["true_shape"] for view in view1 + view2], dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    images = (
        torch.stack(images, 1).float().permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    )
    images = images / 2 + 0.5
    images = images.reshape(B, N, H, W, 3)
    # estimate focal length
    images = images[0]
    pts3d = pts3d[0]
    masks_conf = masks_conf[0]
    xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(
        posinf=0, neginf=0
    )  # homogeneous (x,y,1)
    pp = torch.tensor((W / 2, H / 2)).to(xy_over_z)
    pixels = xy_grid(W, H, device=xy_over_z.device).view(1, -1, 2) - pp.view(
        -1, 1, 2
    )  # B,HW,2
    u, v = pixels[:1].unbind(dim=-1)
    x, y, z = pts3d[:1].reshape(-1, 3).unbind(dim=-1)
    fx_votes = (u * z) / x
    fy_votes = (v * z) / y
    # assume square pixels, hence same focal for X and Y
    f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
    focal = torch.nanmedian(f_votes, dim=-1).values
    focal = focal.item()
    pts3d = pts3d.numpy()
    # use PNP to estimate camera poses
    pred_poses = []
    for i in range(pts3d.shape[0]):
        shape_input_each = shape[:, i]
        mesh_grid = xy_grid(shape_input_each[0, 1], shape_input_each[0, 0])
        cur_inlier = conf[0, i] > torch.quantile(conf[0, i], 0.6)
        cur_inlier = cur_inlier.detach().cpu().numpy()
        ransac_thres = 0.5
        confidence = 0.9999
        iterationsCount = 10_000
        cur_pts3d = pts3d[i]
        K = np.float32([(focal, 0, W / 2), (0, focal, H / 2), (0, 0, 1)])
        success, r_pose, t_pose, _ = cv2.solvePnPRansac(
            cur_pts3d[cur_inlier].astype(np.float64),
            mesh_grid[cur_inlier].astype(np.float64),
            K,
            None,
            flags=cv2.SOLVEPNP_SQPNP,
            iterationsCount=iterationsCount,
            reprojectionError=1,
            confidence=confidence,
        )
        r_pose = cv2.Rodrigues(r_pose)[0]
        RT = np.r_[np.c_[r_pose, t_pose], [(0, 0, 0, 1)]]
        cam2world = np.linalg.inv(RT)
        pred_poses.append(cam2world)
    pred_poses = np.stack(pred_poses, axis=0)
    pred_poses = torch.tensor(pred_poses)
    # use knn to clean the point cloud
    K = 10
    points = torch.tensor(pts3d.reshape(1, -1, 3)).cuda()
    knn = knn_points(points, points, K=K)
    dists = knn.dists
    mean_dists = dists.mean(dim=-1)
    masks_dist = mean_dists < torch.quantile(mean_dists.reshape(-1), 0.95)
    masks_dist = masks_dist.detach().cpu().numpy()
    masks_conf = (masks_conf > 0) & masks_dist.reshape(-1, H, W)
    masks_conf = masks_conf > 0
    focals = [focal] * len(images)

    return (
        to_numpy(images),
        to_numpy(pts3d),
        to_numpy(masks_conf),
        to_numpy(focals),
        to_numpy(pred_poses),
    )


def get_input_files(input_path: str) -> list[Path]:
    # Check if input path contains two images or folders
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path {input_path} does not exist.")
    elif os.path.isfile(input_path):
        raise ValueError(
            f"Input path {input_path} must be a directory containing images or folders containing images."
        )
    elif os.path.isdir(input_path) and len(os.listdir(input_path)) == 0:
        raise ValueError(f"Input directory {input_path} is empty.")

    inputs = []

    if all(os.path.isfile(os.path.join(input_path, f)) for f in os.listdir(input_path)):
        inputs.append(input_path)
    elif directories := [os.path.join(input_path, d) for d in os.listdir(input_path)]:
        for directory in directories:
            if os.path.isdir(directory) and len(os.listdir(directory)) >= 1:
                inputs.append(directory)

    if len(inputs) == 0:
        raise ValueError(
            f"No valid inputs found in the input path {input_path}. Please provide a directory containing images."
        )
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs", nargs="+", help="Path to the input files or directories."
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to the output directory.", default="out"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint file.",
        default="pretrained/geometry_pose.pth",
    )
    parser.add_argument(
        "--min_conf_thr",
        type=float,
        help="Minimum confidence threshold for point cloud generation.",
        default=0.1,
    )

    args = parser.parse_args()

    inputs = list(
        set(
            [
                x
                for xs in [
                    get_input_files(input_path.rstrip("/"))
                    for input_path in args.inputs
                ]
                for x in xs
            ]
        )
    )
    print(f"Input directories: {inputs}")

    # Set up the output directory.
    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving outputs to {output_dir}.")

    image_size = 512
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    backbone = backbone.eval().cuda()

    pl.ion()
    # for gpu >= Ampere and pytorch >= 1.12
    torch.backends.cuda.matmul.allow_tf32 = True
    batch_size = 1
    inf = float("inf")
    weights_path = args.checkpoint
    ckpt = torch.load(weights_path, map_location=device)
    model = AsymmetricMASt3R(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
        head_type="catmlp+dpt",
        output_mode="pts3d+desc24",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        two_confs=True,
        desc_conf_mode=("exp", 0, inf),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model: nn.Module = model.to(device).eval()

    for input_dir in inputs:
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"Processing input directory: {input_dir}")

        images, pts3d, masks_conf, focals, pred_poses = local_get_reconstructed_scene(
            model, backbone, input_dir, image_size, args.min_conf_thr
        )

        basename = os.path.basename(input_dir)
        print(f"Processing {basename}...")
        # Save the output
        output_path = os.path.join(output_dir, basename)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Save npz file
        npz_file = os.path.join(output_path, f"{basename}.npz")
        np.savez_compressed(
            npz_file,
            images=images,
            pts3d=pts3d,
            masks_conf=masks_conf,
            focals=focals,
            pred_poses=pred_poses,
        )

        depth_dir = os.path.join(output_path, "depth")
        gt_dir = os.path.join(output_path, "gt")
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        H, W = pts3d.shape[1:3]
        for idx, (_pts3d, pose) in enumerate(zip(pts3d, pred_poses)):
            w2c = np.linalg.inv(pose)
            R = w2c[:3, :3]
            T = w2c[:3, -1]
            pts3d_c2w = np.einsum("kl, Nl -> Nk", R, _pts3d.reshape(-1, 3)) + T[None]
            disp = pts3d_c2w[..., -1]
            disp = disp.reshape(H, W)
            disp_vis = colorize(disp)
            imageio.imwrite(os.path.join(depth_dir, f"depth_vis_{idx}.png"), disp_vis)

            img = images[idx]
            if img.max() <= 1.0 and img.min() >= 0.0:
                img = (img * 255).astype(np.uint8)
            elif img.max() <= 1.0 and img.min() < 0.0:
                img = (img + 1.0) * 255 / 2.0
                img = img.astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            image = Image.fromarray(img)
            image.save(os.path.join(gt_dir, f"image_{idx}.png"))

        print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()

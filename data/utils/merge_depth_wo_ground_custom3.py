# import os
# import torch
# import open3d as o3d
# import json
# from imageio.v2 import imread
# import numpy as np
# import cv2
# from tqdm import tqdm
# import argparse

# ####In this version, the axes are corrected for the custom data through front_rect_mat
# ###In v2, special rotation angles are applied. For cam1 is iden, for cam2 and cam3 are specially calculated.
# #### same as v2 but run as independent

# F_cam2 = np.array([
#     [-0.39972326,  0.89958295, -0.17598816],
#     [ 0.17222990, -0.11486587, -0.97833670],
#     [-0.90031004, -0.42137436, -0.10902052],
# ], dtype=float)

# F_cam3 = np.array([
#     [ 0.78392352,  0.36586111,  0.50160698],
#     [-0.49957523,  0.85141409,  0.15974552],
#     [-0.36863058, -0.37581870,  0.85021868],
# ], dtype=float)

# front_rect_mat = np.array([
#                         [1.0,  0.0,  0.0],   # x_cam/right stays right
#                         [0.0,  0.0, 1.0],   # y_cam/down maps to -z_world (up vs down)
#                         [0.0,  -1.0,  0.0],   # z_cam/forward maps to +y_world/back
#                     ], dtype=float)

# def get_opts():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--out', type=str, required=True)
#     parser.add_argument('--total', type=int, default=1500000)
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = get_opts()
#     with open(os.path.join(args.out, 'meta_data.json'), 'r') as rf:
#         meta_data = json.load(rf)

#     ##########################################################################
#     #                        unproject pixels                             #
#     ##########################################################################
    
#     points, colors = [], []
#     sample_per_frame = args.total // len(meta_data["frames"])
#     for frame in tqdm(meta_data["frames"]):
#         intrinsic = np.array(frame["intrinsics"])
#         c2w = np.array(frame["camtoworld"])

#         ############## converting c2w to opengel convention #################
#         R_old = c2w[:3, :3]
#         t_old = c2w[:3, 3]

#         rgb_path = frame["rgb_path"]
#         # infer camera id from path, e.g. "./images/cam_2/000030.png"
#         if "/cam_1/" in rgb_path:
#             cam_id = 1
#         elif "/cam_2/" in rgb_path:
#             cam_id = 2
#         elif "/cam_3/" in rgb_path:
#             cam_id = 3
#         else:
#             cam_id = None  # fallback

#         # rotation: per-camera rule
#         if cam_id == 1:
#             R_new = R_old  # no change
#         elif cam_id == 2:
#             R_new = F_cam2 @ R_old @ F_cam2.T
#         elif cam_id == 3:
#             R_new = F_cam3 @ R_old @ F_cam3.T
#         else:
#             # if some other cam appears, just leave it unchanged
#             R_new = R_old

#         # translation: always left-multiply by front_rect_mat
#         t_new = front_rect_mat @ t_old

#         c2w[:3, :3] = R_new
#         c2w[:3, 3]  = t_new
#         #################################################

#         cx, cy, fx, fy = (
#             intrinsic[0, 2],
#             intrinsic[1, 2],
#             intrinsic[0, 0],
#             intrinsic[1, 1],
#         )
#         H, W = frame["height"], frame["width"]

#         rgb_path = frame["rgb_path"]
#         frame_cam = frame["rgb_path"].split("/")[-2]
#         im = np.array(imread(os.path.join(args.out, rgb_path)))
#         depth_path = os.path.join(
#             args.out,
#             rgb_path.replace("images", "depth")
#             .replace("./", "")
#             .replace(".jpg", ".pt")
#             .replace(".png", ".pt"),
#         )
#         depth = torch.load(depth_path).numpy()

#         x = np.arange(0, depth.shape[1])  # generate pixel coordinates
#         y = np.arange(0, depth.shape[0])
#         xx, yy = np.meshgrid(x, y)
#         pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)

#         # unproject depth to pointcloud
#         x = (pixels[..., 0] - cx) * depth.reshape(-1) / fx
#         y = (pixels[..., 1] - cy) * depth.reshape(-1) / fy
#         z = depth.reshape(-1)
#         local_points = np.stack([x, y, z], axis=1)
#         local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

#         # mask dynamic
#         mask_path = os.path.join(args.out,
#                                 rgb_path.replace('images', 'masks').replace('./', '').replace('.jpg', '.npy').replace('.png', '.npy'))
#         if os.path.exists(mask_path):
#             dynamic_mask = np.load(mask_path).reshape(-1)

#         # non-ground semantics
#         smts_path = os.path.join(
#             args.out,
#             rgb_path.replace("images", "semantics")
#             .replace("./", "")
#             .replace(".jpg", ".npy")
#             .replace(".png", ".npy"),
#         )
#         if os.path.exists(smts_path):
#             smts = np.load(smts_path).reshape(-1)
#             smt_mask = smts > 1
            
#         mask = dynamic_mask & smt_mask
#         # mask = smt_mask
#         local_points = local_points[mask]
#         local_colors = local_colors[mask]

#         # random downsample
#         if local_points.shape[0] < sample_per_frame:
#             continue
#         sample_idx = np.random.choice(
#             np.arange(local_points.shape[0]), sample_per_frame
#         )
#         local_points = local_points[sample_idx]
#         local_colors = local_colors[sample_idx]

#         local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]
#         # local_points_w = (R_new @ local_points.T).T + t_new

#         points.append(local_points_w)
#         colors.append(local_colors)

#     points = np.concatenate(points)
#     colors = np.concatenate(colors)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     o3d.io.write_point_cloud(os.path.join(args.out, "points3d.ply"), pcd)

import os
import json

import numpy as np
import torch
import open3d as o3d
from imageio.v2 import imread
from tqdm import tqdm

"""
Standalone script to unproject depth into a point cloud with
per-camera rotations (F_cam2, F_cam3) and dynamic / non-ground masking.

Run:
    python merge_depth_wo_ground_v2_hardcoded.py
"""

# =========================
# HARD-CODED CONFIG
# =========================
OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v4"
TOTAL_POINTS = 200000   # total points across all frames

# ---------- Per-camera rotation corrections ----------
F_cam2 = np.array([
    [-0.39972326,  0.89958295, -0.17598816],
    [ 0.17222990, -0.11486587, -0.97833670],
    [-0.90031004, -0.42137436, -0.10902052],
], dtype=float)

F_cam3 = np.array([
    [ 0.78392352,  0.36586111,  0.50160698],
    [-0.49957523,  0.85141409,  0.15974552],
    [-0.36863058, -0.37581870,  0.85021868],
], dtype=float)

# Used only for translation re-orientation
front_rect_mat = np.array([
    [1.0,  0.0,  0.0],   # x_cam/right stays right
    [0.0,  0.0,  1.0],   # y_cam/down -> +z_world
    [0.0, -1.0,  0.0],   # z_cam/forward -> -y_world
], dtype=float)


def main():
    meta_path = os.path.join(OUT_DIR, "meta_data.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta_data.json not found at {meta_path}")

    with open(meta_path, "r") as rf:
        meta_data = json.load(rf)

    frames = meta_data.get("frames", [])
    if len(frames) == 0:
        raise RuntimeError("meta_data.json has no 'frames' or it is empty.")

    # how many points per frame to sample
    sample_per_frame = TOTAL_POINTS // len(frames)
    if sample_per_frame <= 0:
        raise ValueError(
            f"TOTAL_POINTS={TOTAL_POINTS} too small for {len(frames)} frames "
            f"(sample_per_frame={sample_per_frame})."
        )

    points, colors = [], []

    for frame in tqdm(frames, desc="Unprojecting frames"):
        intrinsic = np.array(frame["intrinsics"], dtype=float)
        c2w = np.array(frame["camtoworld"], dtype=float)

        # ------------ per-camera rotation / translation ------------
        R_old = c2w[:3, :3]
        t_old = c2w[:3, 3]

        rgb_path = frame["rgb_path"]  # e.g. "./images/cam_2/000030.png"

        if "/cam_1/" in rgb_path:
            cam_id = 1
        elif "/cam_2/" in rgb_path:
            cam_id = 2
        elif "/cam_3/" in rgb_path:
            cam_id = 3
        else:
            cam_id = None  # unknown cam, leave as-is

        if cam_id == 1:
            R_new = R_old
        elif cam_id == 2:
            R_new = F_cam2 @ R_old @ F_cam2.T
        elif cam_id == 3:
            R_new = F_cam3 @ R_old @ F_cam3.T
        else:
            R_new = R_old

        # translation: always re-orient using front_rect_mat
        t_new = front_rect_mat @ t_old

        c2w[:3, :3] = R_new
        c2w[:3, 3] = t_new
        # -----------------------------------------------------------

        cx, cy, fx, fy = (
            intrinsic[0, 2],
            intrinsic[1, 2],
            intrinsic[0, 0],
            intrinsic[1, 1],
        )
        H, W = frame["height"], frame["width"]

        # ---------- load RGB ----------
        img_full_path = os.path.join(OUT_DIR, rgb_path)
        if not os.path.exists(img_full_path):
            raise FileNotFoundError(f"RGB image not found at {img_full_path}")
        im = np.array(imread(img_full_path))

        # ---------- load depth ----------
        depth_path = os.path.join(
            OUT_DIR,
            rgb_path.replace("images", "depth")
            .replace("./", "")
            .replace(".jpg", ".pt")
            .replace(".png", ".pt"),
        )
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth map (.pt) not found at {depth_path}")
        depth = torch.load(depth_path).numpy()  # (H, W)

        # ---------- generate pixels grid ----------
        x_coords = np.arange(0, depth.shape[1])
        y_coords = np.arange(0, depth.shape[0])
        xx, yy = np.meshgrid(x_coords, y_coords)
        pixels = np.vstack((xx.ravel(), yy.ravel())).T  # (N, 2)

        depth_flat = depth.reshape(-1)
        x = (pixels[..., 0] - cx) * depth_flat / fx
        y = (pixels[..., 1] - cy) * depth_flat / fy
        z = depth_flat
        local_points = np.stack([x, y, z], axis=1)  # (N,3)
        local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

        # ---------- build masks safely ----------
        N = local_points.shape[0]
        dynamic_mask = np.ones(N, dtype=bool)
        smt_mask = np.ones(N, dtype=bool)

        # dynamic mask
        mask_path = os.path.join(
            OUT_DIR,
            rgb_path.replace("images", "masks")
            .replace("./", "")
            .replace(".jpg", ".npy")
            .replace(".png", ".npy"),
        )
        if os.path.exists(mask_path):
            dynamic_mask = np.load(mask_path).reshape(-1).astype(bool)
            if dynamic_mask.shape[0] != N:
                raise ValueError(
                    f"dynamic_mask shape {dynamic_mask.shape[0]} != {N} for {mask_path}"
                )

        # non-ground semantics: keep smts > 1
        smts_path = os.path.join(
            OUT_DIR,
            rgb_path.replace("images", "semantics")
            .replace("./", "")
            .replace(".jpg", ".npy")
            .replace(".png", ".npy"),
        )
        if os.path.exists(smts_path):
            smts = np.load(smts_path).reshape(-1)
            if smts.shape[0] != N:
                raise ValueError(
                    f"semantics shape {smts.shape[0]} != {N} for {smts_path}"
                )
            smt_mask = (smts > 1)

        mask = dynamic_mask & smt_mask

        local_points = local_points[mask]
        local_colors = local_colors[mask]

        # ---------- random downsample per frame ----------
        if local_points.shape[0] < sample_per_frame:
            # not enough points in this frame after masking; skip
            continue

        sample_idx = np.random.choice(
            np.arange(local_points.shape[0]),
            sample_per_frame,
            replace=False,
        )
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]

        # ---------- transform to world ----------
        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        points.append(local_points_w)
        colors.append(local_colors)

    if len(points) == 0:
        raise RuntimeError("No points collected; check masks/semantics/TOTAL_POINTS.")

    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)

    # ---------- write PLY ----------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_path = os.path.join(OUT_DIR, "points3d_v1.ply")
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved point cloud to {out_path}")
    print(f"Num points: {points.shape[0]}")


if __name__ == "__main__":
    main()

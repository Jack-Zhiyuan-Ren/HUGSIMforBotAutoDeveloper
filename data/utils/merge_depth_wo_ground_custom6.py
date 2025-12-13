


# import os
# import torch
# import open3d as o3d
# import json
# from imageio.v2 import imread
# import numpy as np
# import cv2
# from tqdm import tqdm
# import argparse

# #### In this version, the axes are corrected for the custom data through front_rect_mat
# ### In v2, special rotation angles are applied. For cam1 is iden, for cam2 and cam3 are specially calculated.
# #### same as v2 but without altering c2w
# # same as v4 but ran independently

# # =========================
# # HARD-CODED CONFIG
# # =========================
# OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"
# TOTAL_POINTS = 200000
# DATASET_TYPE = "waymo"   # options in this script: "nuscenes", "pandaset", "waymo", "kitti360"


# def get_opts():
#     """
#     Make arguments OPTIONAL and default to the hard-coded config above,
#     so you can just run:  python this_script.py
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--out', type=str, default=OUT_DIR,
#                         help='Output / dataset directory containing meta_data.json')
#     parser.add_argument('--total', type=int, default=TOTAL_POINTS,
#                         help='Total number of points to sample')
#     parser.add_argument('--datatype', type=str, default=DATASET_TYPE,
#                         choices=["nuscenes", "pandaset", "waymo", "kitti360"],
#                         help='Dataset type (currently unused, kept for consistency)')
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = get_opts()

#     out_dir = args.out
#     total_points = args.total

#     with open(os.path.join(out_dir, 'meta_data.json'), 'r') as rf:
#         meta_data = json.load(rf)

#     ##########################################################################
#     #                       unproject pixels                                 #
#     ##########################################################################

#     points, colors = [], []
#     sample_per_frame = total_points // len(meta_data["frames"])

#     for frame in tqdm(meta_data["frames"]):
#         intrinsic = np.array(frame["intrinsics"])
#         c2w = np.array(frame["camtoworld"])

#         cx, cy, fx, fy = (
#             intrinsic[0, 2],
#             intrinsic[1, 2],
#             intrinsic[0, 0],
#             intrinsic[1, 1],
#         )
#         H, W = frame["height"], frame["width"]

#         rgb_path = frame["rgb_path"]

#         if "/cam_1/" not in rgb_path:
#             continue


#         frame_cam = frame["rgb_path"].split("/")[-2]
#         im = np.array(imread(os.path.join(out_dir, rgb_path)))

#         depth_path = os.path.join(
#             out_dir,
#             rgb_path.replace("images", "depth")
#                    .replace("./", "")
#                    .replace(".jpg", ".pt")
#                    .replace(".png", ".pt"),
#         )
#         depth = torch.load(depth_path).numpy()

#         # generate pixel coordinates
#         x = np.arange(0, depth.shape[1])
#         y = np.arange(0, depth.shape[0])
#         xx, yy = np.meshgrid(x, y)
#         pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)

#         # unproject depth to pointcloud (camera frame)
#         x = (pixels[..., 0] - cx) * depth.reshape(-1) / fx
#         y = (pixels[..., 1] - cy) * depth.reshape(-1) / fy
#         z = depth.reshape(-1)
#         local_points = np.stack([x, y, z], axis=1)
#         local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

#         # -------------------------
#         # dynamic + semantic masks
#         # -------------------------
#         # start with "keep everything" masks so script won't crash if files missing
#         dynamic_mask = np.ones(local_points.shape[0], dtype=bool)
#         smt_mask = np.ones(local_points.shape[0], dtype=bool)

#         # dynamic mask
#         mask_path = os.path.join(
#             out_dir,
#             rgb_path.replace('images', 'masks')
#                     .replace('./', '')
#                     .replace('.jpg', '.npy')
#                     .replace('.png', '.npy')
#         )
#         if os.path.exists(mask_path):
#             dynamic_mask = np.load(mask_path).reshape(-1)

#         # non-ground semantics (smts > 1)
#         smts_path = os.path.join(
#             out_dir,
#             rgb_path.replace("images", "semantics")
#                     .replace("./", "")
#                     .replace(".jpg", ".npy")
#                     .replace(".png", ".npy"),
#         )
#         if os.path.exists(smts_path):
#             smts = np.load(smts_path).reshape(-1)
#             smt_mask = smts > 1

#         mask = dynamic_mask & smt_mask

#         local_points = local_points[mask]
#         local_colors = local_colors[mask]

#         # random downsample
#         if local_points.shape[0] < sample_per_frame:
#             continue

#         sample_idx = np.random.choice(
#             np.arange(local_points.shape[0]), sample_per_frame, replace=False
#         )
#         local_points = local_points[sample_idx]
#         local_colors = local_colors[sample_idx]

#         # transform to world coordinates
#         local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

#         points.append(local_points_w)
#         colors.append(local_colors)

#     if len(points) == 0:
#         raise RuntimeError("No points collected — check masks / paths / meta_data.json.")

#     points = np.concatenate(points, axis=0)
#     colors = np.concatenate(colors, axis=0)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     out_ply = os.path.join(out_dir, "points3d_v1.ply")
#     o3d.io.write_point_cloud(out_ply, pcd)
#     print(f"Saved point cloud to: {out_ply}")


import os
import json
import argparse

import torch
import numpy as np
import open3d as o3d
from imageio.v2 import imread
from tqdm import tqdm

#### In this version:
# - Only uses the "front" camera per dataset
# - For Waymo, that means ONLY /cam_1/
# - Builds a single point cloud: points3d_v1.ply (non-ground, dynamic-masked)
# Change from v5: only do cam1 


# =========================
# HARD-CODED CONFIG
# =========================
OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"
TOTAL_POINTS = 200000
DATASET_TYPE = "waymo"   # options: "nuscenes", "pandaset", "waymo", "kitti360"


def get_opts():
    """
    Arguments are OPTIONAL and default to the hard-coded config above,
    so you can just run:  python this_script.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default=OUT_DIR,
        help="Dataset directory containing meta_data.json",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=TOTAL_POINTS,
        help="Total number of points to sample",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default=DATASET_TYPE,
        choices=["nuscenes", "pandaset", "waymo", "kitti360"],
        help="Dataset type (controls which camera is treated as 'front')",
    )
    return parser.parse_args()


def is_front_cam_frame(rgb_path: str, datatype: str) -> bool:
    """Return True if this frame is from the 'front' camera for the given dataset."""
    if datatype == "nuscenes":
        return "/CAM_FRONT/" in rgb_path
    elif datatype == "pandaset":
        return "/front_camera/" in rgb_path
    elif datatype == "waymo":
        # *** Only cam_1 for Waymo ***
        return "/cam_1/" in rgb_path
    elif datatype == "kitti360":
        return "/cam_0/" in rgb_path
    else:
        raise NotImplementedError(f"Unknown datatype: {datatype}")


if __name__ == "__main__":
    args = get_opts()

    out_dir = args.out
    total_points = args.total
    datatype = args.datatype

    # ------------------------------
    # Load meta_data.json
    # ------------------------------
    meta_path = os.path.join(out_dir, "meta_data_v31.json")
    with open(meta_path, "r") as rf:
        meta_data = json.load(rf)

    frames = meta_data["frames"]

    # ------------------------------
    # Select only front-camera frames
    # For Waymo: ONLY cam_1 frames
    # ------------------------------
    front_frames = [f for f in frames if is_front_cam_frame(f["rgb_path"], datatype)]

    if len(front_frames) == 0:
        raise RuntimeError(
            f"No front-camera frames found for datatype='{datatype}'. "
            f"Check rgb_path patterns in meta_data.json."
        )

    # Distribute TOTAL_POINTS across front-camera frames only
    sample_per_frame = total_points // len(front_frames)
    if sample_per_frame == 0:
        raise RuntimeError(
            f"TOTAL_POINTS={total_points} is too small for {len(front_frames)} frames."
        )

    print(f"Using {len(front_frames)} front-camera frames.")
    print(f"Sampling ~{sample_per_frame} points per frame (target total ~{sample_per_frame * len(front_frames)}).")

    ##########################################################################
    #                       Unproject pixels                                 #
    ##########################################################################

    all_points, all_colors = [], []

    for frame in tqdm(front_frames, desc="Unprojecting front-camera frames"):
        intrinsic = np.array(frame["intrinsics"], dtype=float)
        c2w = np.array(frame["camtoworld"], dtype=float)

        cx, cy, fx, fy = (
            intrinsic[0, 2],
            intrinsic[1, 2],
            intrinsic[0, 0],
            intrinsic[1, 1],
        )

        rgb_path = frame["rgb_path"]
        img_full_path = os.path.join(out_dir, rgb_path)

        # Read RGB
        im = np.array(imread(img_full_path))
        # H, W = im.shape[:2]
        H, W = frame["height"], frame["width"]

        # Read depth
        depth_path = os.path.join(
            out_dir,
            rgb_path.replace("images", "depth")
                    .replace("./", "")
                    .replace(".jpg", ".pt")
                    .replace(".png", ".pt"),
        )
        depth = torch.load(depth_path).numpy()  # assume (H, W)

        # generate pixel coordinates
        x = np.arange(0, W)
        y = np.arange(0, H)
        xx, yy = np.meshgrid(x, y)
        pixels = np.vstack((xx.ravel(), yy.ravel())).T  # (H*W, 2)

        depth_flat = depth.reshape(-1)

        # unproject depth to pointcloud (camera frame)
        X = (pixels[:, 0] - cx) * depth_flat / fx
        Y = (pixels[:, 1] - cy) * depth_flat / fy
        Z = depth_flat
        local_points = np.stack([X, Y, Z], axis=1)  # (N, 3)
        local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

        # -------------------------
        # dynamic + semantic masks
        # -------------------------
        # Start with "keep everything" masks so script won't crash if files missing
        dynamic_mask = np.ones(local_points.shape[0], dtype=bool)
        smt_mask = np.ones(local_points.shape[0], dtype=bool)

        # dynamic mask
        mask_path = os.path.join(
            out_dir,
            rgb_path.replace("images", "masks")
                    .replace("./", "")
                    .replace(".jpg", ".npy")
                    .replace(".png", ".npy"),
        )
        if os.path.exists(mask_path):
            dynamic_mask = np.load(mask_path).reshape(-1).astype(bool)

        # semantic mask: keep NON-ground (smts > 1) as in your previous script
        smts_path = os.path.join(
            out_dir,
            rgb_path.replace("images", "semantics")
                    .replace("./", "")
                    .replace(".jpg", ".npy")
                    .replace(".png", ".npy"),
        )
        if os.path.exists(smts_path):
            smts = np.load(smts_path).reshape(-1)
            smt_mask = smts > 1  # non-ground only; change to <=1 if you want ground

        mask = dynamic_mask & smt_mask

        local_points = local_points[mask]
        local_colors = local_colors[mask]

        # random downsample
        if local_points.shape[0] < sample_per_frame:
            # not enough points in this frame after masking; skip it
            continue

        sample_idx = np.random.choice(
            np.arange(local_points.shape[0]), sample_per_frame, replace=False
        )
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]

        # transform to world coordinates
        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        all_points.append(local_points_w)
        all_colors.append(local_colors)

    if len(all_points) == 0:
        raise RuntimeError("No points collected — check masks / depth / meta_data.json / cam_1 paths.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    print(f"Collected {points.shape[0]} points total.")

    # ------------------------------
    # Save point cloud
    # ------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_ply = os.path.join(out_dir, "points3d_v31.ply")  # cam_1 only in this script
    #points3d_v3.ply -4.7 degrees rotation about x axis
    #points3d_v4.ply -5 degree rotation about x axis
    #points3d_v5.ply -4 degrees rotation about x axis
    #points3d_v6.ply -1 degree rotation about x axis
    #points3d_v7.ply -2 degree rotation about x axis
    #points3d_v8.ply -1.8 degree rotation about x axis
    #points3d_v9.ply -1.5 degree rotation about x axis
    #points3d_v10.ply -1.6 degree rotation about x axis
    #points3d_v11.ply +1 degree rotation about x axis
    #points3d_v12.ply -5 degree rotation about x axis
    #points3d_v13.ply -10 degree rotation about x axis
    #points3d_v14.ply z offsets 60 at last frame
    #points3d_v15.ply z offsets 60 at last frame. -1 degree rotation about x axis.
    #points3d_v16.ply z offsets +10 for every frame.
    #points3d_v17.ply z offsets -10 for every frame.
    #points3d_v18.ply z offsets -10 for every frame. 98th frame z value reduced from 92 to 82.
    #points3d_v19.ply cam1 rotated -20 degrees along x axis.
    #points3d_v20.ply cam1 rotated -15 degrees along x axis.
    #points3d_v21.ply cam1 rotated -15 degrees along x axis. 98th frame z value reduced from 92 to 82.
    #points3d_v22.ply cam1 rotated -12 degrees along x axis. 98th frame z value reduced from 92 to 82.
    #points3d_v23.ply cam1 rotated -14 degrees along x axis. 98th frame z value reduced from 92 to 82.
    #points3d_v24.ply cam1 rotated -13.8 degrees along x axis. 98th frame z value reduced from 92 to 82 with linear interpolation for other frames.
    #pooints3d_v25.ply cam1 rotated -21 degrees along x axis. 98th frame z value reduced from 92 to 82.
    #pooints3d_v26.ply cam1 rotated -23 degrees along x axis. 98th frame z value reduced from 92 to 82.
    #points3d_v27.ply cam1 rotated -45 degrees along x axis. 98th frame z value reduced from 92 to 82.
    #points3d_v28.ply cam1 rotated -40 degrees along x axis. 98th frame z value reduced from 92 to 82.
    #points3d_v29.ply cam1 rotated -50 degrees along x axis. 98th frame z value reduced from 92 to 82. 
    #points3d_v30.ply cam1 no rotation change. 98th frame z value reduced from 92 to 82.
    #points3d_v31.ply cam1 rotated -4.8 degrees along x axis. 98th frame z value reduced from 92 to 82.



    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"Saved point cloud to: {out_ply}")

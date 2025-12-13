


# import os
# import json
# import argparse

# import torch
# import numpy as np
# import open3d as o3d
# from imageio.v2 import imread
# from tqdm import tqdm

# #### In this version:
# # - Only uses the "front" camera per dataset
# # - For Waymo, that means ONLY /cam_1/
# # - Builds a single point cloud: points3d_v1.ply (non-ground, dynamic-masked)
# # Change from v5: only do cam1 
# # Change from v6: you can limit the amount of frames you are using. Overlay the local points on the image before applying them to world.


# # =========================
# # HARD-CODED CONFIG
# # =========================
# OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"
# TOTAL_POINTS = 200000
# DATASET_TYPE = "waymo"   # options: "nuscenes", "pandaset", "waymo", "kitti360"


# def get_opts():
#     """
#     Arguments are OPTIONAL and default to the hard-coded config above,
#     so you can just run:  python this_script.py
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--out",
#         type=str,
#         default=OUT_DIR,
#         help="Dataset directory containing meta_data.json",
#     )
#     parser.add_argument(
#         "--total",
#         type=int,
#         default=TOTAL_POINTS,
#         help="Total number of points to sample",
#     )
#     parser.add_argument(
#         "--datatype",
#         type=str,
#         default=DATASET_TYPE,
#         choices=["nuscenes", "pandaset", "waymo", "kitti360"],
#         help="Dataset type (controls which camera is treated as 'front')",
#     )
#     return parser.parse_args()


# def is_front_cam_frame(rgb_path: str, datatype: str) -> bool:
#     """Return True if this frame is from the 'front' camera for the given dataset."""
#     if datatype == "nuscenes":
#         return "/CAM_FRONT/" in rgb_path
#     elif datatype == "pandaset":
#         return "/front_camera/" in rgb_path
#     elif datatype == "waymo":
#         # *** Only cam_1 for Waymo ***
#         return "/cam_1/" in rgb_path
#     elif datatype == "kitti360":
#         return "/cam_0/" in rgb_path
#     else:
#         raise NotImplementedError(f"Unknown datatype: {datatype}")


# if __name__ == "__main__":
#     args = get_opts()

#     out_dir = args.out
#     total_points = args.total
#     datatype = args.datatype

#     # ------------------------------
#     # Load meta_data.json
#     # ------------------------------
#     meta_path = os.path.join(out_dir, "meta_data_v31.json")
#     with open(meta_path, "r") as rf:
#         meta_data = json.load(rf)

#     frames = meta_data["frames"]

#     # ------------------------------
#     # Select only front-camera frames
#     # For Waymo: ONLY cam_1 frames
#     # ------------------------------
#     front_frames = [f for f in frames if is_front_cam_frame(f["rgb_path"], datatype)]

#     if len(front_frames) == 0:
#         raise RuntimeError(
#             f"No front-camera frames found for datatype='{datatype}'. "
#             f"Check rgb_path patterns in meta_data.json."
#         )

#     # Distribute TOTAL_POINTS across front-camera frames only
#     sample_per_frame = total_points // len(front_frames)
#     if sample_per_frame == 0:
#         raise RuntimeError(
#             f"TOTAL_POINTS={total_points} is too small for {len(front_frames)} frames."
#         )

#     print(f"Using {len(front_frames)} front-camera frames.")
#     print(f"Sampling ~{sample_per_frame} points per frame (target total ~{sample_per_frame * len(front_frames)}).")

#     ##########################################################################
#     #                       Unproject pixels                                 #
#     ##########################################################################

#     all_points, all_colors = [], []

#     for frame in tqdm(front_frames, desc="Unprojecting front-camera frames"):
#         intrinsic = np.array(frame["intrinsics"], dtype=float)
#         c2w = np.array(frame["camtoworld"], dtype=float)

#         cx, cy, fx, fy = (
#             intrinsic[0, 2],
#             intrinsic[1, 2],
#             intrinsic[0, 0],
#             intrinsic[1, 1],
#         )

#         rgb_path = frame["rgb_path"]
#         img_full_path = os.path.join(out_dir, rgb_path)

#         # Read RGB
#         im = np.array(imread(img_full_path))
#         # H, W = im.shape[:2]
#         H, W = frame["height"], frame["width"]

#         # Read depth
#         depth_path = os.path.join(
#             out_dir,
#             rgb_path.replace("images", "depth")
#                     .replace("./", "")
#                     .replace(".jpg", ".pt")
#                     .replace(".png", ".pt"),
#         )
#         depth = torch.load(depth_path).numpy()  # assume (H, W)

#         # generate pixel coordinates
#         x = np.arange(0, W)
#         y = np.arange(0, H)
#         xx, yy = np.meshgrid(x, y)
#         pixels = np.vstack((xx.ravel(), yy.ravel())).T  # (H*W, 2)

#         depth_flat = depth.reshape(-1)

#         # unproject depth to pointcloud (camera frame)
#         X = (pixels[:, 0] - cx) * depth_flat / fx
#         Y = (pixels[:, 1] - cy) * depth_flat / fy
#         Z = depth_flat
#         local_points = np.stack([X, Y, Z], axis=1)  # (N, 3)
#         local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

#         # -------------------------
#         # dynamic + semantic masks
#         # -------------------------
#         # Start with "keep everything" masks so script won't crash if files missing
#         dynamic_mask = np.ones(local_points.shape[0], dtype=bool)
#         smt_mask = np.ones(local_points.shape[0], dtype=bool)

#         # dynamic mask
#         mask_path = os.path.join(
#             out_dir,
#             rgb_path.replace("images", "masks")
#                     .replace("./", "")
#                     .replace(".jpg", ".npy")
#                     .replace(".png", ".npy"),
#         )
#         if os.path.exists(mask_path):
#             dynamic_mask = np.load(mask_path).reshape(-1).astype(bool)

#         # semantic mask: keep NON-ground (smts > 1) as in your previous script
#         smts_path = os.path.join(
#             out_dir,
#             rgb_path.replace("images", "semantics")
#                     .replace("./", "")
#                     .replace(".jpg", ".npy")
#                     .replace(".png", ".npy"),
#         )
#         if os.path.exists(smts_path):
#             smts = np.load(smts_path).reshape(-1)
#             smt_mask = smts > 1  # non-ground only; change to <=1 if you want ground

#         mask = dynamic_mask & smt_mask

#         local_points = local_points[mask]
#         local_colors = local_colors[mask]

#         # random downsample
#         if local_points.shape[0] < sample_per_frame:
#             # not enough points in this frame after masking; skip it
#             continue

#         sample_idx = np.random.choice(
#             np.arange(local_points.shape[0]), sample_per_frame, replace=False
#         )
#         local_points = local_points[sample_idx]
#         local_colors = local_colors[sample_idx]

#         # transform to world coordinates
#         local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

#         all_points.append(local_points_w)
#         all_colors.append(local_colors)

#     if len(all_points) == 0:
#         raise RuntimeError("No points collected — check masks / depth / meta_data.json / cam_1 paths.")

#     points = np.concatenate(all_points, axis=0)
#     colors = np.concatenate(all_colors, axis=0)

#     print(f"Collected {points.shape[0]} points total.")

#     # ------------------------------
#     # Save point cloud
#     # ------------------------------
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     out_ply = os.path.join(out_dir, "points3d_v31.ply")  # cam_1 only in this script




#     o3d.io.write_point_cloud(out_ply, pcd)
#     print(f"Saved point cloud to: {out_ply}")


import os
import json

import torch
import numpy as np
import open3d as o3d
from imageio.v2 import imread
from tqdm import tqdm
import cv2

# #### In this version:
# # - Only uses the "front" camera per dataset
# # - For Waymo, that means ONLY /cam_1/
# # - Builds a single point cloud: points3d_v1.ply (non-ground, dynamic-masked)
# # Change from v5: only do cam1 
# # Change from v6: you can limit the amount of frames you are using. Overlay the local points on the image before applying them to world.


# =========================
# HARD-CODED CONFIG
# =========================
OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v7"
TOTAL_POINTS = 200000
DATASET_TYPE = "waymo"   # options: "nuscenes", "pandaset", "waymo", "kitti360"
MAX_FRAMES = 98          # <--- hardcoded: only use first front-camera frame


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
    # Use hardcoded values
    out_dir = OUT_DIR
    total_points = TOTAL_POINTS
    datatype = DATASET_TYPE
    max_frames = MAX_FRAMES

    # ------------------------------
    # Load meta_data.json
    # ------------------------------
    meta_path = os.path.join(out_dir, "meta_data_v2.json")
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

    print(f"Found {len(front_frames)} front-camera frames in meta_data.")

    # ------------------------------
    # Limit number of frames (hardcoded)
    # ------------------------------
    if max_frames is not None and max_frames > 0:
        front_frames = front_frames[:max_frames]
        print(f"Limiting to first {len(front_frames)} front-camera frames (MAX_FRAMES={max_frames}).")

    num_active_frames = len(front_frames)
    if num_active_frames == 0:
        raise RuntimeError("After applying MAX_FRAMES, there are no frames to process.")

    # Distribute TOTAL_POINTS across *active* front-camera frames only
    sample_per_frame = total_points // num_active_frames
    if sample_per_frame == 0:
        raise RuntimeError(
            f"TOTAL_POINTS={total_points} is too small for {num_active_frames} frames."
        )

    print(f"Using {num_active_frames} front-camera frames.")
    print(f"Sampling ~{sample_per_frame} points per frame "
          f"(target total ~{sample_per_frame * num_active_frames}).")

    # Directory to save overlay images
    overlay_dir = os.path.join(out_dir, "debug_3d_overlays_v7")
    os.makedirs(overlay_dir, exist_ok=True)
    print(f"Overlay images will be saved in: {overlay_dir}")

    ##########################################################################
    #                       Unproject pixels                                 #
    ##########################################################################

    all_points, all_colors = [], []

    for frame_idx, frame in enumerate(tqdm(front_frames, desc="Unprojecting front-camera frames")):
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

        # semantic mask: keep NON-ground (smts > 1)
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
        pixels_masked = pixels[mask]  # matching pixel coords for remaining points

        # random downsample
        if local_points.shape[0] < sample_per_frame:
            # not enough points in this frame after masking; skip it
            continue

        sample_idx = np.random.choice(
            np.arange(local_points.shape[0]), sample_per_frame, replace=False
        )
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]
        pixel_samples = pixels_masked[sample_idx]

        # -------------------------
        # OVERLAY LOCAL POINTS ON IMAGE (pixel space)
        # -------------------------
        overlay = im.copy()
        # imread gives RGB, OpenCV expects BGR
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        for (u, v) in pixel_samples:
            u_i = int(round(u))
            v_i = int(round(v))
            if 0 <= u_i < W and 0 <= v_i < H:
                cv2.circle(overlay_bgr, (u_i, v_i), 1, (0, 0, 255), -1)  # red dots

        base_name = os.path.basename(rgb_path)
        overlay_name = f"overlay_{frame_idx:04d}_{base_name}"
        overlay_path = os.path.join(overlay_dir, overlay_name)
        cv2.imwrite(overlay_path, overlay_bgr)

        # -------------------------
        # transform to world coordinates
        # -------------------------
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

    out_ply = os.path.join(out_dir, "points3d_v7.ply")  # cam_1 only in this script
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"Saved point cloud to: {out_ply}")

    ######
    #v1: only 1 frame, meta_data
    #v2: 2frames, meta_data
    #v3: 10frames, meta_data
    #v4: 30frames, meta_data
    #v5: 98frames, meta_data
    #v6: 98frames, meta_data_v1
    #v7: 98frames, meta_data_v2



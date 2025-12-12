


# # #### In this version:
# # # - Only uses the "front" camera per dataset
# # # - For Waymo, that means ONLY /cam_1/
# # # - Builds a single point cloud: points3d_v1.ply (non-ground, dynamic-masked)
# # # Change from v5: only do cam1 
# # # Change from v6: you can limit the amount of frames you are using. Overlay the local points on the image before applying them to world.
# change from v7 do all three cams
# change from v8, account for distortion
# change from v9, dependent version

import os
import json
import argparse

import torch
import numpy as np
import open3d as o3d
from imageio.v2 import imread
from tqdm import tqdm
import cv2

# =========================
# HARD-CODED CONFIG
# =========================
OUT_DIR_DEFAULT = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v11"
TOTAL_POINTS_DEFAULT = 200000
DATASET_TYPE_DEFAULT = "waymo"   # options: "nuscenes", "pandaset", "waymo", "kitti360"
MAX_FRAMES_DEFAULT = None          # hardcoded: max number of frames to use
# =========================

# Distortion coefficients per Waymo camera (k1, k2, p1, p2, k3)
# Match these to your cam*_intr_ds vectors
DIST_COEFFS = {
    "cam_1": np.array([-0.317,  0.1197,    0.0, 0.0, 0.0],     dtype=np.float32),
    "cam_2": np.array([-0.317,  0.1197, 0.0, 0.0, 0.0],     dtype=np.float32),
    "cam_3": np.array([-0.3165, 0.1199, 0.0, 0.0, 0.0],     dtype=np.float32),
}

def get_opts():
    parser = argparse.ArgumentParser(
        description="Merge depth without ground into a non-ground point cloud (points3d_v10.ply)."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=OUT_DIR_DEFAULT,
        help=f"Output / dataset root directory (default: {OUT_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=TOTAL_POINTS_DEFAULT,
        help=f"Total number of non-ground points to sample (default: {TOTAL_POINTS_DEFAULT})",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default=DATASET_TYPE_DEFAULT,
        choices=["nuscenes", "pandaset", "waymo", "kitti360"],
        help=f"Dataset type (default: {DATASET_TYPE_DEFAULT})",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=MAX_FRAMES_DEFAULT,
        help=f"Max number of frames to use (default: {MAX_FRAMES_DEFAULT}, <=0 means use all)",
    )
    return parser.parse_args()


def is_used_cam_frame(rgb_path: str, datatype: str) -> bool:
    """
    Return True if this frame is from a camera we want to use.
    For Waymo, we now use cam_1, cam_2, and cam_3.
    """
    if datatype == "nuscenes":
        return "/CAM_FRONT/" in rgb_path
    elif datatype == "pandaset":
        return "/front_camera/" in rgb_path
    elif datatype == "waymo":
        # Use cam_1, cam_2, cam_3
        return ("/cam_1/" in rgb_path) or ("/cam_2/" in rgb_path) or ("/cam_3/" in rgb_path)
    elif datatype == "kitti360":
        return "/cam_0/" in rgb_path
    else:
        raise NotImplementedError(f"Unknown datatype: {datatype}")


if __name__ == "__main__":
    # Parse CLI args
    args = get_opts()

    out_dir = args.out
    total_points = args.total
    datatype = args.datatype
    max_frames = args.max_frames

    # ------------------------------
    # Load meta_data.json
    # ------------------------------
    meta_path = os.path.join(out_dir, "meta_data.json")
    with open(meta_path, "r") as rf:
        meta_data = json.load(rf)

    frames = meta_data["frames"]

    # ------------------------------
    # Select frames from desired cameras
    # For Waymo: cam_1, cam_2, cam_3
    # ------------------------------
    used_frames = [f for f in frames if is_used_cam_frame(f["rgb_path"], datatype)]

    if len(used_frames) == 0:
        raise RuntimeError(
            f"No selected-camera frames found for datatype='{datatype}'. "
            f"Check rgb_path patterns in meta_data.json."
        )

    print(f"Found {len(used_frames)} selected-camera frames in meta_data.")

    # ------------------------------
    # Limit number of frames (hardcoded)
    # ------------------------------
    if max_frames is not None and max_frames > 0:
        used_frames = used_frames[:max_frames]
        print(f"Limiting to first {len(used_frames)} frames (MAX_FRAMES={max_frames}).")

    num_active_frames = len(used_frames)
    if num_active_frames == 0:
        raise RuntimeError("After applying MAX_FRAMES, there are no frames to process.")

    # Distribute TOTAL_POINTS across *active* frames from all selected cameras
    sample_per_frame = total_points // num_active_frames
    if sample_per_frame == 0:
        raise RuntimeError(
            f"TOTAL_POINTS={total_points} is too small for {num_active_frames} frames."
        )

    print(f"Using {num_active_frames} frames (cam_1, cam_2, cam_3 for Waymo).")
    print(
        f"Sampling ~{sample_per_frame} points per frame "
        f"(target total ~{sample_per_frame * num_active_frames})."
    )

    # Directory to save overlay images (root)
    overlay_root = os.path.join(out_dir, "debug_3d_overlays")
    os.makedirs(overlay_root, exist_ok=True)
    print(f"Overlay images root will be: {overlay_root}")

    ##########################################################################
    #                       Unproject pixels                                 #
    ##########################################################################

    all_points, all_colors = [], []

    for frame_idx, frame in enumerate(tqdm(used_frames, desc="Unprojecting selected-camera frames")):
        intrinsic = np.array(frame["intrinsics"], dtype=float)
        c2w = np.array(frame["camtoworld"], dtype=float)

        # cx, cy, fx, fy = (
        #     intrinsic[0, 2],
        #     intrinsic[1, 2],
        #     intrinsic[0, 0],
        #     intrinsic[1, 1],
        # )

        rgb_path = frame["rgb_path"]
        img_full_path = os.path.join(out_dir, rgb_path)

        # Figure out which cam this frame is (cam_1 / cam_2 / cam_3)
        cam_folder = rgb_path.split("/")[-2]   # e.g. "cam_1"
        cam_id_str = cam_folder.split("_")[-1] # "1"
        cam_overlay_dir = os.path.join(overlay_root, f"cam{cam_id_str}")
        os.makedirs(cam_overlay_dir, exist_ok=True)

        cam_name = cam_folder  # "cam_1", "cam_2", "cam_3"

        # Read RGB
        im = np.array(imread(img_full_path))
        H, W = frame["height"], frame["width"]  

        # ----- build distortion-aware K (newK) -----
        K3 = intrinsic[:3, :3].astype(np.float32)
        dist = DIST_COEFFS.get(cam_name, np.zeros(5, dtype=np.float32))

        # This mirrors what you did for depth: original K + dist -> newK
        newK, roi = cv2.getOptimalNewCameraMatrix(K3, dist, (W, H), alpha=0)


        fx = newK[0, 0]
        fy = newK[1, 1]
        cx = newK[0, 2]
        cy = newK[1, 2]

        # fx = K3[0, 0]
        # fy = newK3[1, 1]
        # cx = newK3[0, 2]
        # cy = newK3[1, 2]

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

        # -------------------------
        # VISUALIZE MASKS (optional debug)
        # -------------------------

        comb_vis = (mask.reshape(H, W).astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(cam_overlay_dir, f"combined_mask_{frame_idx:04d}.png"), comb_vis)
        ############################# 

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
        overlay_path = os.path.join(cam_overlay_dir, overlay_name)
        cv2.imwrite(overlay_path, overlay_bgr)

        # -------------------------
        # transform to world coordinates
        # -------------------------
        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        all_points.append(local_points_w)
        all_colors.append(local_colors)

    if len(all_points) == 0:
        raise RuntimeError("No points collected — check masks / depth / meta_data.json / cam_1/2/3 paths.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    print(f"Collected {points.shape[0]} points total.")

    # ------------------------------
    # Save point cloud
    # ------------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_ply = os.path.join(out_dir, "points3d.ply")  # cam_1, cam_2, cam_3 in this script
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"Saved point cloud to: {out_ply}")

    ######
    # v1: only 1 frame, meta_data
    # v2: 2frames, meta_data
    # v3: 10frames, meta_data
    # v4: 30frames, meta_data
    # v5: 98frames, meta_data
    # v6: 98frames, meta_data_v1
    # v7: 98frames, meta_data_v2, 
    # v8: 98frames, meta_data_v2, cam_1, cam_2, cam_3
    # v9: 297 frames, meta_data_v3, cam_1, cam_2, cam_3, new origin with the middle_extr
    # v10:Not really important just trying with Max_frames = None
    





import os
import torch
import open3d as o3d
import json
from imageio.v2 import imread
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import argparse

#### In this version, the axes are corrected for the custom data through front_rect_mat
### In v2, special rotation angles are applied. For cam1 is iden, for cam2 and cam3 are specially calculated.
# Will also run as an independent
# Same as v2 but no altering to c2w
# Same as v3 but run as independent
# change from v4: only do cam1s
# change v5: overlay sampled local points on image + limit frames (hardcoded)
# change v6: use cam_1, cam_2, cam_3 and per-cam overlay folders
# change from v7, create dynamic masks. #currently no dynamic masks applied.
# change from v8, same as v8 but dependent


# # =========================
# # HARD-CODED CONFIG
# # =========================
OUT_DIR_DEFAULT      = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v11"
TOTAL_POINTS_DEFAULT  = 200000
DATASET_TYPE_DEFAULT  = "waymo"   # options: "nuscenes", "pandaset", "waymo", "kitti360"
MAX_FRAMES_DEFAULT    = None        # limit number of frames used
# # =========================


def get_opts():
    parser = argparse.ArgumentParser(
        description="Merge depth + ground semantics into ground_points3d_xx.ply and ground_param_v5.pkl"
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
        help=f"Total number of ground points to sample (default: {TOTAL_POINTS_DEFAULT})",
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
        help=f"Max number of camera frames to use (default: {MAX_FRAMES_DEFAULT}, <=0 means use all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()

    out_dir = args.out
    total_points = args.total
    datatype = args.datatype
    max_frames = args.max_frames

    with open(os.path.join(out_dir, "meta_data.json"), "r") as rf:
        meta_data = json.load(rf)

    ##########################################################################
    #                 select frames for the cameras                          #
    ##########################################################################

    used_frames = []       # frames we will actually use (cam_1, cam_2, cam_3 for waymo)
    front_cam_poses = []   # keep pose list (currently for all used frames)

    for frame in meta_data["frames"]:
        rgb_path = frame["rgb_path"]

        use_this_frame = False
        if datatype == "nuscenes":
            if "/CAM_FRONT/" in rgb_path:
                use_this_frame = True
        elif datatype == "pandaset":
            if "/front_camera/" in rgb_path:
                use_this_frame = True
        elif datatype == "waymo":
            # use cam_1 only (extend here if you want cam_2/cam_3 as well)
            if "/cam_1/" in rgb_path or("/cam_2/" in rgb_path) or ("/cam_3/" in rgb_path):
                use_this_frame = True
        elif datatype == "kitti360":
            if "/cam_0/" in rgb_path:
                use_this_frame = True
        else:
            raise NotImplementedError(f"Unknown datatype: {datatype}")

        if use_this_frame:
            used_frames.append(frame)
            if "/cam_1/" in rgb_path:
                front_cam_poses.append(np.array(frame["camtoworld"]))

    if len(used_frames) == 0:
        raise RuntimeError("No camera frames found (e.g. no /cam_1/, /cam_2/, /cam_3/ frames for Waymo).")

    print(f"Found {len(used_frames)} cam_1/cam_2/cam_3 frames in meta_data_v2.json")

    # ---- optionally limit number of frames (hardcoded) ----
    if max_frames is not None and max_frames > 0:
        used_frames = used_frames[:max_frames]
        front_cam_poses = front_cam_poses[:max_frames]
        print(f"Limiting to first {len(used_frames)} frames (MAX_FRAMES={max_frames})")

    # distribute TOTAL_POINTS across the selected frames
    sample_per_frame = total_points // len(used_frames)
    if sample_per_frame == 0:
        raise RuntimeError(
            f"TOTAL_POINTS={total_points} too small for {len(used_frames)} frames."
        )
    print(f"Sampling ~{sample_per_frame} points per frame (cams 1/2/3).")

    ##########################################################################
    #                        unproject pixels                                #
    ##########################################################################
    
    points, colors = [], []

    # directory root to save overlays of sampled ground points on images
    overlay_root = os.path.join(out_dir, "debug_ground_overlays_v1")
    os.makedirs(overlay_root, exist_ok=True)
    print(f"Overlay images root will be: {overlay_root}")

    for idx, frame in enumerate(tqdm(used_frames, desc="Building ground from cams 1/2/3")):
        intrinsic = np.array(frame["intrinsics"])
        c2w = np.array(frame["camtoworld"])

        cx, cy, fx, fy = (
            intrinsic[0, 2],
            intrinsic[1, 2],
            intrinsic[0, 0],
            intrinsic[1, 1],
        )
        H, W = frame["height"], frame["width"]

        rgb_path = frame["rgb_path"]
        frame_cam = rgb_path.split("/")[-2]  # e.g. "cam_1", "cam_2", "cam_3"
        cam_id_str = frame_cam.split("_")[-1]  # "1", "2", "3"

        img_full_path = os.path.join(out_dir, rgb_path)
        im = np.array(imread(img_full_path))  # RGB

        depth_path = os.path.join(
            out_dir,
            rgb_path.replace("images", "depth")
            .replace("./", "")
            .replace(".jpg", ".pt")
            .replace(".png", ".pt"),
        )
        depth = torch.load(depth_path).numpy()  # (H,W)

        # generate pixel coordinates
        x = np.arange(0, depth.shape[1])
        y = np.arange(0, depth.shape[0])
        xx, yy = np.meshgrid(x, y)
        pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)  # (N,2)

        depth_flat = depth.reshape(-1)

        # unproject depth to pointcloud (camera frame)
        x = (pixels[..., 0] - cx) * depth_flat / fx
        y = (pixels[..., 1] - cy) * depth_flat / fy
        z = depth_flat
        local_points = np.stack([x, y, z], axis=1)  # (N,3)
        local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

        # start with "keep everything" mask
        mask = np.ones(local_points.shape[0], dtype=bool)


        # ground semantics (keep smts <= 1)
        smts_path = os.path.join(
            out_dir,
            rgb_path.replace("images", "semantics")
            .replace("./", "")
            .replace(".jpg", ".npy")
            .replace(".png", ".npy"),
        )
        if os.path.exists(smts_path):
            smts = np.load(smts_path).reshape(-1)
            mask &= (smts <= 1)

        

        # apply mask to points, colors, and pixel coords
        local_points = local_points[mask]
        local_colors = local_colors[mask]
        pixels_masked = pixels[mask]

        # random downsample
        if local_points.shape[0] < sample_per_frame:
            # skip if not enough valid ground pixels in this frame
            continue
        sample_idx = np.random.choice(
            np.arange(local_points.shape[0]), sample_per_frame, replace=False
        )
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]
        pixel_samples = pixels_masked[sample_idx]  # (sample_per_frame, 2)

        # -------------------------
        # OVERLAY SAMPLED LOCAL POINTS ON ORIGINAL IMAGE
        # -------------------------
        overlay = im.copy()                        # RGB
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR

        for (u, v) in pixel_samples:
            u_i = int(round(u))
            v_i = int(round(v))
            if 0 <= u_i < W and 0 <= v_i < H:
                cv2.circle(overlay_bgr, (u_i, v_i), 1, (0, 0, 255), -1)  # red dots

        # per-cam subfolder: cam1, cam2, cam3
        cam_overlay_dir = os.path.join(overlay_root, f"cam{cam_id_str}")
        os.makedirs(cam_overlay_dir, exist_ok=True)

        base_name = os.path.basename(rgb_path)
        overlay_name = f"ground_overlay_cam{cam_id_str}_{idx:04d}_{base_name}"
        overlay_path = os.path.join(cam_overlay_dir, overlay_name)
        cv2.imwrite(overlay_path, overlay_bgr)

        # to world coordinates
        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        points.append(local_points_w)
        colors.append(local_colors)

    if len(points) == 0:
        raise RuntimeError("No ground points collected. Check masks / depth / semantics / cam_1/2/3 paths.")

    points = np.concatenate(points)
    colors = np.concatenate(colors)

    ##########################################################################
    #                    Multi-Plane Ground Model                            #
    ##########################################################################
    
    # Read camera height info (still use front_info.json)
    if datatype == "kitti360":
        front_cam_height = 1.55
    elif datatype == "pandaset":
        front_cam_height = 2.2
    else:
        with open(os.path.join(out_dir, "front_info.json"), "r") as f:
            front_info = json.load(f)
        front_cam_height = front_info["height"] + 1
        # front_cam_height = 4

    front_cam_poses = np.stack(front_cam_poses)  # (M, 4, 4)

    # Init ground point cloud
    # distance from each point to each camera center (currently unused)
    points_cam_dist = np.sqrt(
        np.sum(
            (points[:, np.newaxis, :] - front_cam_poses[:-1, :3, 3][np.newaxis, :, :]) ** 2,
            axis=-1,
        )
    )
    
    # nearest cam (projection-to-ground currently disabled)
    nearest_cam_idx = np.argmin(points_cam_dist, axis=1)
    nearest_c2w = front_cam_poses[nearest_cam_idx]                 # (N, 4, 4)
    nearest_w2c = np.linalg.inv(front_cam_poses)[nearest_cam_idx]  # (N, 4, 4)
    points_local = (
        np.einsum("nij,nj->ni", nearest_w2c[:, :3, :3], points) + nearest_w2c[:, :3, 3]
    )  # (N, 3)
    
    # project to ground by fixing height
    points_local[:, 1] = front_cam_height
    points = (
        np.einsum("nij,nj->ni", nearest_c2w[:, :3, :3], points_local)
        + nearest_c2w[:, :3, 3]
    )  # (N, 3)

    # save ground point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    out_ply = os.path.join(out_dir, "ground_points3d_v1.ply")
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"Saved ground point cloud to: {out_ply}")

    #311238_part_0_100_v6
    # ground_points3d_v12.ply : cam_1 only, y offsets 60 at last frame
    # ground_points3d_v13.ply : cam 1 rotates -23 degrees along x axis. 98th frame z value reduced from 92 to 82.
    # ground_points3d_v14.ply : cam 1 rotates -4.8 degrees along x axis. 98th frame z value reduced from 92 to 82.
    #311238_part_0_100_v7
    # ground_points3d_v1.ply: using v2w_frame and no front_rect_mat, 98 frames, meta_data_v2.json
    # ground_points3d_v2.ply: using v2w_frame and no front_rect_mat, 10 frame, meta_data_v2.json
    # ground_points3d_v3.ply: using v2w_frame and no front_rect_mat, 98 frames, meta_data_v2.json, not restricting points to a certain height.
    # ground_points3d_v4.ply: using v2w_frame and no front_rect_mat, 297 frames, meta_data.json. Cam 1, cam2, and cam3 used.
    # ground_points3d_v5.ply: using v2w_frame and no front_rect_mat, 297 frames, meta_data.json. Cam 1, cam2, and cam3 used. Dynamic masks applied.
    # ground_points3d_v6.ply: using v2w_frame and no front_rect_mat, 297 frames, meta_data_v3.json. all three cams. restriting points to a certain height. height + 1

    #311238_part_0_100V_11
    # ground_points3d_v1.ply: Use all three cams
    
    # Get high level command
    forecast = 20
    threshold = 2.5
    high_level_commands = []
    for i, cam_pose in enumerate(front_cam_poses):
        if i + forecast < front_cam_poses.shape[0]:
            forecast_campose = front_cam_poses[i + forecast]
        else:
            forecast_campose = front_cam_poses[-1]
        inv_cam_pose = np.linalg.inv(cam_pose)
        forecast_in_curr = inv_cam_pose @ forecast_campose
        if forecast_in_curr[0, 3] > threshold:
            high_level_commands.append(0)  # right
        elif forecast_in_curr[0, 3] < -threshold:
            high_level_commands.append(1)  # left
        else:
            high_level_commands.append(2)  # forward

    print(high_level_commands)
    with open(os.path.join(out_dir, "ground_param_v1.pkl"), "wb") as f:
        pickle.dump((front_cam_poses, front_cam_height, high_level_commands), f)

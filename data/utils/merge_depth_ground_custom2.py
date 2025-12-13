import os
import torch
import open3d as o3d
import json
from imageio.v2 import imread
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import pickle


####In this version, the axes are corrected for the custom data through front_rect_mat
### In v2, special rotation angles are applied. For cam1 is iden, for cam2 and cam3 are specially calculated. 
# will also run as an independent


front_rect_mat = np.array([
                        [1.0,  0.0,  0.0],       #only use this for translation parts
                        [0.0,  0.0, 1.0],        
                        [0.0,  -1.0,  0.0],   
                    ], dtype=float)

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

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--total", type=int, default=1500000)
    parser.add_argument("--datatype", type=str, default="nuscenes")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    with open(os.path.join(args.out, "meta_data.json"), "r") as rf:
        meta_data = json.load(rf)

    ##########################################################################
    #                        unproject pixels                             #
    ##########################################################################
    
    points, colors = [], []
    sample_per_frame = args.total // len(meta_data["frames"])
    front_cam_poses = []
    for frame in tqdm(meta_data["frames"]):
        intrinsic = np.array(frame["intrinsics"])
        c2w = np.array(frame["camtoworld"])

        ############## converting c2w to new convention #################
        R_old = c2w[:3, :3]
        t_old = c2w[:3, 3]

        rgb_path = frame["rgb_path"]
        # infer camera id from path, e.g. "./images/cam_2/000030.png"
        if "/cam_1/" in rgb_path:
            cam_id = 1
        elif "/cam_2/" in rgb_path:
            cam_id = 2
        elif "/cam_3/" in rgb_path:
            cam_id = 3
        else:
            cam_id = None  # fallback

        # rotation: per-camera rule
        if cam_id == 1:
            R_new = R_old  # no change
        elif cam_id == 2:
            R_new = F_cam2 @ R_old @ F_cam2.T
        elif cam_id == 3:
            R_new = F_cam3 @ R_old @ F_cam3.T
        else:
            # if some other cam appears, just leave it unchanged
            R_new = R_old

        # translation: always left-multiply by front_rect_mat
        t_new = front_rect_mat @ t_old

        c2w[:3, :3] = R_new
        c2w[:3, 3]  = t_new
        #################################################

        if args.datatype == "nuscenes":
            if "/CAM_FRONT/" in frame["rgb_path"]:
                front_cam_poses.append(c2w)
        elif args.datatype == "pandaset":
            if "/front_camera/" in frame["rgb_path"]:
                front_cam_poses.append(c2w)
        elif args.datatype == "waymo":
            if "/cam_1/" in frame["rgb_path"]:
                front_cam_poses.append(c2w)
        elif args.datatype == "kitti360":
            if "/cam_0/" in frame["rgb_path"]:
                front_cam_poses.append(c2w)
        else:
            raise NotImplementedError

        cx, cy, fx, fy = (
            intrinsic[0, 2],
            intrinsic[1, 2],
            intrinsic[0, 0],
            intrinsic[1, 1],
        )
        H, W = frame["height"], frame["width"]

        rgb_path = frame["rgb_path"]
        frame_cam = frame["rgb_path"].split("/")[-2]
        im = np.array(imread(os.path.join(args.out, rgb_path)))
        depth_path = os.path.join(
            args.out,
            rgb_path.replace("images", "depth")
            .replace("./", "")
            .replace(".jpg", ".pt")
            .replace(".png", ".pt"),
        )
        depth = torch.load(depth_path).numpy()

        x = np.arange(0, depth.shape[1])  # generate pixel coordinates
        y = np.arange(0, depth.shape[0])
        xx, yy = np.meshgrid(x, y)
        pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)

        # unproject depth to pointcloud
        x = (pixels[..., 0] - cx) * depth.reshape(-1) / fx
        y = (pixels[..., 1] - cy) * depth.reshape(-1) / fy
        z = depth.reshape(-1)
        local_points = np.stack([x, y, z], axis=1)
        local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

        # ground semantics
        smts_path = os.path.join(
            args.out,
            rgb_path.replace("images", "semantics")
            .replace("./", "")
            .replace(".jpg", ".npy")
            .replace(".png", ".npy"),
        )
        if os.path.exists(smts_path):
            smts = np.load(smts_path).reshape(-1)
            mask = smts <= 1
            local_points = local_points[mask]
            local_colors = local_colors[mask]

        # random downsample
        if local_points.shape[0] < sample_per_frame:
            continue
        sample_idx = np.random.choice(
            np.arange(local_points.shape[0]), sample_per_frame
        )
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]

        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        points.append(local_points_w)
        colors.append(local_colors)

    points = np.concatenate(points)
    colors = np.concatenate(colors)

    ##########################################################################
    #                    Multi-Plane Ground Model                       #
    ##########################################################################
    
    # Read front cam poses
    if args.datatype == "kitti360":
        front_cam_height = 1.55
    elif args.datatype == 'pandaset':
        front_cam_height = 2.2
    else:
        with open(os.path.join(args.out, "front_info.json"), "r") as f:
            front_info = json.load(f)
        front_cam_height = front_info["height"]
        # front_rect_mat = front_info["rect_mat"]
        # front_rect_mat = np.array([
        #                         [1.0,  0.0,  0.0],   # x_cam/right stays right
        #                         [0.0,  0.0, 1.0],   # y_cam/down maps to -z_world (up vs down)
        #                         [0.0,  -1.0,  0.0],   # z_cam/forward maps to +y_world/back
        #                     ], dtype=float)

    front_cam_poses = np.stack(front_cam_poses)


    # R_raw = front_cam_poses[:, :3, :3]  
    # t_raw = front_cam_poses[:, :3, 3] 
    # front_cam_poses[:, :3, :3] = np.einsum('ij,njk->nik', front_rect_mat, R_raw)
    # front_cam_poses[:, :3, 3] = (front_rect_mat @ t_raw.T).T
    
    # Init ground point cloud
    points_cam_dist = np.sqrt(
        np.sum(
            (points[:, np.newaxis, :] - front_cam_poses[:-1, :3, 3][np.newaxis, :, :])
            ** 2,
            axis=-1,
        )
    )
    
    # nearest cam
    nearest_cam_idx = np.argmin(points_cam_dist, axis=1)
    nearest_c2w = front_cam_poses[nearest_cam_idx] # (N, 4, 4)
    nearest_w2c = np.linalg.inv(front_cam_poses)[nearest_cam_idx] # (N, 4, 4)
    points_local = (
        np.einsum("nij,nj->ni", nearest_w2c[:, :3, :3], points)
        + nearest_w2c[:, :3, 3]
    ) # (N, 3)
    points_local[:, 1] = front_cam_height
    points = (
        np.einsum("nij,nj->ni", nearest_c2w[:, :3, :3], points_local)
        + nearest_c2w[:, :3, 3]
    ) # (N, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(args.out, "ground_points3d.ply"), pcd)
    
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
            high_level_commands.append(0) # right
        elif forecast_in_curr[0, 3] < -threshold:
            high_level_commands.append(1) # left
        else:
            high_level_commands.append(2) # forward

    print(high_level_commands)
    with open(os.path.join(args.out, "ground_param.pkl"), "wb") as f:
        pickle.dump((front_cam_poses, front_cam_height, high_level_commands), f)




# ############ independent version

# import os
# import torch
# import open3d as o3d
# import json
# from imageio.v2 import imread
# import numpy as np
# import cv2
# from tqdm import tqdm
# import pickle

# #### In this version, the axes are corrected for the custom data through front_rect_mat
# #### In v2, special rotation angles are applied. For cam1 is iden, for cam2 and cam3 are specially calculated.
# #### Will also run as an independent script.

# # =========================
# # Hard-coded configuration
# # =========================
# OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v4"
# TOTAL_POINTS = 200000
# DATASET_TYPE = "waymo"   # options in this script: "nuscenes", "pandaset", "waymo", "kitti360"


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

# # F_cam3 = np.array([
# #     [ 0.99545759,  0.07426106, -0.05957747],
# #     [-0.05554631,  0.96124271,  0.27005010],
# #     [ 0.07732262, -0.26551412,  0.96100128]
# # ], dtype=float) #v1

# front_rect_mat = np.array([
#     [1.0,  0.0,  0.0],       # only use this for translation parts
#     [0.0,  0.0,  1.0],
#     [0.0, -1.0,  0.0],
# ], dtype=float)


# if __name__ == "__main__":
#     meta_path = os.path.join(OUT_DIR, "meta_data.json")
#     if not os.path.exists(meta_path):
#         raise FileNotFoundError(f"meta_data.json not found at {meta_path}")

#     with open(meta_path, "r") as rf:
#         meta_data = json.load(rf)

#     ##########################################################################
#     #                        unproject pixels                               #
#     ##########################################################################
    
#     points, colors = [], []
#     n_frames = len(meta_data["frames"])
#     if n_frames == 0:
#         raise RuntimeError("meta_data.json has no frames under 'frames' key.")

#     sample_per_frame = TOTAL_POINTS // n_frames
#     if sample_per_frame <= 0:
#         raise ValueError(
#             f"TOTAL_POINTS={TOTAL_POINTS} too small for {n_frames} frames "
#             f"(sample_per_frame={sample_per_frame})."
#         )

#     front_cam_poses = []
#     for frame in tqdm(meta_data["frames"], desc="Unprojecting frames"):
#         intrinsic = np.array(frame["intrinsics"])
#         c2w = np.array(frame["camtoworld"])

#         ############## converting c2w to new convention #################
#         R_old = c2w[:3, :3]
#         t_old = c2w[:3, 3]

#         # rgb_path = frame["rgb_path"]

#         #         # --- NEW: keep only front camera for Waymo ---
#         # if DATASET_TYPE == "waymo" and "/cam_1/" not in rgb_path:
#         #     continue
#         # # ---------------------------------------------
 
#         # # translation: always left-multiply by front_rect_mat
#         # t_new = front_rect_mat @ t_old

#         # c2w[:3, :3] = R_old
#         # c2w[:3, 3]  = t_new

#         #         R_old = c2w[:3, :3]
# #         t_old = c2w[:3, 3]

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

#         # collect front cam poses depending on dataset type
#         if DATASET_TYPE == "nuscenes":
#             if "/CAM_FRONT/" in rgb_path:
#                 front_cam_poses.append(c2w)
#         elif DATASET_TYPE == "pandaset":
#             if "/front_camera/" in rgb_path:
#                 front_cam_poses.append(c2w)
#         elif DATASET_TYPE == "waymo":
#             if "/cam_1/" in rgb_path:
#                 front_cam_poses.append(c2w)
#         elif DATASET_TYPE == "kitti360":
#             if "/cam_0/" in rgb_path:
#                 front_cam_poses.append(c2w)
#         else:
#             raise NotImplementedError(f"Unknown DATASET_TYPE: {DATASET_TYPE}")

#         cx, cy, fx, fy = (
#             intrinsic[0, 2],
#             intrinsic[1, 2],
#             intrinsic[0, 0],
#             intrinsic[1, 1],
#         )
#         H, W = frame["height"], frame["width"]

#         img_full_path = os.path.join(OUT_DIR, rgb_path)
#         if not os.path.exists(img_full_path):
#             raise FileNotFoundError(f"RGB image not found at {img_full_path}")
#         im = np.array(imread(img_full_path))

#         depth_path = os.path.join(
#             OUT_DIR,
#             rgb_path.replace("images", "depth")
#             .replace("./", "")
#             .replace(".jpg", ".pt")
#             .replace(".png", ".pt"),
#         )
#         if not os.path.exists(depth_path):
#             raise FileNotFoundError(f"Depth map (.pt) not found at {depth_path}")
#         depth = torch.load(depth_path).numpy()

#         x = np.arange(0, depth.shape[1])  # generate pixel coordinates
#         y = np.arange(0, depth.shape[0])
#         xx, yy = np.meshgrid(x, y)
#         pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)

#         # unproject depth to pointcloud
#         depth_flat = depth.reshape(-1)
#         x = (pixels[..., 0] - cx) * depth_flat / fx
#         y = (pixels[..., 1] - cy) * depth_flat / fy
#         z = depth_flat
#         local_points = np.stack([x, y, z], axis=1)
#         local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

#         # ground semantics
#         smts_path = os.path.join(
#             OUT_DIR,
#             rgb_path.replace("images", "semantics")
#             .replace("./", "")
#             .replace(".jpg", ".npy")
#             .replace(".png", ".npy"),
#         )
#         if os.path.exists(smts_path):
#             smts = np.load(smts_path).reshape(-1)
#             mask = smts <= 1
#             local_points = local_points[mask]
#             local_colors = local_colors[mask]

#         # random downsample
#         if local_points.shape[0] < sample_per_frame:
#             continue
#         sample_idx = np.random.choice(
#             np.arange(local_points.shape[0]), sample_per_frame, replace=False
#         )
#         local_points = local_points[sample_idx]
#         local_colors = local_colors[sample_idx]

#         local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

#         points.append(local_points_w)
#         colors.append(local_colors)

#     if len(points) == 0:
#         raise RuntimeError("No points collected. Check semantics or TOTAL_POINTS / meta_data.json.")

#     points = np.concatenate(points)
#     colors = np.concatenate(colors)

#     ##########################################################################
#     #                    Multi-Plane Ground Model                            #
#     ##########################################################################
    
#     # Read front cam poses
#     if len(front_cam_poses) == 0:
#         raise RuntimeError("No front camera poses collected. Check DATASET_TYPE and rgb_path patterns.")

#     if DATASET_TYPE == "kitti360":
#         front_cam_height = 1.55
#     elif DATASET_TYPE == "pandaset":
#         front_cam_height = 2.2
#     else:
#         front_info_path = os.path.join(OUT_DIR, "front_info.json")
#         if not os.path.exists(front_info_path):
#             raise FileNotFoundError(f"front_info.json not found at {front_info_path}")
#         with open(front_info_path, "r") as f:
#             front_info = json.load(f)
#         front_cam_height = front_info["height"]

#     front_cam_poses = np.stack(front_cam_poses)

#     # Init ground point cloud
#     points_cam_dist = np.sqrt(
#         np.sum(
#             (points[:, np.newaxis, :] - front_cam_poses[:-1, :3, 3][np.newaxis, :, :])
#             ** 2,
#             axis=-1,
#         )
#     )
    
#     # nearest cam
#     nearest_cam_idx = np.argmin(points_cam_dist, axis=1)
#     nearest_c2w = front_cam_poses[nearest_cam_idx] # (N, 4, 4)
#     nearest_w2c_all = np.linalg.inv(front_cam_poses)  # (F, 4, 4)
#     nearest_w2c = nearest_w2c_all[nearest_cam_idx]    # (N, 4, 4)

#     points_local = (
#         np.einsum("nij,nj->ni", nearest_w2c[:, :3, :3], points)
#         + nearest_w2c[:, :3, 3]
#     ) # (N, 3)
#     points_local[:, 1] = front_cam_height
#     points = (
#         np.einsum("nij,nj->ni", nearest_c2w[:, :3, :3], points_local)
#         + nearest_c2w[:, :3, 3]
#     ) # (N, 3)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     ply_path = os.path.join(OUT_DIR, "ground_points3d_v4.ply")
#     o3d.io.write_point_cloud(ply_path, pcd)
#     print(f"Saved ground point cloud to {ply_path}")
    
#     # Get high level command
    
#     forecast = 20
#     threshold = 2.5
#     high_level_commands = []
#     for i, cam_pose in enumerate(front_cam_poses):
#         if i + forecast < front_cam_poses.shape[0]:
#             forecast_campose = front_cam_poses[i + forecast]
#         else:
#             forecast_campose = front_cam_poses[-1]
#         inv_cam_pose = np.linalg.inv(cam_pose)
#         forecast_in_curr = inv_cam_pose @ forecast_campose
#         if forecast_in_curr[0, 3] > threshold:
#             high_level_commands.append(0) # right
#         elif forecast_in_curr[0, 3] < -threshold:
#             high_level_commands.append(1) # left
#         else:
#             high_level_commands.append(2) # forward

#     print("High level commands (0=right, 1=left, 2=forward):")
#     print(high_level_commands)
#     pkl_path = os.path.join(OUT_DIR, "ground_param_v4.pkl")
#     with open(pkl_path, "wb") as f:
#         pickle.dump((front_cam_poses, front_cam_height, high_level_commands), f)
#     print(f"Saved ground parameters to {pkl_path}")


# import os
# import torch
# import open3d as o3d
# import json
# from imageio.v2 import imread
# import numpy as np
# import cv2
# from tqdm import tqdm
# import pickle

#### In this version, the axes are corrected for the custom data through front_rect_mat
### In v2, special rotation angles are applied. For cam1 is iden, for cam2 and cam3 are specially calculated.
# Will also run as an independent
# Same as v2 but no altering to c2w
# Same as v3 but run as independent
# change from v4: only do cam1s


# # =========================
# # HARD-CODED CONFIG
# # =========================
# OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"
# TOTAL_POINTS = 200000
# DATASET_TYPE = "waymo"   # options: "nuscenes", "pandaset", "waymo", "kitti360"


# if __name__ == "__main__":
#     out_dir = OUT_DIR
#     total_points = TOTAL_POINTS
#     datatype = DATASET_TYPE

#     with open(os.path.join(out_dir, "meta_data.json"), "r") as rf:
#         meta_data = json.load(rf)

#     ##########################################################################
#     #                        unproject pixels                                #
#     ##########################################################################
    
#     points, colors = [], []
#     sample_per_frame = total_points // len(meta_data["frames"])
#     front_cam_poses = []

#     front_cam_frames = []   # frames we will actually use (cam_1 for waymo)



#     for frame in tqdm(meta_data["frames"]):
#         intrinsic = np.array(frame["intrinsics"])
#         c2w = np.array(frame["camtoworld"])


#         use_this_frame = False

#         # Collect front camera poses depending on dataset type
#         if datatype == "nuscenes":
#             if "/CAM_FRONT/" in frame["rgb_path"]:
#                 front_cam_poses.append(c2w)
#         elif datatype == "pandaset":
#             if "/front_camera/" in frame["rgb_path"]:
#                 front_cam_poses.append(c2w)
#         elif datatype == "waymo":
#             if "/cam_1/" in frame["rgb_path"]:
#                 use_this_frame = True
#                 front_cam_poses.append(c2w)
#         elif datatype == "kitti360":
#             if "/cam_0/" in frame["rgb_path"]:
#                 front_cam_poses.append(c2w)
#         else:
#             raise NotImplementedError(f"Unknown datatype: {datatype}")

#         cx, cy, fx, fy = (
#             intrinsic[0, 2],
#             intrinsic[1, 2],
#             intrinsic[0, 0],
#             intrinsic[1, 1],
#         )
#         H, W = frame["height"], frame["width"]

#         rgb_path = frame["rgb_path"]
#         frame_cam = frame["rgb_path"].split("/")[-2]
#         im = np.array(imread(os.path.join(out_dir, rgb_path)))
#         depth_path = os.path.join(
#             out_dir,
#             rgb_path.replace("images", "depth")
#             .replace("./", "")
#             .replace(".jpg", ".pt")
#             .replace(".png", ".pt"),
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

#         # ground semantics (keep smts <= 1)
#         smts_path = os.path.join(
#             out_dir,
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

#         # to world coordinates
#         local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

#         points.append(local_points_w)
#         colors.append(local_colors)

#     points = np.concatenate(points)
#     colors = np.concatenate(colors)

#     ##########################################################################
#     #                    Multi-Plane Ground Model                            #
#     ##########################################################################
    
#     # Read front cam poses
#     if datatype == "kitti360":
#         front_cam_height = 1.55
#     elif datatype == "pandaset":
#         front_cam_height = 2.2
#     else:
#         with open(os.path.join(out_dir, "front_info.json"), "r") as f:
#             front_info = json.load(f)
#         front_cam_height = front_info["height"]

#     front_cam_poses = np.stack(front_cam_poses)

#     # Init ground point cloud
#     points_cam_dist = np.sqrt(
#         np.sum(
#             (points[:, np.newaxis, :] - front_cam_poses[:-1, :3, 3][np.newaxis, :, :]) ** 2,
#             axis=-1,
#         )
#     )
    
#     # nearest cam
#     nearest_cam_idx = np.argmin(points_cam_dist, axis=1)
#     nearest_c2w = front_cam_poses[nearest_cam_idx]                 # (N, 4, 4)
#     nearest_w2c = np.linalg.inv(front_cam_poses)[nearest_cam_idx]  # (N, 4, 4)
#     points_local = (
#         np.einsum("nij,nj->ni", nearest_w2c[:, :3, :3], points) + nearest_w2c[:, :3, 3]
#     )  # (N, 3)

#     # project to ground by fixing height
#     points_local[:, 1] = front_cam_height
#     points = (
#         np.einsum("nij,nj->ni", nearest_c2w[:, :3, :3], points_local)
#         + nearest_c2w[:, :3, 3]
#     )  # (N, 3)

#     # save ground point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     o3d.io.write_point_cloud(os.path.join(out_dir, "ground_points3d_v1.ply"), pcd)
    
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
#             high_level_commands.append(0)  # right
#         elif forecast_in_curr[0, 3] < -threshold:
#             high_level_commands.append(1)  # left
#         else:
#             high_level_commands.append(2)  # forward

#     print(high_level_commands)
#     with open(os.path.join(out_dir, "ground_param_v1.pkl"), "wb") as f:
#         pickle.dump((front_cam_poses, front_cam_height, high_level_commands), f)

import os
import torch
import open3d as o3d
import json
from imageio.v2 import imread
import numpy as np
import cv2
from tqdm import tqdm
import pickle

#### In this version, the axes are corrected for the custom data through front_rect_mat
### In v2, special rotation angles are applied. For cam1 is iden, for cam2 and cam3 are specially calculated.
# Will also run as an independent
# Same as v2 but no altering to c2w
# Same as v3 but run as independent
# change from v4: only do cam1s


# =========================
# HARD-CODED CONFIG
# =========================
OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"
TOTAL_POINTS = 200000
DATASET_TYPE = "waymo"   # options: "nuscenes", "pandaset", "waymo", "kitti360"


if __name__ == "__main__":
    out_dir = OUT_DIR
    total_points = TOTAL_POINTS
    datatype = DATASET_TYPE

    with open(os.path.join(out_dir, "meta_data_v31.json"), "r") as rf:
        meta_data = json.load(rf)

    ##########################################################################
    #                 select frames for the *front* camera                  #
    ##########################################################################

    front_cam_frames = []   # frames we will actually use (cam_1 for waymo)
    front_cam_poses = []    # corresponding cam2worlds

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
            # *** change from v4: only use cam_1 ***
            if "/cam_1/" in rgb_path:
                use_this_frame = True
        elif datatype == "kitti360":
            if "/cam_0/" in rgb_path:
                use_this_frame = True
        else:
            raise NotImplementedError(f"Unknown datatype: {datatype}")

        if use_this_frame:
            front_cam_frames.append(frame)
            front_cam_poses.append(np.array(frame["camtoworld"]))

    if len(front_cam_frames) == 0:
        raise RuntimeError("No front-camera frames found (e.g. no /cam_1/ frames for Waymo).")

    # distribute TOTAL_POINTS only across the selected front camera frames
    sample_per_frame = total_points // len(front_cam_frames)

    ##########################################################################
    #                        unproject pixels                                #
    ##########################################################################
    
    points, colors = [], []

    for frame in tqdm(front_cam_frames, desc="Building ground from front camera"):
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
        frame_cam = rgb_path.split("/")[-2]  # not used but left for debugging if needed
        im = np.array(imread(os.path.join(out_dir, rgb_path)))
        depth_path = os.path.join(
            out_dir,
            rgb_path.replace("images", "depth")
            .replace("./", "")
            .replace(".jpg", ".pt")
            .replace(".png", ".pt"),
        )
        depth = torch.load(depth_path).numpy()

        # generate pixel coordinates
        x = np.arange(0, depth.shape[1])
        y = np.arange(0, depth.shape[0])
        xx, yy = np.meshgrid(x, y)
        pixels = np.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)

        # unproject depth to pointcloud (camera frame)
        x = (pixels[..., 0] - cx) * depth.reshape(-1) / fx
        y = (pixels[..., 1] - cy) * depth.reshape(-1) / fy
        z = depth.reshape(-1)
        local_points = np.stack([x, y, z], axis=1)
        local_colors = im.reshape(-1, 3).astype(np.float32) / 255.0

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
            mask = smts <= 1
            local_points = local_points[mask]
            local_colors = local_colors[mask]

        # random downsample
        if local_points.shape[0] < sample_per_frame:
            # skip if not enough valid ground pixels in this frame
            continue
        sample_idx = np.random.choice(
            np.arange(local_points.shape[0]), sample_per_frame, replace=False
        )
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]

        # to world coordinates
        local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]

        points.append(local_points_w)
        colors.append(local_colors)

    points = np.concatenate(points)
    colors = np.concatenate(colors)

    ##########################################################################
    #                    Multi-Plane Ground Model                            #
    ##########################################################################
    
    # Read front cam poses
    if datatype == "kitti360":
        front_cam_height = 1.55
    elif datatype == "pandaset":
        front_cam_height = 2.2
    else:
        with open(os.path.join(out_dir, "front_info.json"), "r") as f:
            front_info = json.load(f)
        front_cam_height = front_info["height"]
        # front_cam_height = 4

    front_cam_poses = np.stack(front_cam_poses)  # (M, 4, 4)

    # Init ground point cloud
    # distance from each point to each front camera center
    points_cam_dist = np.sqrt(
        np.sum(
            (points[:, np.newaxis, :] - front_cam_poses[:-1, :3, 3][np.newaxis, :, :]) ** 2,
            axis=-1,
        )
    )
    
    # nearest cam
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
    o3d.io.write_point_cloud(os.path.join(out_dir, "ground_points3d_v14.ply"), pcd)

    # ground_points3d_v12.ply : cam_1 only, y offsets 60 at last frame
    # ground_points3d_v13.ply : cam 1 rotates -23 degrees along x axis. 98th frame z value reduced from 92 to 82.
    # ground_points3d_v14.ply : cam 1 rotates -4.8 degrees along x axis. 98th frame z value reduced from 92 to 82.
    
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
    with open(os.path.join(out_dir, "ground_param_v12.pkl"), "wb") as f:
        pickle.dump((front_cam_poses, front_cam_height, high_level_commands), f)

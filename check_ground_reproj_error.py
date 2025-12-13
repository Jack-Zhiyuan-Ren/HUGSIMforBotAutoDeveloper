# import os
# import json
# import numpy as np
# import torch
# from imageio.v2 import imread
# import open3d as o3d
# from tqdm import tqdm

# # ============================
# # CONFIG – EDIT
# # ============================
# # OUT_DIR   = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"
# OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1083056"
# PLY_NAME  = "ground_points3d.ply"   # or whatever you're using for training
# MAX_FRAMES = 30                        # limit for speed
# SAMPLE_POINTS_PER_FRAME = 50000        # how many PLY points to sample per frame
# # ============================


# def is_front_cam(frame):
#     # adjust if your path is different
#     return "/cam_1/" in frame["rgb_path"]


# def main():
#     meta_path = os.path.join(OUT_DIR, "meta_data.json")
#     with open(meta_path, "r") as f:
#         meta_data = json.load(f)

#     frames = [fr for fr in meta_data["frames"] if is_front_cam(fr)]
#     if MAX_FRAMES is not None:
#         frames = frames[:MAX_FRAMES]

#     # load ground point cloud (world coords)
#     ply_path = os.path.join(OUT_DIR, PLY_NAME)
#     print("Loading PLY:", ply_path)
#     pcd = o3d.io.read_point_cloud(ply_path)
#     pts_world_all = np.asarray(pcd.points, dtype=np.float32)
#     print("Total PLY points:", pts_world_all.shape[0])

#     all_err = []

#     for frame in tqdm(frames, desc="Frames"):
#         intrinsic = np.array(frame["intrinsics"], dtype=np.float32)
#         K = intrinsic[:3, :3]

#         c2w = np.array(frame["camtoworld"], dtype=np.float32)
#         w2c = np.linalg.inv(c2w)

#         rgb_path = frame["rgb_path"]
#         # depth
#         depth_path = os.path.join(
#             OUT_DIR,
#             rgb_path.replace("images", "depth")
#                    .replace("./", "")
#                    .replace(".jpg", ".pt")
#                    .replace(".png", ".pt"),
#         )
#         if not os.path.exists(depth_path):
#             print("Missing depth:", depth_path)
#             continue
#         depth = torch.load(depth_path).numpy()  # (H,W)
#         H, W = depth.shape

#         # randomly sample some PLY points for this frame
#         if pts_world_all.shape[0] <= SAMPLE_POINTS_PER_FRAME:
#             sample_idx = np.arange(pts_world_all.shape[0])
#         else:
#             sample_idx = np.random.choice(
#                 pts_world_all.shape[0],
#                 SAMPLE_POINTS_PER_FRAME,
#                 replace=False,
#             )
#         pts_world = pts_world_all[sample_idx]  # (M,3)

#         # world -> camera
#         pts_cam = (w2c[:3, :3] @ pts_world.T).T + w2c[:3, 3]  # (M,3)
#         x = pts_cam[:, 0]
#         y = pts_cam[:, 1]
#         z = pts_cam[:, 2]

#         # keep only points in front
#         front_mask = z > 0.1
#         x = x[front_mask]
#         y = y[front_mask]
#         z = z[front_mask]

#         if x.size == 0:
#             continue

#         # camera -> pixels
#         pts_cam2 = np.stack([x, y, z], axis=0)  # (3,M)
#         uvw = K @ pts_cam2
#         u = (uvw[0] / uvw[2]).astype(np.int32)
#         v = (uvw[1] / uvw[2]).astype(np.int32)

#         in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
#         u = u[in_bounds]
#         v = v[in_bounds]
#         z_proj = z[in_bounds]

#         if u.size == 0:
#             continue

#         # depth from image at those pixels
#         z_true = depth[v, u]

#         # keep only valid depth > 0
#         valid_depth = z_true > 0
#         z_true = z_true[valid_depth]
#         z_proj = z_proj[valid_depth]

#         if z_true.size == 0:
#             continue

#         # error in meters along the ray
#         err = z_true - z_proj
#         all_err.append(err)

#     if not all_err:
#         print("No valid samples collected.")
#         return

#     all_err = np.concatenate(all_err)

#     print("\n===== Reprojection depth error (PLY vs depth) =====")
#     print("Num samples:", all_err.shape[0])
#     print("Mean error   (m):", np.mean(all_err))
#     print("Median error (m):", np.median(all_err))
#     print("Std          (m):", np.std(all_err))
#     p5  = np.percentile(all_err, 5)
#     p95 = np.percentile(all_err, 95)
#     print("5th pct (m):", p5, " 95th pct (m):", p95)


# if __name__ == "__main__":
#     main()



# cam1 only

# import os
# import json
# import numpy as np
# import torch
# from imageio.v2 import imread
# import open3d as o3d
# from tqdm import tqdm

# # ============================
# # CONFIG – EDIT THESE
# # ============================
# OUT_DIR   = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v7"
# # OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1083056"
# PLY_NAME  = "points3d_v5.ply"   # or whatever PLY you want to check
# MAX_FRAMES = 98                     # max number of front-cam frames to use
# SAMPLE_POINTS_PER_FRAME = 50000        # PLY points to sample per frame
# Z_MIN = 1.0                            # ignore points closer than this (m)
# # ============================


# def is_front_cam(frame):
#     """Return True if this frame is front camera (cam_1)."""
#     rgb_path = frame["rgb_path"]
#     return "/cam_1/" in rgb_path     # adjust if your naming differs


# def main():
#     meta_path = os.path.join(OUT_DIR, "meta_data.json")
#     with open(meta_path, "r") as f:
#         meta_data = json.load(f)

#     # ---- only cam_1 frames ----
#     frames_all = meta_data["frames"]
#     frames = [fr for fr in frames_all if is_front_cam(fr)]
#     if not frames:
#         print("No cam_1 frames found – check is_front_cam().")
#         return
#     if MAX_FRAMES is not None:
#         frames = frames[:MAX_FRAMES]

#     # ---- load ground point cloud (world coords) ----
#     ply_path = os.path.join(OUT_DIR, PLY_NAME)
#     print("Loading PLY:", ply_path)
#     pcd = o3d.io.read_point_cloud(ply_path)
#     pts_world_all = np.asarray(pcd.points, dtype=np.float32)
#     print("Total PLY points:", pts_world_all.shape[0])

#     all_err = []

#     for frame in tqdm(frames, desc="Front-cam frames"):
#         intrinsic = np.array(frame["intrinsics"], dtype=np.float32)
#         K = intrinsic[:3, :3]

#         c2w = np.array(frame["camtoworld"], dtype=np.float32)
#         w2c = np.linalg.inv(c2w)

#         rgb_path = frame["rgb_path"]

#         depth_path = os.path.join(
#             OUT_DIR,
#             rgb_path.replace("images", "depth")
#                     .replace("./", "")
#                     .replace(".jpg", ".pt")
#                     .replace(".png", ".pt"),
#         )
#         if not os.path.exists(depth_path):
#             print("Missing depth:", depth_path)
#             continue

#         depth = torch.load(depth_path).numpy()  # (H,W)
#         H, W = depth.shape

#         # ---- sample some PLY points ----
#         if pts_world_all.shape[0] <= SAMPLE_POINTS_PER_FRAME:
#             sample_idx = np.arange(pts_world_all.shape[0])
#         else:
#             sample_idx = np.random.choice(
#                 pts_world_all.shape[0],
#                 SAMPLE_POINTS_PER_FRAME,
#                 replace=False,
#             )
#         pts_world = pts_world_all[sample_idx]  # (M,3)

#         # world -> camera
#         pts_cam = (w2c[:3, :3] @ pts_world.T).T + w2c[:3, 3]  # (M,3)
#         x = pts_cam[:, 0]
#         y = pts_cam[:, 1]
#         z = pts_cam[:, 2]

#         # keep only points in front of camera
#         front_mask = z > Z_MIN
#         x = x[front_mask]
#         y = y[front_mask]
#         z = z[front_mask]

#         if x.size == 0:
#             continue

#         # camera -> pixels
#         pts_cam2 = np.stack([x, y, z], axis=0)  # (3,M)
#         uvw = K @ pts_cam2
#         u = (uvw[0] / uvw[2]).astype(np.int32)
#         v = (uvw[1] / uvw[2]).astype(np.int32)

#         in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
#         u = u[in_bounds]
#         v = v[in_bounds]
#         z_proj = z[in_bounds]

#         if u.size == 0:
#             continue

#         # depth from image
#         z_true = depth[v, u]

#         # only valid depth
#         valid_depth = z_true > 0
#         z_true = z_true[valid_depth]
#         z_proj = z_proj[valid_depth]

#         if z_true.size == 0:
#             continue

#         # error along camera ray (meters)
#         err = z_true - z_proj
#         all_err.append(err)

#     if not all_err:
#         print("No valid samples collected.")
#         return

#     all_err = np.concatenate(all_err)

#     print("\n===== Reprojection depth error (PLY vs depth, cam_1 only) =====")
#     print("Num samples:", all_err.shape[0])
#     print("Mean error   (m):", np.mean(all_err))
#     print("Median error (m):", np.median(all_err))
#     print("Std          (m):", np.std(all_err))
#     p5  = np.percentile(all_err, 5)
#     p95 = np.percentile(all_err, 95)
#     print("5th pct (m):", p5, " 95th pct (m):", p95)


# if __name__ == "__main__":
#     main()


# cam1 only
# saves the reprojection overlays

# cam1 only

# import os
# import json
# import numpy as np
# import torch
# from imageio.v2 import imread
# import open3d as o3d
# from tqdm import tqdm
# import cv2   ### NEW

# # ============================
# # CONFIG – EDIT THESE
# # ============================
# OUT_DIR   = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v7"
# # OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1083056"
# PLY_NAME  = "ground_points3d_v3.ply"      # or whatever PLY you want to check
# MAX_FRAMES = 98                  # max number of front-cam frames to use
# SAMPLE_POINTS_PER_FRAME = 50000    # PLY points to sample per frame
# Z_MIN = 1.0                        # ignore points closer than this (m)
# # ============================


# def is_front_cam(frame):
#     """Return True if this frame is front camera (cam_1)."""
#     rgb_path = frame["rgb_path"]
#     return "/cam_1/" in rgb_path     # adjust if your naming differs


# def main():
#     meta_path = os.path.join(OUT_DIR, "meta_data_v2.json")
#     with open(meta_path, "r") as f:
#         meta_data = json.load(f)

#     # ---- only cam_1 frames ----
#     frames_all = meta_data["frames"]
#     frames = [fr for fr in frames_all if is_front_cam(fr)]
#     if not frames:
#         print("No cam_1 frames found – check is_front_cam().")
#         return
#     if MAX_FRAMES is not None:
#         frames = frames[:MAX_FRAMES]

#     # ---- load ground point cloud (world coords) ----
#     ply_path = os.path.join(OUT_DIR, PLY_NAME)
#     print("Loading PLY:", ply_path)
#     pcd = o3d.io.read_point_cloud(ply_path)
#     pts_world_all = np.asarray(pcd.points, dtype=np.float32)
#     print("Total PLY points:", pts_world_all.shape[0])

#     # ---- directory to save reprojection overlays ----  ### NEW
#     reproj_dir = os.path.join(OUT_DIR, "debug_ground_reproj_point3d_v3_cam1")
#     os.makedirs(reproj_dir, exist_ok=True)
#     print("Reprojection overlays will be saved in:", reproj_dir)

#     all_err = []

#     for frame_idx, frame in enumerate(tqdm(frames, desc="Front-cam frames")):  ### NEW: frame_idx
#         intrinsic = np.array(frame["intrinsics"], dtype=np.float32)
#         K = intrinsic[:3, :3]

#         c2w = np.array(frame["camtoworld"], dtype=np.float32)
#         w2c = np.linalg.inv(c2w)

#         rgb_path = frame["rgb_path"]

#         # ---- load depth ----
#         depth_path = os.path.join(
#             OUT_DIR,
#             rgb_path.replace("images", "depth")
#                     .replace("./", "")
#                     .replace(".jpg", ".pt")
#                     .replace(".png", ".pt"),
#         )
#         if not os.path.exists(depth_path):
#             print("Missing depth:", depth_path)
#             continue

#         depth = torch.load(depth_path).numpy()  # (H,W)
#         H, W = depth.shape

#         # ---- load RGB image for visualization ----  ### NEW
#         img_full_path = os.path.join(OUT_DIR, rgb_path)
#         if not os.path.exists(img_full_path):
#             print("Missing RGB image:", img_full_path)
#             continue
#         rgb = np.array(imread(img_full_path))  # (H,W,3) RGB
#         if rgb.shape[0] != H or rgb.shape[1] != W:
#             print("Warning: RGB / depth size mismatch:", rgb.shape, depth.shape)

#         # ---- sample some PLY points ----
#         if pts_world_all.shape[0] <= SAMPLE_POINTS_PER_FRAME:
#             sample_idx = np.arange(pts_world_all.shape[0])
#         else:
#             sample_idx = np.random.choice(
#                 pts_world_all.shape[0],
#                 SAMPLE_POINTS_PER_FRAME,
#                 replace=False,
#             )
#         pts_world = pts_world_all[sample_idx]  # (M,3)

#         # world -> camera
#         pts_cam = (w2c[:3, :3] @ pts_world.T).T + w2c[:3, 3]  # (M,3)
#         x = pts_cam[:, 0]
#         y = pts_cam[:, 1]
#         z = pts_cam[:, 2]

#         # keep only points in front of camera
#         front_mask = z > Z_MIN
#         x = x[front_mask]
#         y = y[front_mask]
#         z = z[front_mask]

#         if x.size == 0:
#             continue

#         # camera -> pixels
#         pts_cam2 = np.stack([x, y, z], axis=0)  # (3,M)
#         uvw = K @ pts_cam2
#         u = (uvw[0] / uvw[2]).astype(np.int32)
#         v = (uvw[1] / uvw[2]).astype(np.int32)

#         in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
#         u = u[in_bounds]
#         v = v[in_bounds]
#         z_proj = z[in_bounds]

#         if u.size == 0:
#             continue

#         # depth from image
#         z_true = depth[v, u]

#         # only valid depth
#         valid_depth = z_true > 0
#         z_true = z_true[valid_depth]
#         z_proj = z_proj[valid_depth]
#         u_valid = u[valid_depth]          ### NEW
#         v_valid = v[valid_depth]          ### NEW

#         if z_true.size == 0:
#             continue

#         # error along camera ray (meters)
#         err = z_true - z_proj
#         all_err.append(err)

#         # ==============================
#         # VISUALIZE REPROJECTED POINTS  ### NEW
#         # ==============================
#         overlay = rgb.copy()  # RGB
#         overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR

#         # Draw each valid reprojected point as a small red dot
#         for uu, vv in zip(u_valid, v_valid):
#             if 0 <= uu < W and 0 <= vv < H:
#                 cv2.circle(overlay_bgr, (int(uu), int(vv)), 1, (0, 0, 255), -1)

#         base_name = os.path.basename(rgb_path)
#         overlay_name = f"reproj_cam1_{frame_idx:04d}_{base_name}"
#         overlay_path = os.path.join(reproj_dir, overlay_name)
#         cv2.imwrite(overlay_path, overlay_bgr)
#         # ==============================

#     if not all_err:
#         print("No valid samples collected.")
#         return

#     all_err = np.concatenate(all_err)

#     print("\n===== Reprojection depth error (PLY vs depth, cam_1 only) =====")
#     print("Num samples:", all_err.shape[0])
#     print("Mean error   (m):", np.mean(all_err))
#     print("Median error (m):", np.median(all_err))
#     print("Std          (m):", np.std(all_err))
#     p5  = np.percentile(all_err, 5)
#     p95 = np.percentile(all_err, 95)
#     print("5th pct (m):", p5, " 95th pct (m):", p95)


# if __name__ == "__main__":
#     main()


# ## all three cams
# import os
# import json
# import numpy as np
# import torch
# from imageio.v2 import imread
# import open3d as o3d
# from tqdm import tqdm
# import cv2   ### NEW

# # ============================
# # CONFIG – EDIT THESE
# # ============================
# OUT_DIR   = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v11"
# # OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1083056"
# PLY_NAME  = "ground_points3d_v1.ply"      # or whatever PLY you want to check
# MAX_FRAMES = 297              # max number of frames to use
# SAMPLE_POINTS_PER_FRAME = 50000  # PLY points to sample per frame
# Z_MIN = 1.0                      # ignore points closer than this (m)
# # ============================


# def is_used_cam(frame):
#     """
#     Return True if this frame is from a camera we want to use.
#     For your Waymo setup, we now use cam_1, cam_2, and cam_3.
#     """
#     rgb_path = frame["rgb_path"]
#     return ("/cam_1/" in rgb_path) or ("/cam_2/" in rgb_path) or ("/cam_3/" in rgb_path)


# def main():
#     meta_path = os.path.join(OUT_DIR, "meta_data.json")
#     with open(meta_path, "r") as f:
#         meta_data = json.load(f)

#     # ---- select cam_1, cam_2, cam_3 frames ----
#     frames_all = meta_data["frames"]
#     frames = [fr for fr in frames_all if is_used_cam(fr)]
#     if not frames:
#         print("No cam_1 / cam_2 / cam_3 frames found – check is_used_cam().")
#         return
#     if MAX_FRAMES is not None:
#         frames = frames[:MAX_FRAMES]

#     # ---- load ground point cloud (world coords) ----
#     ply_path = os.path.join(OUT_DIR, PLY_NAME)
#     print("Loading PLY:", ply_path)
#     pcd = o3d.io.read_point_cloud(ply_path)
#     pts_world_all = np.asarray(pcd.points, dtype=np.float32)
#     print("Total PLY points:", pts_world_all.shape[0])

#     # ---- directory root to save reprojection overlays ----  ### NEW
#     reproj_root = os.path.join(OUT_DIR, "debug_reproj_point3d_v1")
#     os.makedirs(reproj_root, exist_ok=True)
#     print("Reprojection overlays root:", reproj_root)

#     all_err = []

#     for frame_idx, frame in enumerate(tqdm(frames, desc="Cam_1/2/3 frames")):
#         intrinsic = np.array(frame["intrinsics"], dtype=np.float32)
#         K = intrinsic[:3, :3]

#         c2w = np.array(frame["camtoworld"], dtype=np.float32)
#         w2c = np.linalg.inv(c2w)

#         rgb_path = frame["rgb_path"]

#         # ---- figure out cam id and make per-cam folder ----  ### NEW
#         # e.g., rgb_path = "./images/cam_2/000123.png"
#         cam_folder = rgb_path.split("/")[-2]         # "cam_2"
#         cam_id_str = cam_folder.split("_")[-1]       # "2"
#         reproj_dir = os.path.join(reproj_root, f"cam{cam_id_str}")
#         os.makedirs(reproj_dir, exist_ok=True)

#         # ---- load depth ----
#         depth_path = os.path.join(
#             OUT_DIR,
#             rgb_path.replace("images", "depth")
#                     .replace("./", "")
#                     .replace(".jpg", ".pt")
#                     .replace(".png", ".pt"),
#         )
#         if not os.path.exists(depth_path):
#             print("Missing depth:", depth_path)
#             continue

#         depth = torch.load(depth_path).numpy()  # (H,W)
#         H, W = depth.shape

#         # ---- load RGB image for visualization ----  ### NEW
#         img_full_path = os.path.join(OUT_DIR, rgb_path)
#         if not os.path.exists(img_full_path):
#             print("Missing RGB image:", img_full_path)
#             continue
#         rgb = np.array(imread(img_full_path))  # (H,W,3) RGB
#         if rgb.shape[0] != H or rgb.shape[1] != W:
#             print("Warning: RGB / depth size mismatch:", rgb.shape, depth.shape)

#         # ---- sample some PLY points ----
#         if pts_world_all.shape[0] <= SAMPLE_POINTS_PER_FRAME:
#             sample_idx = np.arange(pts_world_all.shape[0])
#         else:
#             sample_idx = np.random.choice(
#                 pts_world_all.shape[0],
#                 SAMPLE_POINTS_PER_FRAME,
#                 replace=False,
#             )
#         pts_world = pts_world_all[sample_idx]  # (M,3)

#         # world -> camera
#         pts_cam = (w2c[:3, :3] @ pts_world.T).T + w2c[:3, 3]  # (M,3)
#         x = pts_cam[:, 0]
#         y = pts_cam[:, 1]
#         z = pts_cam[:, 2]

#         # keep only points in front of camera
#         front_mask = z > Z_MIN
#         x = x[front_mask]
#         y = y[front_mask]
#         z = z[front_mask]

#         if x.size == 0:
#             continue

#         # camera -> pixels
#         pts_cam2 = np.stack([x, y, z], axis=0)  # (3,M)
#         uvw = K @ pts_cam2
#         u = (uvw[0] / uvw[2]).astype(np.int32)
#         v = (uvw[1] / uvw[2]).astype(np.int32)

#         in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
#         u = u[in_bounds]
#         v = v[in_bounds]
#         z_proj = z[in_bounds]

#         if u.size == 0:
#             continue

#         # depth from image
#         z_true = depth[v, u]

#         # only valid depth
#         valid_depth = z_true > 0
#         z_true = z_true[valid_depth]
#         z_proj = z_proj[valid_depth]
#         u_valid = u[valid_depth]          ### NEW
#         v_valid = v[valid_depth]          ### NEW

#         if z_true.size == 0:
#             continue

#         # error along camera ray (meters)
#         err = z_true - z_proj
#         all_err.append(err)

#         # ==============================
#         # VISUALIZE REPROJECTED POINTS  ### NEW
#         # ==============================
#         overlay = rgb.copy()  # RGB
#         overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR

#         # Draw each valid reprojected point as a small red dot
#         for uu, vv in zip(u_valid, v_valid):
#             if 0 <= uu < W and 0 <= vv < H:
#                 cv2.circle(overlay_bgr, (int(uu), int(vv)), 1, (0, 0, 255), -1)

#         base_name = os.path.basename(rgb_path)
#         overlay_name = f"reproj_cam{cam_id_str}_{frame_idx:04d}_{base_name}"
#         overlay_path = os.path.join(reproj_dir, overlay_name)
#         cv2.imwrite(overlay_path, overlay_bgr)
#         # ==============================

#     if not all_err:
#         print("No valid samples collected.")
#         return

#     all_err = np.concatenate(all_err)

#     print("\n===== Reprojection depth error (PLY vs depth, cams 1/2/3) =====")
#     print("Num samples:", all_err.shape[0])
#     print("Mean error   (m):", np.mean(all_err))
#     print("Median error (m):", np.median(all_err))
#     print("Std          (m):", np.std(all_err))
#     p5  = np.percentile(all_err, 5)
#     p95 = np.percentile(all_err, 95)
#     print("5th pct (m):", p5, " 95th pct (m):", p95)


# if __name__ == "__main__":
#     main()



# all three cams
# Distortion Aware



import os
import json
import numpy as np
import torch
from imageio.v2 import imread
import open3d as o3d
from tqdm import tqdm
import cv2   ### NEW

# ============================
# CONFIG – EDIT THESE
# ============================
OUT_DIR   = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v11"
# OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1083056"
PLY_NAME  = "points3d.ply"      # or whatever PLY you want to check
MAX_FRAMES = 297              # max number of frames to use
SAMPLE_POINTS_PER_FRAME = 50000  # PLY points to sample per frame
Z_MIN = 1.0                      # ignore points closer than this (m)
# ============================
DIST_COEFFS = {
    1: np.array([-0.317,  0.1197, 0.0, 0.0, 0.0], dtype=np.float32),  # cam_1 / FRONT
    2: np.array([-0.317,  0.1197, 0.0, 0.0, 0.0], dtype=np.float32),  # cam_2
    3: np.array([-0.3165, 0.1199, 0.0, 0.0, 0.0], dtype=np.float32),  # cam_3
}

def is_used_cam(frame):
    """
    Return True if this frame is from a camera we want to use.
    For your Waymo setup, we now use cam_1, cam_2, and cam_3.
    """
    rgb_path = frame["rgb_path"]
    return ("/cam_1/" in rgb_path) or ("/cam_2/" in rgb_path) or ("/cam_3/" in rgb_path)


def main():
    meta_path = os.path.join(OUT_DIR, "meta_data.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    # ---- select cam_1, cam_2, cam_3 frames ----
    frames_all = meta_data["frames"]
    frames = [fr for fr in frames_all if is_used_cam(fr)]
    if not frames:
        print("No cam_1 / cam_2 / cam_3 frames found – check is_used_cam().")
        return
    if MAX_FRAMES is not None:
        frames = frames[:MAX_FRAMES]

    # ---- load ground point cloud (world coords) ----
    ply_path = os.path.join(OUT_DIR, PLY_NAME)
    print("Loading PLY:", ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)
    pts_world_all = np.asarray(pcd.points, dtype=np.float32)
    print("Total PLY points:", pts_world_all.shape[0])

    # ---- directory root to save reprojection overlays ----  ### NEW
    reproj_root = os.path.join(OUT_DIR, "debug_reproj_point3d")
    os.makedirs(reproj_root, exist_ok=True)
    print("Reprojection overlays root:", reproj_root)

    all_err = []

    for frame_idx, frame in enumerate(tqdm(frames, desc="Cam_1/2/3 frames")):
        intrinsic = np.array(frame["intrinsics"], dtype=np.float32)
        K = intrinsic[:3, :3]

        c2w = np.array(frame["camtoworld"], dtype=np.float32)
        w2c = np.linalg.inv(c2w)

        rgb_path = frame["rgb_path"]

        # ---- figure out cam id and make per-cam folder ----  ### NEW
        # e.g., rgb_path = "./images/cam_2/000123.png"
        cam_folder = rgb_path.split("/")[-2]         # "cam_2"
        cam_id_str = cam_folder.split("_")[-1]       # "2"
        reproj_dir = os.path.join(reproj_root, f"cam{cam_id_str}")
        os.makedirs(reproj_dir, exist_ok=True)

        # ---- load depth ----
        depth_path = os.path.join(
            OUT_DIR,
            rgb_path.replace("images", "depth")
                    .replace("./", "")
                    .replace(".jpg", ".pt")
                    .replace(".png", ".pt"),
        )
        if not os.path.exists(depth_path):
            print("Missing depth:", depth_path)
            continue

        depth = torch.load(depth_path).numpy()  # (H,W)
        H, W = depth.shape

        # ---- load RGB image for visualization ----  ### NEW
        img_full_path = os.path.join(OUT_DIR, rgb_path)
        if not os.path.exists(img_full_path):
            print("Missing RGB image:", img_full_path)
            continue
        rgb = np.array(imread(img_full_path))  # (H,W,3) RGB
        if rgb.shape[0] != H or rgb.shape[1] != W:
            print("Warning: RGB / depth size mismatch:", rgb.shape, depth.shape)

        # ---- sample some PLY points ----
        if pts_world_all.shape[0] <= SAMPLE_POINTS_PER_FRAME:
            sample_idx = np.arange(pts_world_all.shape[0])
        else:
            sample_idx = np.random.choice(
                pts_world_all.shape[0],
                SAMPLE_POINTS_PER_FRAME,
                replace=False,
            )
        pts_world = pts_world_all[sample_idx]  # (M,3)

        # world -> camera
        pts_cam = (w2c[:3, :3] @ pts_world.T).T + w2c[:3, 3]  # (M,3)
        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        z = pts_cam[:, 2]

        # keep only points in front of camera
        front_mask = z > Z_MIN
        x = x[front_mask]
        y = y[front_mask]
        z = z[front_mask]

        if x.size == 0:
            continue

        # camera -> pixels
        pts_cam2 = np.stack([x, y, z], axis=0)  # (3,M)
        uvw = K @ pts_cam2
        u = (uvw[0] / uvw[2]).astype(np.int32)
        v = (uvw[1] / uvw[2]).astype(np.int32)

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[in_bounds]
        v = v[in_bounds]
        z_proj = z[in_bounds]

        if u.size == 0:
            continue

        # depth from image
        z_true = depth[v, u]

        # only valid depth
        valid_depth = z_true > 0
        z_true = z_true[valid_depth]
        z_proj = z_proj[valid_depth]
        u_valid = u[valid_depth]          ### NEW
        v_valid = v[valid_depth]          ### NEW

        if z_true.size == 0:
            continue

        # error along camera ray (meters)
        err = z_true - z_proj
        all_err.append(err)

        #         ----------------------------------------
        # camera -> pixels (distortion-aware)
        # # ----------------------------------------
        # Points are already in *camera* coordinates (after w2c).
        # Shape: (M,) for x,y,z after front_mask
        pts_cam_front = np.stack([x, y, z], axis=1).astype(np.float32)  # (M,3)

        # Use the 3x3 intrinsics and per-camera distortion
        K3 = K.astype(np.float32)
        dist = DIST_COEFFS.get(cam_id_str, np.zeros(5, dtype=np.float32))

        # (Optional) get "optimal" new camera matrix for this resolution.
        # We keep projecting onto the original distorted image, so we still
        # use K3 in projectPoints.
        _newK, _roi = cv2.getOptimalNewCameraMatrix(K3, dist, (W, H), alpha=0)

        # points are already in camera frame → identity rvec, tvec
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)

        # OpenCV expects shape (N,1,3)
        pts_obj = pts_cam_front.reshape(-1, 1, 3)

        # Project with distortion onto the (distorted) image plane
        pts_img, _ = cv2.projectPoints(pts_obj, rvec, tvec, K3, dist)
        pts_uv = np.round(pts_img.reshape(-1, 2)).astype(np.int32)   # (M,2)

        u = pts_uv[:, 0]
        v = pts_uv[:, 1]

        # Keep points that fall inside the image
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_bounds):
            continue

        u = u[in_bounds]
        v = v[in_bounds]
        z_proj = z[in_bounds]  # same subset on depth along ray

        # depth from image
        z_true = depth[v, u]

        # only valid depth
        valid_depth = z_true > 0
        if not np.any(valid_depth):
            continue

        z_true = z_true[valid_depth]
        z_proj = z_proj[valid_depth]
        u_valid = u[valid_depth]
        v_valid = v[valid_depth]

        # error along camera ray (meters)
        err = z_true - z_proj
        all_err.append(err)

        # ==============================
        # VISUALIZE REPROJECTED POINTS  ### NEW
        # ==============================
        overlay = rgb.copy()  # RGB
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR

        # Draw each valid reprojected point as a small red dot
        for uu, vv in zip(u_valid, v_valid):
            if 0 <= uu < W and 0 <= vv < H:
                cv2.circle(overlay_bgr, (int(uu), int(vv)), 1, (0, 0, 255), -1)

        base_name = os.path.basename(rgb_path)
        overlay_name = f"reproj_cam{cam_id_str}_{frame_idx:04d}_{base_name}"
        overlay_path = os.path.join(reproj_dir, overlay_name)
        cv2.imwrite(overlay_path, overlay_bgr)
        # ==============================

    if not all_err:
        print("No valid samples collected.")
        return

    all_err = np.concatenate(all_err)

    print("\n===== Reprojection depth error (PLY vs depth, cams 1/2/3) =====")
    print("Num samples:", all_err.shape[0])
    print("Mean error   (m):", np.mean(all_err))
    print("Median error (m):", np.median(all_err))
    print("Std          (m):", np.std(all_err))
    p5  = np.percentile(all_err, 5)
    p95 = np.percentile(all_err, 95)
    print("5th pct (m):", p5, " 95th pct (m):", p95)


if __name__ == "__main__":
    main()

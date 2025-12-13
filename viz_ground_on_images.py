import os
import cv2
import torch
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf

from scene import load_cameras
from scene.dataset_readers import fetchPly

def project_points(points_world, K, c2w, width, height, step=50):
    """
    points_world: (N,3) numpy
    K: (3,3) numpy
    c2w: (4,4) numpy, world-from-camera
    returns uvs: (M,2) int, where M ~= N/step
    """

    # world -> camera: use inverse of c2w
    w2c = np.linalg.inv(c2w)    # (4,4)

    pts_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=1)  # (N,4)
    pts_cam = (w2c @ pts_h.T).T  # (N,4)
    x = pts_cam[:, 0]
    y = pts_cam[:, 1]
    z = pts_cam[:, 2]

    # keep only points in front of camera
    valid = z > 0
    x = x[valid]
    y = y[valid]
    z = z[valid]

    pts_cam_3 = np.stack([x, y, z], axis=1).T  # (3,N_valid)

    # camera -> image
    K3 = K[:3, :3]
    uvw = K3 @ pts_cam_3              # (3,N)
    u = uvw[0, :] / uvw[2, :]
    v = uvw[1, :] / uvw[2, :]

    u = u.astype(np.int32)
    v = v.astype(np.int32)

    # in-bounds filter
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    u = u[mask]
    v = v[mask]

    # subsample for display
    u = u[::step]
    v = v[::step]

    return np.stack([u, v], axis=1)  # (M,2)

def main():
    # --- configs (edit paths) ---
    base_cfg = "/workspace/Jack/HUGSIM/configs/waymo_gs_base.yaml"
    data_cfg = "/workspace/Jack/HUGSIM/configs/waymo.yaml"
    source_path = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"   # e.g. "/workspace/.../311238_part_0_100_v5"
    # source_path = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1083056"
    # source_path = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_new_v51"


    cfg = OmegaConf.merge(OmegaConf.load(base_cfg), OmegaConf.load(data_cfg))
    cfg.source_path = source_path

    # load cameras in the same way as training
    train_cams, test_cams, _ = load_cameras(cfg, cfg.data_type, True)

    # load ground point cloud
    # pcd = fetchPly(os.path.join(cfg.source_path, "ground_points3d_v10.ply"))
    pcd = fetchPly(os.path.join(cfg.source_path, "points3d_v3.ply"))
    pts = np.asarray(pcd.points, dtype=np.float32)

    # choose a few viewpoints to check
    # cam_indices = [0, 5, 10]  # change as needed
    cam_indices = range(len(train_cams))  # all indices

    # os.makedirs(os.path.join(cfg.source_path, "debug_ground_proj_v3"), exist_ok=True)
    #debug_ground_proj_v3 -4.7 rotations about x
    os.makedirs(os.path.join(cfg.source_path, "debug_points_proj_v3"), exist_ok=True)
    # debug_points_proj_v3 -4.7 degrees rotation about x

    for idx in cam_indices:
    # for idx in enumerate(train_cams):
        v = train_cams[idx]

        # get data from viewpoint
        img = v.original_image.permute(1,2,0).cpu().numpy()  # [H,W,3], 0-1
        img = (img * 255).astype(np.uint8).copy()

        H, W = img.shape[:2]
        K = v.K.cpu().numpy()[:3, :3]
        c2w = v.c2w.cpu().numpy()

        # project
        uvs = project_points(pts, K, c2w, W, H, step=50)

        # overlay red points
        for (u, v_) in uvs:
            cv2.circle(img, (int(u), int(v_)), 1, (0, 0, 255), -1)

        # out_path = os.path.join(cfg.source_path, "debug_ground_proj_v3", f"cam_{idx:03d}.png")
        out_path = os.path.join(cfg.source_path, "debug_points_proj_v3", f"cam_{idx:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("saved", out_path)

if __name__ == "__main__":
    main()


# import os
# import cv2
# import torch
# import numpy as np
# import open3d as o3d
# import json
# from omegaconf import OmegaConf

# from scene import load_cameras
# # from scene.dataset_readers import fetchPly  # not needed for the plane test
# # This will project a plane to check front cam height


# def project_plane_points(K, width, height, front_cam_height,
#                          x_min=-20.0, x_max=20.0,
#                          z_min=2.0,  z_max=60.0,
#                          num_x=80, num_z=80):
#     """
#     Build a grid of points lying on a plane at constant camera Y = front_cam_height
#     and project to image.

#     Assumes camera coords: X = right, Y = up, Z = forward.
#     Adjust front_cam_height or axes if your convention differs.
#     """

#     # Build grid in camera coordinates
#     xs = np.linspace(x_min, x_max, num_x)   # left/right
#     zs = np.linspace(z_min, z_max, num_z)   # forward
#     XX, ZZ = np.meshgrid(xs, zs)

#     YY = np.full_like(XX, front_cam_height)  # <-- PLANE: Y = front_cam_height
#     cam_pts = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3)  # (N,3)

#     # Project camera-frame points with intrinsics
#     cam_pts_T = cam_pts.T  # (3,N)
#     uvw = K @ cam_pts_T
#     u = uvw[0] / uvw[2]
#     v = uvw[1] / uvw[2]

#     u = u.astype(np.int32)
#     v = v.astype(np.int32)

#     # Keep only points that fall in the image
#     mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
#     u = u[mask]
#     v = v[mask]

#     return np.stack([u, v], axis=1)  # (M,2)


# def main():
#     # --- configs (edit paths) ---
#     base_cfg = "/workspace/Jack/HUGSIM/configs/waymo_gs_base.yaml"
#     data_cfg = "/workspace/Jack/HUGSIM/configs/waymo.yaml"
#     source_path = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"
#     # source_path = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1083056"


#     cfg = OmegaConf.merge(OmegaConf.load(base_cfg), OmegaConf.load(data_cfg))
#     cfg.source_path = source_path

#     # load cameras in the same way as training
#     train_cams, test_cams, _ = load_cameras(cfg, cfg.data_type, True)

#     # ---------------------------------------------------------
#     # Load front_cam_height (from the same place as merge script)
#     # ---------------------------------------------------------
#     # Option 1: from front_info.json (what your merge script uses)
#     front_info_path = os.path.join(source_path, "front_info.json")
#     with open(front_info_path, "r") as f:
#         front_info = json.load(f)
#     # front_cam_height = front_info["height"]

#     front_cam_height = 1.6

#     print("front_cam_height from front_info.json:", front_cam_height)

#     # (If you want to test -front_cam_height, just change here)
#     # front_cam_height = -front_cam_height
#     # ---------------------------------------------------------

#     out_dir = os.path.join(cfg.source_path, "debug_plane_proj_v2")
#     os.makedirs(out_dir, exist_ok=True)

#     # Helper: decide if this is a "front" camera (cam_1)
#     def is_front_cam(cam):
#         # Adjust this if your naming is different
#         name = getattr(cam, "image_name", "")
#         return "cam_1" in name

#     # You can use ONLY front cams, since height is defined for them
#     # If you want all cams, replace this loop with: for idx, v in enumerate(train_cams):
#     front_indices = [i for i, cam in enumerate(train_cams) if is_front_cam(cam)]
#     if not front_indices:
#         print("WARNING: No front cams found by name; falling back to all cameras.")
#         front_indices = range(len(train_cams))

#     for idx in front_indices:
#         v = train_cams[idx]

#         # get data from viewpoint
#         img = v.original_image.permute(1, 2, 0).cpu().numpy()  # [H,W,3], 0-1
#         img = (img * 255).astype(np.uint8).copy()
#         H, W = img.shape[:2]

#         # Robust K handling: allow either flat 9-vector or 3x3
#         K_raw = v.K.cpu().numpy()
#         if K_raw.ndim == 1:
#             assert K_raw.size >= 9, f"Unexpected K size {K_raw.size}"
#             K = K_raw[:9].reshape(3, 3)
#         else:
#             K = K_raw[:3, :3]

#         # Project a synthetic ground plane at Y = front_cam_height
#         uvs = project_plane_points(
#             K, W, H,
#             front_cam_height=front_cam_height,
#             x_min=-20.0, x_max=20.0,
#             z_min=2.0,  z_max=60.0,
#             num_x=80, num_z=80,
#         )

#         # overlay red points
#         for (u, v_) in uvs:
#             cv2.circle(img, (int(u), int(v_)), 1, (0, 0, 255), -1)

#         out_path = os.path.join(out_dir, f"cam_{idx:03d}.png")
#         cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#         print("saved", out_path)


# if __name__ == "__main__":
#     main()

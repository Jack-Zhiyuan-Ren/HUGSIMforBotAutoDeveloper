

#same as create_dynamic_mask2 but with regular meta_data
#no obj id and cams in verts and no obj_id and cams in dynamics
#only change is that w2c is np.eye(4) #This is removed for v4
#change from v3: Takes account of distortion
#change from v4: independent version
#change from v5: same but dependent version


import numpy as np
import json
import os
from imageio.v2 import imwrite
import cv2
import argparse

"""
Standalone create_dynamic_mask_custom3.py

Hardcoded for:
    seq_name = "311238_part_0_100_v7"
    basedir  = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v7"
    data_type = "waymo"

Uses downsampled intrinsics (by 2) + distortion for cam_1, cam_2, cam_3.
"""

# -----------------------------
# Hardcoded config
# -----------------------------
# SEQ_NAME = "311238_part_0_100_v7"
# BASEDIR = f"/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/{SEQ_NAME}"
# DATA_TYPE = "waymo"

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('--data_type', type=str, required=True)
    return parser.parse_args()

# Downsampled intrinsics + distortion (fx, fy, cx, cy, k1, k2, p1, p2, k3)
cam1_intr_ds = np.array(
    [[786.0027, 785.79835, 480.9809, 272.695, -0.317, 0.1197, 0.0, 0.0, 0.0]],
    dtype=float
)

# cam1_intr_ds = np.array(
#     [[786.0027, 785.79835, 480.9809, 272.695, 0, 0, 0.0, 0.0, 0.0]],
#     dtype=float
# )

cam2_intr_ds = np.array(
    [[786.0027, 785.79835, 480.9809, 272.695, -0.317, 0.1197, 0.0, 0.0, 0.0]],
    dtype=float
)

cam3_intr_ds = np.array(
    [[786.20225, 785.9637, 478.8244, 271.50045, -0.3165, 0.1199, 0.0, 0.0, 0.0]],
    dtype=float
)


def checkcorner(corner, h, w):
    """Unused helper from original script; kept for completeness."""
    if np.all(corner < 0) or (corner[0] >= h and corner[1] >= w):
        return False
    else:
        return True


def main():
    args = get_opts()
    basedir = args.data_path
    data_type = args.data_type   # <-- add this line

    os.makedirs(os.path.join(basedir, 'masks'), exist_ok=True)

    if data_type != 'waymo':
        raise NotImplementedError("This standalone version is hardcoded for data_type='waymo' only.")

    # Waymo cameras we use
    cameras = ['cam_1', 'cam_2', 'cam_3']
    for cam in cameras:
        os.makedirs(os.path.join(basedir, 'masks', cam), exist_ok=True)

    #mask_v1: includes distortion

    # Map camera name -> downsampled intrinsics vector
    cam_intr_map = {
        'cam_1': cam1_intr_ds[0],
        'cam_2': cam2_intr_ds[0],
        'cam_3': cam3_intr_ds[0],
    }

    # -----------------------------
    # Load meta_data.json
    # -----------------------------
    with open(os.path.join(basedir, "meta_data.json")) as f:
        meta_data = json.load(f)

    verts = meta_data['verts']

    # -----------------------------
    # Process each frame
    # -----------------------------
    for f in meta_data['frames']:
        rgb_path = f['rgb_path']       # e.g. "images/cam_2/000008.png"
        c2w = np.array(f['camtoworld'], dtype=float)

        # ---- choose intrinsics + distortion (from hardcoded downsampled vectors) ----
        # cam_name = rgb_path.split('/')[1]   # "cam_2"
        cam_name = os.path.basename(os.path.dirname(rgb_path))  # "cam_1", "cam_2", "cam_3"
        if cam_name not in cam_intr_map:
            print(f"[WARN] Camera {cam_name} not in cam_intr_map, skipping frame {rgb_path}")
            continue

        intr_vec = cam_intr_map[cam_name]  # [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = intr_vec

        # world -> camera
        w2c = np.linalg.inv(c2w)

        # semantics (downsampled resolution corresponding to intr_ds)
        smt_path = os.path.join(
            basedir,
            rgb_path.replace('images', 'semantics').replace('.jpg', '.npy').replace('.png', '.npy')
        )
        smt = np.load(smt_path)

        # ``car'', ``truck'', ``bus'', ``train'', ``motorcycle'', ``bicycle''
        car_mask = (
            (smt == 11) | (smt == 12) | (smt == 13) |
            (smt == 14) | (smt == 15) | (smt == 18)
        )
        mask = np.zeros_like(car_mask).astype(np.bool_)

        H, W = mask.shape[0], mask.shape[1]

        for iid, rt in f['dynamics'].items():
            rt = np.array(rt, dtype=float)
            points = np.array(verts[iid], dtype=float)   # (8,3)

            # world -> world box corners
            points_w = (rt[:3, :3] @ points.T).T + rt[:3, 3]

            # world -> camera
            xyz_cam = (w2c[:3, :3] @ points_w.T).T + w2c[:3, 3]

            # Depth validity
            Z = xyz_cam[:, 2]
            valid_depth = Z > 0
            if not valid_depth.any():
                continue

            # ---- normalized coordinates (only where depth > 0 to avoid /0) ----
            X = np.zeros_like(Z)
            Y = np.zeros_like(Z)
            X[valid_depth] = xyz_cam[valid_depth, 0] / Z[valid_depth]
            Y[valid_depth] = xyz_cam[valid_depth, 1] / Z[valid_depth]

            # radial distortion
            r2 = X**2 + Y**2
            r4 = r2**2
            r6 = r2**3

            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

            # tangential distortion (p1, p2 are 0 in your intrinsics, but we keep formula)
            x_tangential = 2 * p1 * X * Y + p2 * (r2 + 2 * X**2)
            y_tangential = p1 * (r2 + 2 * Y**2) + 2 * p2 * X * Y

            Xd = X * radial + x_tangential
            Yd = Y * radial + y_tangential

            u = fx * Xd + cx
            v = fy * Yd + cy

            xy_screen = np.stack([u, v], axis=-1)  # (8,2)

            valid_x = (xy_screen[:, 0] >= 0) & (xy_screen[:, 0] < W)
            valid_y = (xy_screen[:, 1] >= 0) & (xy_screen[:, 1] < H)
            valid_pixel = valid_x & valid_y & valid_depth

            if valid_pixel.any():
                xy_int = np.round(xy_screen).astype(int)
                bbox_mask = np.zeros((H, W), dtype=np.uint8)

                # Same 6 faces as original
                cv2.fillPoly(bbox_mask, [xy_int[[0, 1, 4, 5, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_int[[2, 3, 6, 7, 2]]], 1)
                cv2.fillPoly(bbox_mask, [xy_int[[0, 2, 7, 5, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_int[[1, 3, 6, 4, 1]]], 1)
                cv2.fillPoly(bbox_mask, [xy_int[[0, 2, 3, 1, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_int[[5, 4, 6, 7, 5]]], 1)

                overlap_pixels = np.logical_and(bbox_mask != 0, car_mask).sum()
                print(
                    "frame", f["rgb_path"], "id", iid,
                    "| box area:", int((bbox_mask != 0).sum()),
                    "| car_mask:", int(car_mask.sum()),
                    "| overlap:", int(overlap_pixels)
                )

                bbox_mask = bbox_mask & car_mask
                mask = mask | (bbox_mask != 0)

        # save inverse mask (~mask) as npy + png
        save_path = os.path.join(basedir, rgb_path.replace('images', 'masks'))
        np.save(
            save_path.replace('.jpg', '.npy').replace('.png', '.npy'),
            ~mask
        )
        imwrite(save_path + '.png', (~mask).astype(np.uint8) * 255)


if __name__ == "__main__":
    main()

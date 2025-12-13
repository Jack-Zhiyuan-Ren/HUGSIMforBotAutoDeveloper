import numpy as np
import json
import os
from imageio.v2 import imread, imwrite
import argparse
import cv2

#same as create_dynamic_mask2 but with regular meta_data
#no obj id and cams in verts and no obj_id and cams in dynamics
#only change is that w2c is np.eye(4) #This is removed for v4
#change from v3: Takes account of distortion

cam1_intr_ds = np.array(
    [[786.0027, 785.79835, 480.9809, 272.695, -0.317, 0.1197, 0.0, 0.0, 0.0]],
    dtype=float
)

cam2_intr_ds = np.array(
    [[786.0027, 785.79835, 480.9809, 272.695, -0.317, 0.1197, 0.0, 0.0, 0.0]],
    dtype=float
)

cam3_intr_ds = np.array(
    [[786.20225, 785.9637, 478.8244, 271.50045, -0.3165, 0.1199, 0.0, 0.0, 0.0]],
    dtype=float
)



def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('--data_type', type=str, required=True)
    return parser.parse_args()

#Intended to cull out-of-image points, but it’s never called.
def checkcorner(corner, h, w):
    if np.all(corner < 0) or (corner[0] >= h and corner[1] >= w):
        return False
    else:
        return True

def main():
    args = get_opts()
    basedir = args.data_path
    os.makedirs(os.path.join(basedir, 'masks'), exist_ok=True)
    if args.data_type == 'kitti360':
        cameras = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
    elif args.data_type == 'pandaset':
        AVAILABLE_CAMERAS = ("front", "front_left", "front_right", "back", "left", "right")
        cameras = [cam + "_camera" for cam in AVAILABLE_CAMERAS]
    elif args.data_type == 'nuscenes':
        cameras = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
                             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
    elif args.data_type == 'waymo':
        cameras = ['cam_1', 'cam_2', 'cam_3']
    else:
        raise NotImplementedError
    for cam in cameras:
        os.makedirs(os.path.join(basedir, 'masks', cam), exist_ok=True)

        # For Waymo, map camera name -> downsampled intrinsics vector
    cam_intr_map = None
    if args.data_type == 'waymo':
        cam_intr_map = {
            'cam_1': cam1_intr_ds[0],
            'cam_2': cam2_intr_ds[0],
            'cam_3': cam3_intr_ds[0],
        }


    # Opening JSON file
    with open(os.path.join(basedir, "meta_data.json")) as f:
        meta_data = json.load(f)

    verts = meta_data['verts']
    for f in meta_data['frames']:
        rgb_path = f['rgb_path']
        c2w = np.array(f['camtoworld'])
        # -------- choose intrinsics + distortion --------
        if args.data_type == 'waymo':
            # e.g. "images/cam_2/000008.png" -> "cam_2"
            cam_name = rgb_path.split('/')[1]
            intr_vec = cam_intr_map[cam_name]  # [fx, fy, cx, cy, k1, k2, p1, p2, k3]
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = intr_vec
        else:
            # fallback: use intrinsics from meta_data without distortion
            intr = np.array(f['intrinsics'], dtype=float)
            # assume intr is at least 3x3
            fx = intr[0, 0]
            fy = intr[1, 1]
            cx = intr[0, 2]
            cy = intr[1, 2]
            k1 = k2 = p1 = p2 = k3 = 0.0

        w2c = np.linalg.inv(c2w)



        smt = np.load(os.path.join(basedir, rgb_path.replace('images', 'semantics').replace('.jpg', '.npy')).replace('.png', '.npy'))
        car_mask = (smt == 11) | (smt == 12) | (smt == 13) | (smt == 14) | (smt == 15) | (smt == 18) # ``car'', ``truck'', ``bus'', ``train'', ``motorcycle'', ``bicycle'' 
        mask = np.zeros_like(car_mask).astype(np.bool_)

        for iid, rt in f['dynamics'].items():
            H, W = mask.shape[0], mask.shape[1]
            rt = np.array(rt)
            points = np.array(verts[iid])
            points = (rt[:3, :3] @ points.T).T + rt[:3, 3]
            xyz_cam = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
            # valid_depth = xyz_cam[:, 2] > 0



            # ---------- project with radial distortion ----------
            Z = xyz_cam[:, 2]
            valid_depth = Z > 0
            X = xyz_cam[:, 0] / Z
            Y = xyz_cam[:, 1] / Z

            r2 = X**2 + Y**2
            r4 = r2**2
            r6 = r2**3

            radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

            x_tangential = 2 * p1 * X * Y + p2 * (r2 + 2 * X**2)
            y_tangential = p1 * (r2 + 2 * Y**2) + 2 * p2 * X * Y

            Xd = X * radial + x_tangential
            Yd = Y * radial + y_tangential

            u = fx * Xd + cx
            v = fy * Yd + cy



            # xyz_screen = (intr[:3, :3] @ xyz_cam.T).T + intr[:3, 3]
            # xy_screen  = xyz_screen[:, :2] / xyz_screen[:, 2][:, None]
            xy_screen = np.stack([u, v], axis=-1)  # (8,2)
            ##########################################################

            valid_x = (xy_screen[:, 0] >= 0) & (xy_screen[:, 0] < W)
            valid_y = (xy_screen[:, 1] >= 0) & (xy_screen[:, 1] < H)
            valid_pixel = valid_x & valid_y & valid_depth

            if valid_pixel.any():
                xy_screen = np.round(xy_screen).astype(int)
                bbox_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(bbox_mask, [xy_screen[[0, 1, 4, 5, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen[[2, 3, 6, 7, 2]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen[[0, 2, 7, 5, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen[[1, 3, 6, 4, 1]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen[[0, 2, 3, 1, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen[[5, 4, 6, 7, 5]]], 1)

                overlap_pixels = np.logical_and(bbox_mask != 0, car_mask).sum()
                print("frame", f["rgb_path"], "id", iid,
                        "| box area:", int((bbox_mask != 0).sum()),
                        "| car_mask:", int(car_mask.sum()),
                        "| overlap:", int(overlap_pixels))

                bbox_mask = bbox_mask & car_mask
                mask = mask | (bbox_mask != 0)

        save_path = os.path.join(basedir, rgb_path.replace('images', 'masks'))
        np.save(save_path.replace('.jpg', '.npy').replace('.png', '.npy'), ~mask)
        imwrite(save_path+'.png', (~mask).astype(np.uint8) * 255)

if __name__ == "__main__":
    main()
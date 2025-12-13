# import numpy as np
# import json
# import os
# from imageio.v2 import imread, imwrite
# import argparse
# import cv2

# # This version accomdates for the lastest version meta_data that includes obj.id 
# # This is suitable for the custom data.

# def get_opts():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--data_path', type=str, required=True)
#     parser.add_argument('--data_type', type=str, required=True)
#     return parser.parse_args()

# #Intended to cull out-of-image points, but it’s never called.
# def checkcorner(corner, h, w):
#     if np.all(corner < 0) or (corner[0] >= h and corner[1] >= w):
#         return False
#     else:
#         return True

# def main():
#     args = get_opts()
#     basedir = args.data_path
#     os.makedirs(os.path.join(basedir, 'masks'), exist_ok=True)
#     if args.data_type == 'kitti360':
#         cameras = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
#     elif args.data_type == 'pandaset':
#         AVAILABLE_CAMERAS = ("front", "front_left", "front_right", "back", "left", "right")
#         cameras = [cam + "_camera" for cam in AVAILABLE_CAMERAS]
#     elif args.data_type == 'nuscenes':
#         cameras = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
#                              "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
#     elif args.data_type == 'waymo':
#         cameras = ['cam_1', 'cam_2', 'cam_3']
#     else:
#         raise NotImplementedError
#     for cam in cameras:
#         os.makedirs(os.path.join(basedir, 'masks', cam), exist_ok=True)
#     # Opening JSON file
#     with open(os.path.join(basedir, "meta_data.json")) as f:
#         meta_data = json.load(f)

#     verts = meta_data['verts']


#     for f in meta_data['frames']:
#         rgb_path = f['rgb_path']
#         c2w = np.array(f['camtoworld'])
#         intr = np.array(f['intrinsics'])
#         # w2c = np.linalg.inv(c2w)

#         #new
#         w2c = np.eye(4)

#         smt = np.load(os.path.join(basedir, rgb_path.replace('images', 'semantics').replace('.jpg', '.npy')).replace('.png', '.npy'))
#         car_mask = (smt == 11) | (smt == 12) | (smt == 13) | (smt == 14) | (smt == 15) | (smt == 18) # ``car'', ``truck'', ``bus'', ``train'', ``motorcycle'', ``bicycle'' 
#         mask = np.zeros_like(car_mask).astype(np.bool_)

#         for iid, rt in f['dynamics'].items():
#             H, W = mask.shape[0], mask.shape[1]
#             rt = np.array(dyn['rt'], dtype=float)  # shape (4, 4)

#             #points = np.array(verts[iid])
#             points = np.array(verts[iid]['verts'], dtype=float)  # (8, 3) #new

#             points = (rt[:3, :3] @ points.T).T + rt[:3, 3]
#             xyz_cam = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
#             valid_depth = xyz_cam[:, 2] > 0
#             xyz_screen = (intr[:3, :3] @ xyz_cam.T).T + intr[:3, 3]
#             xy_screen  = xyz_screen[:, :2] / xyz_screen[:, 2][:, None]
#             valid_x = (xy_screen[:, 0] >= 0) & (xy_screen[:, 0] < W)
#             valid_y = (xy_screen[:, 1] >= 0) & (xy_screen[:, 1] < H)
#             valid_pixel = valid_x & valid_y & valid_depth

#             if valid_pixel.any():
#                 xy_screen = np.round(xy_screen).astype(int)
#                 bbox_mask = np.zeros((H, W), dtype=np.uint8)
#                 cv2.fillPoly(bbox_mask, [xy_screen[[0, 1, 4, 5, 0]]], 1)
#                 cv2.fillPoly(bbox_mask, [xy_screen[[2, 3, 6, 7, 2]]], 1)
#                 cv2.fillPoly(bbox_mask, [xy_screen[[0, 2, 7, 5, 0]]], 1)
#                 cv2.fillPoly(bbox_mask, [xy_screen[[1, 3, 6, 4, 1]]], 1)
#                 cv2.fillPoly(bbox_mask, [xy_screen[[0, 2, 3, 1, 0]]], 1)
#                 cv2.fillPoly(bbox_mask, [xy_screen[[5, 4, 6, 7, 5]]], 1)

#                 overlap_pixels = np.logical_and(bbox_mask != 0, car_mask).sum()
#                 print("frame", f["rgb_path"], "id", iid,
#                         "| box area:", int((bbox_mask != 0).sum()),
#                         "| car_mask:", int(car_mask.sum()),
#                         "| overlap:", int(overlap_pixels))

#                 bbox_mask = bbox_mask & car_mask
#                 mask = mask | (bbox_mask != 0)

#         save_path = os.path.join(basedir, rgb_path.replace('images', 'masks'))
#         np.save(save_path.replace('.jpg', '.npy').replace('.png', '.npy'), ~mask)
#         imwrite(save_path+'.png', (~mask).astype(np.uint8) * 255)

# if __name__ == "__main__":
#     main()

import numpy as np
import json
import os
from imageio.v2 import imread, imwrite
import argparse
import cv2

# This version accommodates the latest version of meta_data that includes obj_id and nested verts.
# Suitable for the custom data (e.g., meta_data_311238_part_0_new_v48.json).

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('--data_type', type=str, required=True)
    # NEW: allow specifying the meta file name
    parser.add_argument('--meta_name', type=str, default='meta_data.json')
    return parser.parse_args()

# Intended to cull out-of-image points, but it’s never called.
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

    # --- Load new-format meta_data ---
    meta_path = os.path.join(basedir, args.meta_name)
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)

    # In new meta_data:
    # meta_data['frames'] is a list of per-image dicts
    # meta_data['verts'] is a dict: { "1": { "verts": [...8x3...], "obj_id": ..., "cams": [...] }, ... }
    verts = meta_data['verts']

    for f_info in meta_data['frames']:
        rgb_path = f_info['rgb_path']
        c2w = np.array(f_info['camtoworld'], dtype=float)
        intr = np.array(f_info['intrinsics'], dtype=float)
        w2c = np.linalg.inv(c2w)
        ################
        w2c = np.eye(4)
        ##################

        # load semantic map
        smt_path = os.path.join(
            basedir,
            rgb_path.replace('images', 'semantics').replace('.jpg', '.npy').replace('.png', '.npy')
        )
        smt = np.load(smt_path)

        # ``car'', ``truck'', ``bus'', ``train'', ``motorcycle'', ``bicycle''
        car_mask = (
            (smt == 11) |
            (smt == 12) |
            (smt == 13) |
            (smt == 14) |
            (smt == 15) |
            (smt == 18)
        )
        mask = np.zeros_like(car_mask, dtype=np.bool_)

        # NEW: dynamics entries are dicts with keys: "rt", "obj_id", "cams"
        for iid, dyn in f_info['dynamics'].items():
            H, W = mask.shape

            # 4x4 transform matrix
            rt = np.array(dyn['rt'], dtype=float)  # shape (4, 4)

            # verts entry is now nested: verts[iid]["verts"]
            if iid not in verts:
                # If for some reason this id has no verts, skip
                continue
            points_local = np.array(verts[iid]['verts'], dtype=float)  # (8, 3)

            # world coordinates: apply object transform
            points_w = (rt[:3, :3] @ points_local.T).T + rt[:3, 3]

            # camera coordinates
            xyz_cam = (w2c[:3, :3] @ points_w.T).T + w2c[:3, 3]

            # depth positive
            valid_depth = xyz_cam[:, 2] > 0

            # project to screen using 4x4 intrinsics
            xyz_screen = (intr[:3, :3] @ xyz_cam.T).T + intr[:3, 3]
            xy_screen = xyz_screen[:, :2] / xyz_screen[:, 2][:, None]

            valid_x = (xy_screen[:, 0] >= 0) & (xy_screen[:, 0] < W)
            valid_y = (xy_screen[:, 1] >= 0) & (xy_screen[:, 1] < H)
            valid_pixel = valid_x & valid_y & valid_depth

            if valid_pixel.any():
                xy_screen_int = np.round(xy_screen).astype(int)

                bbox_mask = np.zeros((H, W), dtype=np.uint8)

                # faces of the box using 8 verts (same indexing as before)
                cv2.fillPoly(bbox_mask, [xy_screen_int[[0, 1, 4, 5, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen_int[[2, 3, 6, 7, 2]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen_int[[0, 2, 7, 5, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen_int[[1, 3, 6, 4, 1]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen_int[[0, 2, 3, 1, 0]]], 1)
                cv2.fillPoly(bbox_mask, [xy_screen_int[[5, 4, 6, 7, 5]]], 1)

                overlap_pixels = np.logical_and(bbox_mask != 0, car_mask).sum()
                print(
                    "frame", f_info["rgb_path"], "id", iid,
                    "| box area:", int((bbox_mask != 0).sum()),
                    "| car_mask:", int(car_mask.sum()),
                    "| overlap:", int(overlap_pixels)
                )

                # restrict box to semantic car pixels
                bbox_mask = bbox_mask & car_mask
                mask = mask | (bbox_mask != 0)

        save_path = os.path.join(basedir, rgb_path.replace('images', 'masks'))
        np.save(
            save_path.replace('.jpg', '.npy').replace('.png', '.npy'),
            (~mask)
        )
        imwrite(save_path + '.png', (~mask).astype(np.uint8) * 255)

if __name__ == "__main__":
    main()

# import os
# import json
# import numpy as np
# from tqdm import tqdm

# # ===================== HARD-CODED PATH =====================
# # Change this to the folder that contains your original meta_data.json
# OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v5"   # <--- EDIT THIS

# META_IN  = os.path.join(OUT_DIR, "meta_data.json")
# META_OUT = os.path.join(OUT_DIR, "meta_data_modded.json")
# # ===========================================================

# front_rect_mat = np.array([
#     [1.0,  0.0,  0.0],       # only use this for translation parts
#     [0.0,  0.0,  1.0],
#     [0.0, -1.0,  0.0],
# ], dtype=float)

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


# def main():
#     if not os.path.exists(META_IN):
#         raise FileNotFoundError(f"Cannot find {META_IN}")

#     with open(META_IN, "r") as rf:
#         meta_data = json.load(rf)

#     frames = meta_data.get("frames", [])
#     print(f"Loaded {len(frames)} frames from {META_IN}")

#     for frame in tqdm(frames, desc="Modifying camtoworld"):
#         c2w = np.array(frame["camtoworld"], dtype=float)

#         # Extract old rotation and translation
#         R_old = c2w[:3, :3]
#         t_old = c2w[:3, 3]

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

#         # write back
#         c2w[:3, :3] = R_new
#         c2w[:3, 3] = t_new
#         frame["camtoworld"] = c2w.tolist()

#     # Save to new meta_data file (meta_data_modded.json)
#     with open(META_OUT, "w") as wf:
#         json.dump(meta_data, wf, indent=2)

#     print(f"Saved modified metadata to {META_OUT}")


# if __name__ == "__main__":
#     main()


import os
import json
import numpy as np
from tqdm import tqdm

# ===================== HARD-CODED PATH =====================
# Change this to the folder that contains your original meta_data.json
OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v7"  # <--- EDIT THIS

META_IN  = os.path.join(OUT_DIR, "meta_data.json")
META_OUT = os.path.join(OUT_DIR, "meta_data_v1.json")


########### 311238_part_0_100_v6 #################
# in meta_data_v1, cam2 and cam3 are about 34 degrees roatation from -y and y. The translation is subtraction from cam1f0
# in meta_data_v2, cam1 rotates 45 degrees along -x
#in meta_data_v3, cam1 rotates 10 degrees along -x
#in meta_data_v3, cam1 rotates 1 degrees along -x
#in meta_data_v4, cam1 rotates 5 degrees along -x
#in meta_data_v5, cam1 rotates 4 degrees along -x
#in meta_data_v6, cam1 rotates 4.9 degrees along -x
#in meta_data_v7, cam1 rotates 4.8 degrees along -x
#in meta_data_v8, cam1 rotates 4.7 degrees along -x
#in meta_data_v9, cam1 rotates 2 degrees along -x
#in meta_data_v10, cam1 rotates 1.8 degrees along -x
#in meta_data_v11, cam1 rotates 1.5 degrees along -x
#in meta_data_v12, cam1 rotates 1.6 degrees along -x
#in meta_data_v13, cam1 rotates 1 degrees along x
#in meta_data_v14, there is y offsets. So the 98th frame y value is reduced from 92 to 60. no extra rotation. 
#in meta_data_v15, there is z offsets. So the 98th frame z value is reduced from 92 to 60. -1 degrees rotation along x axis.
#in meta_data_v16, there is z offsets. every frame + 10 z offset. no rotation change.
#in meta_data_v17, there is z offsets. every frame -10 z offset. no rotation change.
#in meta_data_v18, there is z offsets. every frame -10 z offset. 98th frame z value is reduced from 92 to 82. no rotation change.
#in meta_data_v19, cam1 rotates -20 degrees along x axis
#in meta_data_v20, cam1 rotates -15 degrees along x axis
#in meta_data_v21, cam1 rotates -15 degrees along x axis, 98th frame z value is reduced from 92 to 82.
# in meta_data_v22, cam1 rotates -12 degrees along x axis, 98th frame z value is reduced from 92 to 82. 
# in meta_data_v23, cam1 rotates -14 degrees along x axis, 98th frame z value is reduced from 92 to 82. 
# in meta_data_v24, cam1 rotates -13.8 degrees along x axis, 98th frame z value is reduced from 92 to 82 with linear interpolation for other frames.
# in meta_data_v25, cam1 rotates -21 degrees along x axis, 98th frame z value is reduced from 92 to 82. 
# in meta_data_v26, cam1 rotates -23 degrees along x axis, 98th frame z value is reduced from 92 to 82
# in meta_data_v27, cam1 rotates -45 degrees along x axis, 98th frame z value is reduced from 92 to 82 
# in meta_data_v28, cam1 rotates -40 degrees along x axis 98th frame z value is reduced from 92 to 82 
# in meta_data_v29, cam1 rotates -50 degrees along x axiis 98th frame z value is reduced from 92 to 82
# in meta_data_v30. No rotation change. 98th frame z value is reduced from 92 to 82.
# in meta_data_v31. cam1 rotates -4.8 about x. 98th frame z value is reduced from 92 to 82.
##############################################################################

########### 311238_part_0_100_v7 #################
# in meta_data_v1, the translation is changed to subtraction. 



# ===========================================================

# Fixed relative transforms (inv(extr1) @ extr2/3) that provide rotations for cam2 and cam3

T_x_neg45 = np.array([
    [1.0, 0.0,        0.0,        0.0],
    [0.0, 0.70710678, 0.70710678, 0.0],
    [0.0,-0.70710678, 0.70710678, 0.0],
    [0.0, 0.0,        0.0,        1.0],
], dtype=float)


T_x_neg10 = np.array([
    [1.0,         0.0,         0.0,        0.0],
    [0.0,  0.98480775,  0.17364818,  0.0],
    [0.0, -0.17364818,  0.98480775,  0.0],
    [0.0,         0.0,         0.0,        1.0],
], dtype=float)

T_x_neg1 = np.array([
    [1.0,         0.0,         0.0,        0.0],
    [0.0,  0.99984770,  0.01745241,  0.0],
    [0.0, -0.01745241,  0.99984770,  0.0],
    [0.0,         0.0,         0.0,        1.0],
], dtype=float)

T_x_neg5 = np.array([
    [1.0,         0.0,         0.0,        0.0],
    [0.0,  0.99619470,  0.08715574,  0.0],
    [0.0, -0.08715574,  0.99619470,  0.0],
    [0.0,         0.0,         0.0,        1.0],
], dtype=float)

T_x_neg4 = np.array([
    [1.0,         0.0,         0.0,        0.0],
    [0.0,  0.99756405,  0.06975647,  0.0],
    [0.0, -0.06975647,  0.99756405,  0.0],
    [0.0,         0.0,         0.0,        1.0],
], dtype=float)

T_x_neg4_9 = np.array([
    [1.0,               0.0,               0.0,             0.0],
    [0.0,  0.9963452961909064,  0.08541692313736747,  0.0],
    [0.0, -0.08541692313736747, 0.9963452961909064,  0.0],
    [0.0,               0.0,               0.0,             1.0],
], dtype=float)

T_x_neg4_8 = np.array([
    [1.0,                  0.0,                  0.0, 0.0],
    [0.0,  0.9964928592495044,  0.08367784333231548, 0.0],
    [0.0, -0.08367784333231548, 0.9964928592495044, 0.0],
    [0.0,                  0.0,                  0.0, 1.0],
], dtype=float)

T_x_neg4_7 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.9966373868180366,  0.08193850863004093, 0.0],
    [0.0, -0.08193850863004093, 0.9966373868180366, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)

T_x_neg2 = np.array([
[1.0,               0.0,               0.0, 0.0],
[0.0,  0.9993908270,  0.0348994967,    0.0],
[0.0, -0.0348994967,  0.9993908270,    0.0],
[0.0,               0.0,               0.0, 1.0],
], dtype=float)

T_x_neg1_8 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.9995065603657316,  0.03141075907812829, 0.0],
    [0.0, -0.03141075907812829, 0.9995065603657316, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)

T_x_neg1_5 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.9996573249755573,  0.026176948307873153, 0.0],
    [0.0, -0.026176948307873153, 0.9996573249755573, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)

T_x_neg1_6 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.9996101150403544,  0.02792163872356888, 0.0],
    [0.0, -0.02792163872356888, 0.9996101150403544, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)

T_x_pos1 = np.array([
    [1.0,                 0.0,                  0.0, 0.0],
    [0.0,  0.9998476951563913, -0.01745240643728351, 0.0],
    [0.0,  0.01745240643728351,  0.9998476951563913, 0.0],
    [0.0,                 0.0,                  0.0, 1.0],
], dtype=float)

T_x_neg20 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.9396926207859084,  0.3420201433256687, 0.0],
    [0.0, -0.3420201433256687,  0.9396926207859084, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)

T_x_neg15 = np.array([
    [1.0,                 0.0,                  0.0, 0.0],
    [0.0,  0.9659258262890683,  0.25881904510252074, 0.0],
    [0.0, -0.25881904510252074, 0.9659258262890683, 0.0],
    [0.0,                 0.0,                  0.0, 1.0],
], dtype=float)

T_x_neg12 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.9781476007338057,  0.20791169081775934, 0.0],
    [0.0, -0.20791169081775934, 0.9781476007338057, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)

T_x_neg14 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.9702957262759965,  0.24192189559966773, 0.0],
    [0.0, -0.24192189559966773, 0.9702957262759965, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)

T_x_neg13_8 = np.array([
    [1.0,                 0.0,                  0.0, 0.0],
    [0.0,  0.9711342799096361,  0.23853345757858088, 0.0],
    [0.0, -0.23853345757858088, 0.9711342799096361, 0.0],
    [0.0,                 0.0,                  0.0, 1.0],
], dtype=float)

T_x_neg21 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.9335804264972017,  0.35836794954530027, 0.0],
    [0.0, -0.35836794954530027, 0.9335804264972017, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)

T_x_neg23 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.9205048534524404,  0.39073112848927377, 0.0],
    [0.0, -0.39073112848927377, 0.9205048534524404, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)

T_x_neg40 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.766044443118978,  0.6427876096865393, 0.0],
    [0.0, -0.6427876096865393, 0.766044443118978, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)

T_x_neg50 = np.array([
    [1.0,                 0.0,                 0.0, 0.0],
    [0.0,  0.6427876096865394,  0.766044443118978, 0.0],
    [0.0, -0.766044443118978,   0.6427876096865394, 0.0],
    [0.0,                 0.0,                 0.0, 1.0],
], dtype=float)





T_cam2 = np.array([
    [ 8.20229579e-01,  7.39009272e-02, -5.67241271e-01, -1.01644863e+00],
    [-6.17934843e-02,  9.97263465e-01,  4.05709441e-02, -5.27281283e-04],
    [ 5.68687551e-01,  1.77438084e-03,  8.22551074e-01, -3.04127474e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
], dtype=float)

T_cam3 = np.array([
    [0.8208416,  -0.06344476,  0.56762227,  1.74325742],
    [0.03928235,  0.99772861,  0.05471358, -0.04687121],
    [-0.56980431, -0.02261248, 0.82146882, -0.02516175],
    [0.0,         0.0,         0.0,         1.0]
], dtype=float)

R_cam2 = T_cam2[:3, :3]
R_cam3 = T_cam3[:3, :3]
R_cam1 = T_x_neg4_8[:3,:3]


def get_cam_id_from_path(rgb_path: str):
    """Return 1/2/3 depending on which cam appears in the rgb_path."""
    if "/cam_1/" in rgb_path:
        return 1
    elif "/cam_2/" in rgb_path:
        return 2
    elif "/cam_3/" in rgb_path:
        return 3
    else:
        return None


def parse_frame_index_from_path(rgb_path: str):
    """
    Extract the integer frame index from a path like './images/cam_1/000007.png'.
    Returns None if it can't parse.
    """
    base = os.path.basename(rgb_path)  # '000007.png'
    stem, _ = os.path.splitext(base)   # '000007'
    try:
        return int(stem)
    except ValueError:
        return None


def find_reference_ego_pose(frames):
    """
    Prefer: exact cam_1 frame 0 => './images/cam_1/000000.png' (or any path ending in 'cam_1/000000.png').
    Fallback: smallest cam_1 frame index.
    """
    ego_key = None
    ref_ego = None

    # -------- First pass: try to find exact cam_1 frame 0 (000000.png) --------
    for fr in frames:
        rgb_path = fr.get("rgb_path", "")
        if "/cam_1/" not in rgb_path:
            continue

        # figure out which key the ego pose lives under
        if ego_key is None:
            if "ego_pose_image" in fr:
                ego_key = "ego_pose_image"
            else:
                continue  # this frame doesn't have ego pose info, skip

        # check if this is exactly frame 0 by filename
        base = os.path.basename(rgb_path)  # e.g. '000000.png'
        stem, _ = os.path.splitext(base)
        if stem == "000000":
            ref_ego = np.array(fr[ego_key], dtype=float)
            print(f"Reference frame found: cam_1 frame 0 ({rgb_path})")
            return ref_ego, ego_key

    # -------- Second pass: fallback to smallest cam_1 frame index --------
    best_idx = None
    best_ego = None

    for fr in frames:
        rgb_path = fr.get("rgb_path", "")
        if "/cam_1/" not in rgb_path:
            continue

        # determine ego key if still None
        if ego_key is None:
            if "ego_pose_image" in fr:
                ego_key = "ego_pose_frame"
            else:
                continue

        idx = parse_frame_index_from_path(rgb_path)
        if idx is None:
            continue

        if best_idx is None or idx < best_idx:
            best_idx = idx
            best_ego = np.array(fr[ego_key], dtype=float)

    if best_ego is None:
        raise RuntimeError("Could not find any reference frame for cam_1 (no valid ego pose).")

    print(f"No explicit cam_1 frame 0; using smallest cam_1 frame index = {best_idx}")
    return best_ego, ego_key


def main():
    if not os.path.exists(META_IN):
        raise FileNotFoundError(f"Cannot find {META_IN}")

    with open(META_IN, "r") as rf:
        meta_data = json.load(rf)

    frames = meta_data.get("frames", [])
    print(f"Loaded {len(frames)} frames from {META_IN}")

    # ---- Find reference ego pose at cam_1 frame 0 (or fallback smallest index) ----
    ref_ego, ego_key = find_reference_ego_pose(frames)
    print(f"Using ego pose key '{ego_key}' for translation computation.")

    # ---- Modify camtoworld for each frame ----
    for idx, frame in enumerate(tqdm(frames, desc="Modifying camtoworld")):
        c2w = np.array(frame["camtoworld"], dtype=float)

        # original rotation (for cam_1 we keep this)
        R_old = c2w[:3, :3]

        rgb_path = frame["rgb_path"]
        cam_id = get_cam_id_from_path(rgb_path)

        # ---------- ROTATION ----------
        if cam_id == 1:
            R_new = R_old  # cam_1: keep original rotation
        elif cam_id == 2:
            R_new = R_old  # cam_2: use fixed rotation from T_cam2
        elif cam_id == 3:
            R_new = R_old  # cam_3: use fixed rotation from T_cam3
        else:
            # unknown cam: leave unchanged
            R_new = R_old

        # ---------- TRANSLATION ----------
        ego_curr = np.array(frame[ego_key], dtype=float)

        # x position (0,3): ego_pose_frame[1,3] at cam_1 frame 0
        #                   minus ego_pose_frame[1,3] at current cam and frame
        x_new = ref_ego[1, 3] - ego_curr[1, 3]

        # y position (1,3): ego_pose_frame[2,3] at current frame
        #                   minus ego_pose_frame[2,3] at cam_1 frame 0

       
            # linear offset: 0 at idx=0, 32 at idx=98

        y_new = (ego_curr[2, 3] - ref_ego[2, 3])
        
        # z position (2,3): ego_pose_frame[0,3] at current frame
        #                   minus ego_pose_frame[0,3] at cam_1 frame 0

         ########### Making offsets for z ##############
        target_reduction_at_last = 92.0 - 82.0   # = 32
        last_idx = 98.0                          # 98th frame (index 98)
        z_offset = target_reduction_at_last * (idx / last_idx) 
        z_new = (ego_curr[0, 3] - ref_ego[0, 3]) #- z_offset
               # how much we want to reduce y_new at the last frame

        #################################################
        t_new = np.array([x_new, y_new, z_new], dtype=float)

        # write back
        c2w[:3, :3] = R_new
        c2w[:3, 3]  = t_new
        frame["camtoworld"] = c2w.tolist()

    # Save to new meta_data file (meta_data_modded.json)
    with open(META_OUT, "w") as wf:
        json.dump(meta_data, wf, indent=2)

    print(f"Saved modified metadata to {META_OUT}")


if __name__ == "__main__":
    main()

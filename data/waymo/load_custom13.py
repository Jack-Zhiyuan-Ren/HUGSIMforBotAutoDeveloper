

import os
import open3d as o3d
import numpy as np
from tqdm import tqdm
import json
import cv2
from scipy.spatial.transform import Rotation as SCR
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils

# Plan to implement the most basic form: Only using camera extr and radar readings
# same as load_custom7 but for the full pipeline.
# same as load_custom8 but no obj id and cams in verts and no obj_id and cams in dynamics
# front height fixed
# v10: v2w changed for front_rect_mat_4x4, cam2world output will have opengl coords
# INDEPENDENT VERSION: all config is hard-coded below (no argparse)
# change from v10: using v2w_frame instead of v2w_img
# change from v11: using an middle as the orgin. extr_middle_1
# change from v12: using the og method for bounding box

# ================== USER CONFIG (EDIT THESE) ==================
BASE_PATH  = "/workspace/Jack/HUGSIM/raw_data/split_311238_10parts"
SEGMENT    = "311238_part_0.tfrecord"
CAMERAS    = [1, 2, 3]   # camera enums, e.g. [dataset_pb2.CameraName.FRONT, ...] if using enums directly
OUTPATH    = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_new_v60"
DOWNSAMPLE = 2.0         # image downsample factor
MAX_FRAMES = None        # e.g. 50 to stop after 50 frames; None = all frames
# =============================================================

extr_middle_1 = np.array([                                                  #cam1 frame 0 -90 degree about x axis(veh)
    [1, 0, 0, -0.375137],
    [0, 0,  1,  2.551414],
    [0, -1, 0,  1.918190],
    [0.0,       0.0,       0.0,       1.0]], dtype=float)

ego_pose_frame_f1 = np.array([
    [ 0.034452807158231735,   0.9994063377380371,   -0.00013757309352513403, 805507.0],
    [-0.9993419647216797,     0.034449029713869095, -0.011351890861988068,   3299242.0],
    [-0.011340412311255932,   0.0005285870865918696, 0.9999355673789978,     14.962104797363281],
    [ 0.0,                     0.0,                   0.0,                   1.0],
], dtype=float)

type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

def roty_matrix(roty):
    c = np.cos(roty)
    s = np.sin(roty)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def get_vertices(dim, bottom_center=np.array([0.0, 0.0, 0.0])):
    """
    dim: length, height, width
    bottom_center: center of bottom face of 3D bounding box
    return: vertices of 3D bounding box (8*3)
    """
    vertices = bottom_center[None, :].repeat(8, axis=0)
    vertices[:4, 0] = vertices[:4, 0] + dim[0] / 2
    vertices[4:, 0] = vertices[4:, 0] - dim[0] / 2
    vertices[[0, 1, 4, 5], 1] = vertices[[0, 1, 4, 5], 1]
    vertices[[2, 3, 6, 7], 1] = vertices[[2, 3, 6, 7], 1] - dim[1]
    vertices[[0, 2, 5, 7], 2] = vertices[[0, 2, 5, 7], 2] + dim[2] / 2
    vertices[[1, 3, 4, 6], 2] = vertices[[1, 3, 4, 6], 2] - dim[2] / 2
    return vertices

camera_names_dict = {
    dataset_pb2.CameraName.FRONT_LEFT: 'FRONT_LEFT',
    dataset_pb2.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
    dataset_pb2.CameraName.FRONT: 'FRONT',
    dataset_pb2.CameraName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
}

def parse_range_image_custom(laser, second_response: bool = False):
    """
    Parse range image for both:
      - standard Waymo frames (compressed)
      - your custom frames (uncompressed RangeImage.range_image).

    Returns:
        ri: (H, W, C) float32
        camera_projection: np.ndarray or None
        range_image_pose: np.ndarray or None
    """
    # --- 1) Try the normal compressed path first (for real Waymo data) ---
    try:
        if not second_response:
            if len(laser.ri_return1.range_image_compressed) > 0:
                return utils.parse_range_image_and_camera_projection(
                    laser, second_response=False
                )
        else:
            if len(laser.ri_return2.range_image_compressed) > 0:
                return utils.parse_range_image_and_camera_projection(
                    laser, second_response=True
                )
    except Exception:
        pass

    # --- 2) Custom path: uncompressed data in RangeImage.range_image ---
    ri_msg = laser.ri_return2 if second_response else laser.ri_return1

    if ri_msg is None:
        raise RuntimeError(f"No ri_return{2 if second_response else 1} for laser {laser.name}")

    mat = ri_msg.range_image  # MatrixFloat
    if (mat is None) or (len(mat.data) == 0):
        raise RuntimeError(
            f"No range_image data found in uncompressed layout for laser {laser.name}"
        )

    dims = mat.shape.dims  # e.g. [H, W, 4]
    ri = np.array(mat.data, dtype=np.float32).reshape(dims)

    camera_projection = None
    range_image_pose  = None

    return ri, camera_projection, range_image_pose


if __name__ == '__main__':
    # ==== Paths and reader ====
    seq_path = os.path.join(BASE_PATH, SEGMENT)
    datafile = WaymoDataFileReader(seq_path)
    num_frames = len(datafile.get_record_table())

    # ==== Create folders ====
    save_dir = OUTPATH
    os.makedirs(save_dir, exist_ok=True)
    cams = CAMERAS
    for cam in cams:
        os.makedirs(os.path.join(save_dir, "images", f"cam_{cam}"), exist_ok=True)

    ##########################################################################
    #         read calib from frame 0 and lidar from first data frame        #
    ##########################################################################

    plane_reader = WaymoDataFileReader(seq_path)
    data_iter = iter(plane_reader)

    try:
        calib_frame = next(data_iter)
    except StopIteration:
        raise RuntimeError("TFRecord is empty – no frames found.")

    lidar_frame = None
    for fr in data_iter:
        if len(fr.lasers) > 0:
            lidar_frame = fr
            break

    if lidar_frame is None:
        raise RuntimeError("No frame with lidar data found after calibration frame.")

    lidar_points = []
    for laser_name in [
        dataset_pb2.LaserName.TOP,
        dataset_pb2.LaserName.FRONT,
        dataset_pb2.LaserName.SIDE_LEFT,
        dataset_pb2.LaserName.SIDE_RIGHT,
    ]:
        laser = utils.get(lidar_frame.lasers, laser_name)
        laser_calibration = utils.get(calib_frame.context.laser_calibrations, laser_name)

        range_images, camera_projections, range_image_top_pose = parse_range_image_custom(laser)

        points, _ = utils.project_to_pointcloud(
            lidar_frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            laser_calibration
        )
        lidar_points.append(points[:, :3])

    lidar_points = np.concatenate(lidar_points, axis=0)

    ground_mask = (np.abs(lidar_points[:, 0]) < 6) & (np.abs(lidar_points[:, 1]) < 3)
    lidar_points = lidar_points[ground_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    o3d.io.write_point_cloud(os.path.join(save_dir, 'ground_lidar.ply'), pcd)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    a, b, c, d = plane_model

    # --- 5) camera calib: use camera_calibrations from frame 0 (calib_frame) ---
    all_campose = {}
    for camera in calib_frame.context.camera_calibrations:
        if camera.name not in cams:
            continue
        c2v = np.array(camera.extrinsic.transform).reshape(4, 4)
        all_campose[camera.name] = c2v  # don't apply opengl2waymo here yet

    FRONT = dataset_pb2.CameraName.FRONT
    front_enum = FRONT if FRONT in all_campose else next(iter(all_campose.keys()))
    front_cam_t = all_campose[front_enum][:3, 3]

    height_ground_at_cam = -(a * front_cam_t[0] + b * front_cam_t[1] + d) / c

    front_cam_info = {
        "height": float(abs(front_cam_t[2])),
        "rect_mat": None,
    }
    with open(os.path.join(save_dir, 'front_info.json'), 'w') as f:
        json.dump(front_cam_info, f, indent=2)

    print("read first frame info done (using calib from frame 0 and lidar from first data frame).")

    ##########################################################################
    #                     Process all frames for data                        #
    ##########################################################################

    it = iter(datafile)
    try:
        frame0 = next(it)
    except StopIteration:
        raise RuntimeError("Empty TFRecord: no frames found")

    ctx_cam_cals = {cal.name: cal for cal in frame0.context.camera_calibrations}
    ctx_laser_cals = {cal.name: cal for cal in frame0.context.laser_calibrations}

    # Per-camera aligned storage
    per_cam_idx        = {cam: 0    for cam in cams}
    ego_poses          = {cam: []   for cam in cams}  # per-image v2w_img at this camera timestamp
    intr               = {cam: []   for cam in cams}
    extr               = {cam: []   for cam in cams}
    imsize             = {cam: []   for cam in cams}
    timestamps_per_cam = {cam: []   for cam in cams}

    vehicles           = {}
    dynamics           = {}
    c2ws               = {}
    global_to_local    = {cam: {} for cam in cams}

    # store raw 3D box parameters for each global frame
    boxes_per_frame = {}   # key: global_frame index, value: list of dicts

    start_timestamp = None
    global_frame = 0

    # map global frame index -> frame.pose.transform (4x4)
    frame_v2w_by_global = {}

    # per-camera list mapping local index -> global frame index
    cam_global_ids = {cam: [] for cam in cams}

    # ---- main frame loop ----
    for rec in tqdm(it, desc="Waymo frames"):
        if MAX_FRAMES is not None and global_frame >= MAX_FRAMES:
            break

        frame = rec

        if not frame.images or not frame.pose.transform:
            continue

        if start_timestamp is None:
            start_timestamp = frame.timestamp_micros / 1e6
        t_abs = frame.timestamp_micros / 1e6
        current_global = global_frame
        global_frame += 1

        # store frame.pose.transform (ego->world) for this global frame
        v2w_frame = np.array(frame.pose.transform).reshape(4, 4)
        frame_v2w_by_global[current_global] = v2w_frame

        # ---- IMAGES (per camera) ----
        for img_pkg in frame.images:
            cam = img_pkg.name
            if cam not in cams:
                continue

            img = cv2.imdecode(np.frombuffer(img_pkg.image, np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            if DOWNSAMPLE > 1:
                h = int(h // DOWNSAMPLE)
                w = int(w // DOWNSAMPLE)
                img = cv2.resize(img, (w, h))

            i_cam = per_cam_idx[cam]
            global_to_local[cam][current_global] = i_cam

            cam_global_ids[cam].append(current_global)

            out_png = os.path.join(save_dir, "images", f"cam_{cam}", f"{i_cam:06d}.png")
            cv2.imwrite(out_png, img)

            imsize[cam].append((h, w))
            timestamps_per_cam[cam].append(t_abs)

            # img_pkg.pose.transform: ego->world at image time
            v2w_img = np.array(img_pkg.pose.transform).reshape(4, 4)
            ego_poses[cam].append(v2w_img)

            # Intrinsics
            cal = ctx_cam_cals[cam]
            K = np.eye(4, dtype=float)
            K[0, 0] = cal.intrinsic[0] / DOWNSAMPLE
            K[1, 1] = cal.intrinsic[1] / DOWNSAMPLE
            K[0, 2] = cal.intrinsic[2] / DOWNSAMPLE
            K[1, 2] = cal.intrinsic[3] / DOWNSAMPLE
            intr[cam].append(K)

            # Extrinsics (c2v)
            c2v_ctx = np.array(cal.extrinsic.transform).reshape(4, 4)
            extr[cam].append(c2v_ctx)

            per_cam_idx[cam] += 1

        # ---- ego pose for 3D boxes ----
        v2w = v2w_frame

        if current_global not in boxes_per_frame:
            boxes_per_frame[current_global] = []

        # ---- 3D bounding boxes → vehicles dict + raw box logging ----
        for obj in frame.laser_labels:
            type_name = type_list[obj.type]

            # raw box params from Waymo
            height = obj.box.height
            width  = obj.box.width
            length = obj.box.length
            cx = obj.box.center_x
            cy = obj.box.center_y
            cz = obj.box.center_z
            heading = obj.box.heading

            boxes_per_frame[current_global].append({
                "id": int(obj.id),
                "type": type_name,
                "height": float(height),
                "width": float(width),
                "length": float(length),
                "center_x": float(cx),
                "center_y": float(cy),
                "center_z": float(cz),
                "heading": float(heading),
            })

            # bottom-center computation
            x = cx
            y = cy
            z = cz - height / 2

            t_b2l = np.array([x, y, z, 1]).reshape((4, 1))
            t_b2w = v2w @ t_b2l
            rotation_y = -heading

            if type_name in ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
                if obj.id not in vehicles:
                    vehicles[obj.id] = {
                        "rt": [],
                        "timestamp": [],
                        "frame": [],
                    }
                vehicles[obj.id]["rt"].append(
                    np.array(t_b2w[:3, 0].tolist() + [length, height, width, rotation_y])
                )
                vehicles[obj.id]["timestamp"].append(t_abs)
                vehicles[obj.id]["frame"].append(current_global)

    # ---- Build c2w using aligned lists ----
    for cam in cams:
        c2ws[cam] = []
        for i in range(len(ego_poses[cam])):
            # which global frame did this local camera index come from?
            global_id   = cam_global_ids[cam][i]
            v2w_frame   = frame_v2w_by_global[global_id]   # <-- use frame-level ego->world
            c2v_ctx     = extr[cam][i]                     # camera->vehicle from calib
            c2w         = v2w_frame @ c2v_ctx              # camera->world
            c2ws[cam].append(c2w)

        c2ws[cam] = np.stack(c2ws[cam], axis=0)

    FRONT = dataset_pb2.CameraName.FRONT
    # inv_pose = np.linalg.inv(c2ws[FRONT][0])
    inv_pose = np.linalg.inv(ego_pose_frame_f1 @ extr_middle_1)

    # normalize all c2ws so FRONT[0] becomes identity
    for cam in cams:
        poses = c2ws[cam]  # (N,4,4)
        poses = np.einsum('njk,ij->nik', poses, inv_pose)
        c2ws[cam] = poses

    # filter dynamic vehicles
    dynamic_id = 0
    for objid, infos in vehicles.items():
        infos['rt'] = np.stack(infos['rt'])
        trans = infos['rt'][:, :3]
        trans = np.einsum('njk,ij->nik', trans[..., None], inv_pose[:3, :3])
        trans = trans[..., 0] + inv_pose[:3, 3]
        movement = np.max(np.max(trans, axis=0) - np.min(trans, axis=0))
        if movement > 1:
            infos["rt"][:, :3] = trans
            dynamics[dynamic_id] = infos
            dynamic_id += 1
    

    # # ---- filter dynamic vehicles ----
    # dynamic_id = 0
    # target_obj = 8053  # debugging

    # for objid, infos in vehicles.items():
    #     base_rt = np.stack(infos["rt"])      # (T, 7)
    #     base_trans = base_rt[:, :3]          # (T, 3)
    #     base_lhw_roty = base_rt[:, 3:]       # (T, 4) -> [L, H, W, roty]

    #     T = base_rt.shape[0]
    #     rt_per_cam = {
    #         cam: np.zeros((T, 7), dtype=np.float32)
    #         for cam in cams
    #     }

    #     for t_idx, fid_global in enumerate(infos["frame"]):
    #         p = base_trans[t_idx]  # (3,)

    #         for cam in cams:
    #             if fid_global not in global_to_local[cam]:
    #                 continue

    #             i_local = global_to_local[cam][fid_global]
    #             c2v = extr[cam][i_local]

    #             c2v_inv = np.linalg.inv(c2v)
    #             R = c2v_inv[:3, :3]
    #             t = c2v_inv[:3, 3]

    #             p_new = R @ p + t

    #             rt_per_cam[cam][t_idx, :3] = p_new
    #             rt_per_cam[cam][t_idx, 3:] = base_lhw_roty[t_idx]

    #     if dataset_pb2.CameraName.FRONT in cams:
    #         front_cam = dataset_pb2.CameraName.FRONT
    #     else:
    #         front_cam = cams[0]

    #     trans_front = rt_per_cam[front_cam][:, :3]
    #     movement = np.max(np.max(trans_front, axis=0) - np.min(trans_front, axis=0))

    #     if movement > 0.1:
    #         cams_for_obj = []
    #         rt_list = []

    #         for cam in cams:
    #             cams_for_obj.append(cam)
    #             rt_list.append(rt_per_cam[cam])
    #         rt_arr = np.stack(rt_list, axis=1)  # (T, C, 7)

    #         infos["rt"]   = rt_arr.tolist()
    #         infos["cams"] = cams_for_obj
    #         infos["obj_id"] = int(objid)

    #         dynamics[dynamic_id] = infos
    #         dynamic_id += 1

    verts = {}
    rts_per_cam = {cam: {} for cam in cams}

    for dynamic_id, infos in dynamics.items():
        # if dynamic_id == 1:
        # print("processing dynamic_id:", dynamic_id)
        # print("infos frame:", infos['frame'])
        lhw = np.array(infos['rt'][0, 3:6])
        points = get_vertices(lhw)              # (8,3) in object space
        trans  = infos['rt'][:, 0:3]            # (T,3) world translations (already normalized earlier)
        roty   = infos['rt'][:, 6]              # (T,) yaw in world
        seq_visible = False
        for idx, fid_global in enumerate(infos['frame']):   # fid_global = your current_global
            # # Build box world transform at this time step
            # rt = np.eye(4)
            # # Camera yaw term from extrinsics is not appropriate here—use world yaw only
            # # If you do need a camera-relative yaw, compute it from c2ws when you project below.
            # rt[:3, :3] = roty_matrix(roty[idx])  # world yaw only
            # rt[:3, 3]  = trans[idx]
    
            # points_w = (rt[:3, :3] @ points.T).T + rt[:3, 3]   # (8,3) world
            frame_visible = False
            
            # print("global frame:", fid_global)
            # print("idx:", idx)
            # Try projection onto each requested camera at this global frame
            # print("processing dynamic_id", dynamic_id, "frame", fid_global)
            for cam in cams:
                ##### DEBUGGING #####
                # if cam == 2:
                #     print("processing cam 2 for dynamic_id", dynamic_id)
                ############################

                if fid_global not in global_to_local[cam]:
                    print("not found global frame", fid_global, "in cam", cam)
                    continue  # that camera didn't produce an image for this record
    
                i_local = global_to_local[cam][fid_global]   # per-cam index
                ##### DEBUGGING #####
                # if dynamic_id == 1 and cam == 1:
                #     print("processing cam 1 for dynamic_id 1")
                #     print("i_local:", i_local)
                #     print("global_to_local[1]:", global_to_local[cam])
                #     print("global_to_local[2]:", global_to_local[2])
                #     print("global_to_local[3]:", global_to_local[3])
                ############################
                c2w = c2ws[cam][i_local]
                w2c = np.linalg.inv(c2w)
                K   = intr[cam][i_local]
                h, w = imsize[cam][i_local]

                # --- camera yaw from c2w ---
                R_cam   = c2w[:3, :3]
                cam_roty = SCR.from_matrix(R_cam).as_euler('yxz')[0]  # radians

                # --- build box transform using camera-relative yaw ---
                yaw_rel = roty[idx]  # or roty[idx] + cam_roty, depending on convention

                rt = np.eye(4)
                rt[:3, :3] = roty_matrix(yaw_rel)
                rt[:3, 3]  = trans[idx]

                # box corners in world, then project
                points_w   = (rt[:3, :3] @ points.T).T + rt[:3, 3]    # (8,3) world
                

    
                points_cam = (w2c[:3, :3] @ points_w.T).T + w2c[:3, 3]      # (8,3)
                Z = points_cam[:, 2]
                # if dynamic_id == 1 and cam == 2:
                #     print("points_cam:", points_cam)
                #     print("Z:", Z)
                if np.all(Z <= 0):
                    continue
    
                pts_scr = (K[:3, :3] @ points_cam.T).T + K[:3, 3]           # (8,3)
                pts_uv  = (pts_scr[:, :2] / Z[:, None]).astype(int)         # (8,2)
                valid_mask = (Z > 0) & (pts_uv[:,0] >= 0) & (pts_uv[:,1] >= 0) & \
                             (pts_uv[:,0] < w) & (pts_uv[:,1] < h)

                # if dynamic_id == 1 and cam == 2:
                #     print("cam 2 debug info:")
                #     print("dynamic_id:", dynamic_id)
                #     print("global frame:", fid_global)
                #     print("projected pts_uv:", pts_uv)
                #     print("valie mask:", valid_mask)
    
                if np.any(valid_mask):
                    frame_visible = True
                    seq_visible   = True
    
                    if i_local not in rts_per_cam[cam]:
                        rts_per_cam[cam][i_local] = {}
    
                    # Store the world-space rt (or camera-space if you prefer)
                    rts_per_cam[cam][i_local][dynamic_id] = rt.tolist()
                    # break  # one cam visibility is enough

                if frame_visible:
                    pass
    
        if seq_visible:
            verts[dynamic_id] = points.tolist()
    # Ensure list lengths are consistent per cam (truncate to the minimum)




    # ---- post process dynamics: visibility per camera ----
    # verts = {}
    # rts_per_cam = {cam: {} for cam in cams}
    # target_id = 1

    # for dynamic_id, infos in dynamics.items():
    #     rt_arr = np.asarray(infos["rt"], dtype=float)  # (T, C, 7)
    #     if rt_arr.ndim != 3 or rt_arr.shape[2] != 7:
    #         raise ValueError(f"Expected rt shape (T,C,7), got {rt_arr.shape} for dynamic_id {dynamic_id}")

    #     T, C, _ = rt_arr.shape
    #     cams_for_obj = infos["cams"]
    #     frames_for_obj = infos["frame"]

    #     lhw = rt_arr[0, 0, 3:6]
    #     lhw = np.array(lhw, dtype=float)
    #     points = get_vertices(lhw)

    #     seq_visible = False

    #     try:
    #         objid_int = int(dynamic_id)
    #     except Exception:
    #         continue

    #     for t_idx, fid_global in enumerate(frames_for_obj):
    #         for j, cam_src in enumerate(cams_for_obj):
    #             trans = rt_arr[t_idx, j, 0:3]
    #             roty  = rt_arr[t_idx, j, 6]

    #             if cam_src == 2:
    #                 trans[0] -= 2.0

    #             rt = np.eye(4, dtype=float)

    #             if cam_src in cams and fid_global in global_to_local[cam_src]:
    #                 i_src = global_to_local[cam_src][fid_global]
    #                 R_src = c2ws[cam_src][i_src][:3, :3]
    #                 cam_roty = SCR.from_matrix(R_src).as_euler('yxz')[0]
    #                 if cam_src == 2:
    #                     cam_roty = -0.4
    #             elif FRONT in cams and fid_global in global_to_local[FRONT]:
    #                 i_front = global_to_local[FRONT][fid_global]
    #                 R_front = c2ws[FRONT][i_front][:3, :3]
    #                 cam_roty = SCR.from_matrix(R_front).as_euler('yxz')[0]
    #             else:
    #                 cam_roty = 0.0
    #                 print('No suitable camera for yaw for frame', fid_global, 'and cam', cam_src)

    #             rt[:3, :3] = roty_matrix(roty - cam_roty)
    #             rt[:3, 3]  = trans

    #             points_w = (rt[:3, :3] @ points.T).T + rt[:3, 3]
    #             frame_visible = False

    #             cam = cam_src
    #             if cam not in cams:
    #                 continue
    #             if fid_global not in global_to_local[cam]:
    #                 continue

    #             i_local = global_to_local[cam][fid_global]
    #             c2w = c2ws[cam][i_local]
    #             w2c = np.linalg.inv(c2w)
    #             w2c = np.eye(4)
    #             K   = intr[cam][i_local]
    #             h, w_img = imsize[cam][i_local]

    #             points_cam = (w2c[:3, :3] @ points_w.T).T + w2c[:3, 3]
    #             Z = points_cam[:, 2]

    #             pts_scr = (K[:3, :3] @ points_cam.T).T + K[:3, 3]
    #             pts_uv  = (pts_scr[:, :2] / Z[:, None]).astype(int)

    #             valid_mask = (
    #                 (Z > 0) &
    #                 (pts_uv[:, 0] >= 0) & (pts_uv[:, 1] >= 0) &
    #                 (pts_uv[:, 0] < w_img) & (pts_uv[:, 1] < h)
    #             )

    #             if np.any(valid_mask):
    #                 frame_visible = True
    #                 seq_visible   = True

    #                 if i_local not in rts_per_cam[cam]:
    #                     rts_per_cam[cam][i_local] = {}

    #                 rts_per_cam[cam][i_local][dynamic_id] = rt.tolist()

    #             if frame_visible:
    #                 pass

    #     if seq_visible:
    #         verts[dynamic_id] = points.tolist()

    # ---- write meta_data.json ----
    meta_data = {
        "camera_model": "OPENCV",
        "frames": [],
        "verts": verts,
        "inv_pose": inv_pose.tolist()
    }

    for cam in cams:
        N = len(intr[cam])
        for i in range(N):
            h, w = imsize[cam][i]

            global_id = cam_global_ids[cam][i]
            frame_v2w = frame_v2w_by_global.get(global_id, None)

            info = {
                "rgb_path": f"./images/cam_{cam}/{i:06d}.png",
                "camtoworld": c2ws[cam][i].tolist(),
                "intrinsics": intr[cam][i].tolist(),
                "width": int(w),
                "height": int(h),
                "timestamp": float(timestamps_per_cam[cam][i]),
                "dynamics": rts_per_cam[cam].get(i, {}),
                "ego_pose_image": ego_poses[cam][i].tolist()
            }

            info["laser_labels"] = boxes_per_frame.get(global_id, [])

            if frame_v2w is not None:
                info["ego_pose_frame"] = frame_v2w.tolist()

            meta_data["frames"].append(info)

    with open(os.path.join(save_dir, 'meta_data.json'), 'w') as wf:
        json.dump(meta_data, wf, indent=2)

    print("Done. meta_data.json written to:", os.path.join(save_dir, 'meta_data.json'))


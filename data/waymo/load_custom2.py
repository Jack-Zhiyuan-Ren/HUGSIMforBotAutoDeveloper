# import os
# import argparse
# import open3d as o3d
# import numpy as np
# from tqdm import tqdm
# import json
# import cv2
# from scipy.spatial.transform import Rotation as SCR
# from simple_waymo_open_dataset_reader import WaymoDataFileReader
# from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
# from simple_waymo_open_dataset_reader import utils

# opengl2waymo = np.array([[0, 0, 1, 0],
#                         [-1, 0, 0, 0],
#                         [0, -1, 0, 0],
#                         [0, 0, 0, 1]])

# opengl2waymo2 = np.array([[1, 0, 0, 0],
#                         [0, 0, -1, 0],
#                         [0, 1, 0, 0],
#                         [0, 0, 0, 1]])

# M = np.array([[ 0.,  0.,  1.],   ## This is the old one
#               [0,  -1,  0.],
#               [ 1., 0.,  0.]], dtype=float)

# # M = np.array([[ 0.,  -1.,  0],    # This is the new one
# #               [1,  0,  0.],
# #               [ 0., 0.,  1]], dtype=float)

# type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

# def get_opts():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-b', '--base_path', type=str, required=True)
#     parser.add_argument('-s', '--segment', type=str, required=True)
#     parser.add_argument('-c', '--cameras', nargs='+', type=int, required=True)
#     parser.add_argument('-o', '--outpath', type=str, required=True)
#     parser.add_argument('--downsample', type=float, default=2)
#     return parser.parse_args()

# def roty_matrix(roty):
#     c = np.cos(roty)
#     s = np.sin(roty)
#     return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

# def get_vertices(dim, bottom_center=np.array([0.0 ,0.0 ,0.0 ])):
#     '''
#     dim: length, height, width
#     bottom_center: center of bottom face of 3D bounding box

#     return: vertices of 3D bounding box (8*3)
#     '''
#     vertices = bottom_center[None, :].repeat(8, axis=0)
#     vertices[:4, 0] = vertices[:4, 0] + dim[0] / 2
#     vertices[4:, 0] = vertices[4:, 0] - dim[0] / 2 
#     vertices[[0,1,4,5], 1] = vertices[[0,1,4,5], 1]
#     vertices[[2,3,6,7], 1] = vertices[[2,3,6,7], 1] - dim[1]
#     vertices[[0,2,5,7], 2] = vertices[[0,2,5,7], 2] + dim[2] / 2
#     vertices[[1,3,4,6], 2] = vertices[[1,3,4,6], 2] - dim[2] / 2

#     return vertices

# camera_names_dict = {
#     dataset_pb2.CameraName.FRONT_LEFT: 'FRONT_LEFT', 
#     dataset_pb2.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
#     dataset_pb2.CameraName.FRONT: 'FRONT', 
#     dataset_pb2.CameraName.SIDE_LEFT: 'SIDE_LEFT',
#     dataset_pb2.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
# }

# def parse_range_image_custom(laser, second_response: bool = False):
#     """
#     Parse range image for both:
#       - standard Waymo frames (compressed)
#       - your custom frames (uncompressed RangeImage.range_image).

#     Returns:
#         ri: (H, W, C) float32
#         camera_projection: np.ndarray or None
#         range_image_pose: np.ndarray or None
#     """
#     # --- 1) Try the normal compressed path first (for real Waymo data) ---
#     #     If this works, we just reuse the original function.
#     try:
#         if not second_response:
#             if len(laser.ri_return1.range_image_compressed) > 0:
#                 return utils.parse_range_image_and_camera_projection(
#                     laser, second_response=False
#                 )
#         else:
#             if len(laser.ri_return2.range_image_compressed) > 0:
#                 return utils.parse_range_image_and_camera_projection(
#                     laser, second_response=True
#                 )
#     except Exception:
#         # If something goes wrong, we fall back to custom path below.
#         pass

#     # --- 2) Custom path: uncompressed data in RangeImage.range_image ---
#     # Choose the appropriate return.
#     ri_msg = laser.ri_return2 if second_response else laser.ri_return1

#     if ri_msg is None:
#         raise RuntimeError(f"No ri_return{2 if second_response else 1} for laser {laser.name}")

#     # In your custom files, the uncompressed matrix is stored in deprecated 'range_image'
#     mat = ri_msg.range_image  # MatrixFloat
#     if (mat is None) or (len(mat.data) == 0):
#         raise RuntimeError(
#             f"No range_image data found in uncompressed layout for laser {laser.name}"
#         )

#     dims = mat.shape.dims  # e.g. [H, W, 4]
#     ri = np.array(mat.data, dtype=np.float32).reshape(dims)

#     # Your custom data doesn't have these:
#     camera_projection = None   # no camera_projection_compressed
#     range_image_pose  = None   # no range_image_pose_compressed

#     return ri, camera_projection, range_image_pose

# if __name__ == '__main__':
#     args = get_opts()

#     seq_path = os.path.join(args.base_path, f"{args.segment}")
#     datafile = WaymoDataFileReader(seq_path)
#     num_frames = len(datafile.get_record_table())
    
#     # create folders
#     save_dir = args.outpath
#     os.makedirs(save_dir, exist_ok=True)
#     cams = args.cameras
#     for cam in cams:
#         os.makedirs(os.path.join(save_dir, "images", f"cam_{cam}"), exist_ok=True)
        

#     ##########################################################################
#     #         read calib from frame 0 and lidar from first data frame        #
#     #           lidar is only used for extracting camera height              #
#     ##########################################################################
    
#     # Use a *separate* reader here so we don't disturb the main loader
#     plane_reader = WaymoDataFileReader(seq_path)
#     data_iter = iter(plane_reader)


#     # --- 1) frame 0: only calibrations (no lidar/images) ---
#     try:
#         calib_frame = next(data_iter)   # this is your custom "anchor" frame
#     except StopIteration:
#         raise RuntimeError("TFRecord is empty – no frames found.")

#     # --- 2) find first frame that actually has lidar points ---
#     lidar_frame = None
#     for fr in data_iter:
#         # depending on your proto, this could be: if fr.lasers:
#         if len(fr.lasers) > 0:
#             lidar_frame = fr
#             break

#     if lidar_frame is None:
#         raise RuntimeError("No frame with lidar data found after calibration frame.")

#     # --- 3) build lidar point cloud using:
#     #       - lidar_frame.lasers  (range images)
#     #       - calib_frame.context.laser_calibrations (extrinsics)
#     lidar_points = []
#     for laser_name in [
#         dataset_pb2.LaserName.TOP,
#         dataset_pb2.LaserName.FRONT,
#         dataset_pb2.LaserName.SIDE_LEFT,
#         dataset_pb2.LaserName.SIDE_RIGHT,
#     ]:
#         # lidar measurements from lidar_frame
#         laser = utils.get(lidar_frame.lasers, laser_name)

#         # calibration from calib_frame (frame 0)
#         laser_calibration = utils.get(calib_frame.context.laser_calibrations, laser_name)

#         range_images, camera_projections, range_image_top_pose = parse_range_image_custom(laser)

#         points, _ = utils.project_to_pointcloud(
#             lidar_frame,          # frame that has the measurements
#             range_images,
#             camera_projections,
#             range_image_top_pose,
#             laser_calibration     # extrinsics from frame 0
#         )
#         lidar_points.append(points[:, :3])  # in ego pose

#     lidar_points = np.concatenate(lidar_points, axis=0)

#     # --- 4) ground selection + plane fitting (same as before) ---
#     ground_mask = (np.abs(lidar_points[:, 0]) < 6) & (np.abs(lidar_points[:, 1]) < 3)
#     lidar_points = lidar_points[ground_mask]

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(lidar_points)
#     o3d.io.write_point_cloud(os.path.join(args.outpath, 'ground_lidar.ply'), pcd)

#     plane_model, inliers = pcd.segment_plane(
#         distance_threshold=0.01,
#         ransac_n=3,
#         num_iterations=1000
#     )
#     a, b, c, d = plane_model

#     # --- 5) camera calib: use camera_calibrations from frame 0 (calib_frame) ---


#     all_campose = {}
#     for camera in calib_frame.context.camera_calibrations:
#         if camera.name not in args.cameras:
#             continue
#         c2v = np.array(camera.extrinsic.transform).reshape(4, 4)
#         all_campose[camera.name] = c2v @ opengl2waymo#combo

#     FRONT = dataset_pb2.CameraName.FRONT
#     front_enum = FRONT if FRONT in all_campose else next(iter(all_campose.keys()))
#     front_cam_t = all_campose[front_enum][:3, 3]


#     # plane: a x + b y + c z + d = 0
#     # solve for ground height at camera (x,y), then camera height above ground
#     height_ground_at_cam = -(a * front_cam_t[0] + b * front_cam_t[1] + d) / c

#     front_cam_info = {
#         "height": front_cam_t[2] - height_ground_at_cam,
#         "rect_mat": None,
#     }
#     with open(os.path.join(args.outpath, 'front_info.json'), 'w') as f:
#         json.dump(front_cam_info, f, indent=2)

#     print("read first frame info done (using calib from frame 0 and lidar from first data frame).")

#     ##########################################################################
#     #                     Process all frames for data                        #
#     ##########################################################################

#     it = iter(datafile)
#     try:
#         frame0 = next(it)
#     except StopIteration:
#         raise RuntimeError("Empty TFRecord: no frames found")

#     ctx_cam_cals = {cal.name: cal for cal in frame0.context.camera_calibrations}
#     ctx_laser_cals = {cal.name: cal for cal in frame0.context.laser_calibrations}

#     cams = args.cameras

#     # Per-camera aligned storage
#     per_cam_idx        = {cam: 0    for cam in cams}
#     ego_poses          = {cam: []   for cam in cams}  # per-image v2w at this camera timestamp
#     intr               = {cam: []   for cam in cams}  # per-image intrinsics
#     extr               = {cam: []   for cam in cams}  # per-image c2v
#     imsize             = {cam: []   for cam in cams}  # per-image (h, w)
#     timestamps_per_cam = {cam: []   for cam in cams}  # per-image absolute timestamp

#     vehicles           = {}
#     dynamics           = {}
#     c2ws               = {}
#     global_to_local    = {cam: {} for cam in cams}

#     combo = np.eye(4)
#     combo[:3, :3] = opengl2waymo[:3, :3] @ M

#     start_timestamp = None
#     global_frame = 0

#     for rec in tqdm(it):
#         frame = rec

#         # Skip frames without images or pose
#         if not frame.images or not frame.pose.transform:
#             continue

#         if start_timestamp is None:
#             start_timestamp = frame.timestamp_micros / 1e6
#         t_abs = frame.timestamp_micros / 1e6
#         current_global = global_frame
#         global_frame += 1

#         # ---- IMAGES (per camera) ----
#         for img_pkg in frame.images:
#             cam = img_pkg.name
#             if cam not in cams:
#                 continue

#             # Decode & optional downsample
#             img = cv2.imdecode(np.frombuffer(img_pkg.image, np.uint8), cv2.IMREAD_COLOR)
#             h, w = img.shape[:2]
#             if args.downsample > 1:
#                 h = int(h // args.downsample)
#                 w = int(w // args.downsample)
#                 img = cv2.resize(img, (w, h))

#             # Per-camera index drives file name and indexing
#             i_cam = per_cam_idx[cam]
#             global_to_local[cam][current_global] = i_cam
#             out_png = os.path.join(save_dir, "images", f"cam_{cam}", f"{i_cam:06d}.png")
#             cv2.imwrite(out_png, img)

#             imsize[cam].append((h, w))
#             timestamps_per_cam[cam].append(t_abs)

#             v2w_img = np.array(img_pkg.pose.transform).reshape(4, 4)
#             ego_poses[cam].append(v2w_img)

#             # Intrinsics from cached context
#             cal = ctx_cam_cals[cam]
#             K = np.eye(4, dtype=float)
#             K[0, 0] = cal.intrinsic[0] / args.downsample
#             K[1, 1] = cal.intrinsic[1] / args.downsample
#             K[0, 2] = cal.intrinsic[2] / args.downsample
#             K[1, 2] = cal.intrinsic[3] / args.downsample
#             intr[cam].append(K)

#             # Extrinsics (c2v) from cached context
#             c2v_ctx = np.array(cal.extrinsic.transform).reshape(4, 4)
#             extr[cam].append(c2v_ctx)

#             per_cam_idx[cam] += 1

#         # ---- ego pose for 3D boxes ----
#         v2w = np.array(frame.pose.transform).reshape(4, 4)

#         # ---- 3D bounding boxes → vehicles dict ----
#         for obj in frame.laser_labels:
#             type_name = type_list[obj.type]
#             height = obj.box.height
#             width  = obj.box.width
#             length = obj.box.length
#             x = obj.box.center_x
#             y = obj.box.center_y
#             z = obj.box.center_z - height / 2

#             t_b2l = np.array([x, y, z, 1]).reshape((4, 1))
#             t_b2w = v2w @ t_b2l
#             rotation_y = -obj.box.heading - np.pi / 2

#             if type_name in ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
#                 if obj.id not in vehicles:
#                     vehicles[obj.id] = {
#                         "rt": [],
#                         "timestamp": [],
#                         "frame": [],
#                     }
#                 vehicles[obj.id]["rt"].append(np.array(t_b2w[:3, 0].tolist() + [length, height, width, rotation_y]))
#                 vehicles[obj.id]["timestamp"].append(t_abs)
#                 vehicles[obj.id]["frame"].append(current_global)


#     # ---- Build c2w using aligned lists ----
#     for cam in cams:
#         c2ws[cam] = []
#         for i in range(len(ego_poses[cam])):
#             v2w_img = ego_poses[cam][i]
#             c2v_ctx = extr[cam][i]
#             c2w = v2w_img @ c2v_ctx @ opengl2waymo #combo
#             c2ws[cam].append(c2w)
#         c2ws[cam] = np.stack(c2ws[cam], axis=0)

#     FRONT = dataset_pb2.CameraName.FRONT
#     inv_pose = np.linalg.inv(c2ws[FRONT][0])

#     # normalize all c2ws so FRONT[0] becomes identity
#     for cam in cams:
#         poses = c2ws[cam]  # (N,4,4)
#         poses = np.einsum('njk,ij->nik', poses, inv_pose)
#         # c2ws[cam] = np.einsum('ij,njk->nik', inv_pose, poses) 
#         c2ws[cam] = poses




#     # ---- filter dynamic vehicles ----
#     dynamic_id = 0
#     for objid, infos in vehicles.items():
#         infos['rt'] = np.stack(infos['rt'])
#         trans = infos['rt'][:, :3]  # (T,3)
#         # transform into normalized world space
#         trans = np.einsum('njk,ij->nik', trans[..., None], inv_pose[:3, :3])
#         trans = trans[..., 0] + inv_pose[:3, 3]

#         movement = np.max(np.max(trans, axis=0) - np.min(trans, axis=0))
#         if movement > 1.0:
#             infos["rt"][:, :3] = trans
#             dynamics[dynamic_id] = infos
#             dynamic_id += 1
    

#     # ---- post process dynamics: visibility per camera ----
#     verts = {}
#     rts_per_cam = {cam: {} for cam in cams}  # rts indexed by cam and local frame index

#     for dynamic_id, infos in dynamics.items():
#         lhw = np.array(infos['rt'][0, 3:6])
#         points = get_vertices(lhw)           # (8,3) in object space
#         trans = infos['rt'][:, 0:3]          # (T,3)
#         roty = infos['rt'][:, 6]             # (T,)

#         seq_visible = False

#         for idx, fid_global in enumerate(infos['frame']):
#             rt = np.eye(4)
#             # get FRONT camera's local index for this global frame
#             if FRONT in cams and fid_global in global_to_local[FRONT]:
#                 i_front = global_to_local[FRONT][fid_global]
#                 R_front = c2ws[FRONT][i_front][:3, :3]   # normalized camera->world
#                 cam_roty = SCR.from_matrix(R_front).as_euler('yxz')[0]
#             else:
#                 print('No FRONT camera for frame', fid_global)      

#             rt[:3, :3] = roty_matrix(roty[idx] + cam_roty)  # world yaw only
#             rt[:3, 3] = trans[idx]

#             points_w = (rt[:3, :3] @ points.T).T + rt[:3, 3]   # (8,3) world
#             frame_visible = False

#             for cam in cams:
#                 if fid_global not in global_to_local[cam]:
#                     continue  # that camera didn't produce an image for this record

#                 i_local = global_to_local[cam][fid_global]
#                 c2w = c2ws[cam][i_local]
#                 w2c = np.linalg.inv(c2w)
#                 K = intr[cam][i_local]
#                 h, w = imsize[cam][i_local]

#                 points_cam = (w2c[:3, :3] @ points_w.T).T + w2c[:3, 3]
#                 Z = points_cam[:, 2]
#                 if np.all(Z <= 0):
#                     continue

#                 pts_scr = (K[:3, :3] @ points_cam.T).T + K[:3, 3]
#                 pts_uv = (pts_scr[:, :2] / Z[:, None]).astype(int)

#                 valid_mask = (Z > 0) & \
#                              (pts_uv[:, 0] >= 0) & (pts_uv[:, 1] >= 0) & \
#                              (pts_uv[:, 0] < w) & (pts_uv[:, 1] < h)

#                 if np.any(valid_mask):
#                     frame_visible = True
#                     seq_visible = True

#                     if i_local not in rts_per_cam[cam]:
#                         rts_per_cam[cam][i_local] = {}

#                     rts_per_cam[cam][i_local][dynamic_id] = rt.tolist()
#                     break  # visibility in one cam is enough

#             if frame_visible:
#                 # we already stored rt in rts_per_cam inside the cam loop
#                 pass

#         if seq_visible:
#             verts[dynamic_id] = points.tolist()

    


#     # write meta_data.json
#     meta_data = {
#         "camera_model": "OPENCV",
#         "frames": [],
#         "verts": verts,
#         "inv_pose": inv_pose.tolist()
#     }


#     for cam in cams:
#         N = len(intr[cam])
#         for i in range(N):
#             h, w = imsize[cam][i]
#             info = {
#                 "rgb_path": f"./images/cam_{cam}/{i:06d}.png",
#                 "camtoworld": c2ws[cam][i].tolist(),
#                 "intrinsics": intr[cam][i].tolist(),
#                 "width": int(w),
#                 "height": int(h),
#                 "timestamp": float(timestamps_per_cam[cam][i]),
#                 "dynamics": rts_per_cam[cam].get(i, {})
#             }
#             meta_data["frames"].append(info)

#     with open(os.path.join(save_dir, 'meta_data.json'), 'w') as wf:
#         json.dump(meta_data, wf, indent=2)



##### Already have recording og v2w
import os
import argparse
import open3d as o3d
import numpy as np
from tqdm import tqdm
import json
import cv2
from scipy.spatial.transform import Rotation as SCR
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils

opengl2waymo = np.array([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])

opengl2waymo2 = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])

M = np.array([[ 0.,  0.,  1.],   ## This is the old one
              [0,  -1,  0.],
              [ 1., 0.,  0.]], dtype=float)

# M = np.array([[ 0.,  -1.,  0],    # This is the new one
#               [1,  0,  0.],
#               [ 0., 0.,  1]], dtype=float)

type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_path', type=str, required=True)
    parser.add_argument('-s', '--segment', type=str, required=True)
    parser.add_argument('-c', '--cameras', nargs='+', type=int, required=True)
    parser.add_argument('-o', '--outpath', type=str, required=True)
    parser.add_argument('--downsample', type=float, default=2)
    return parser.parse_args()

def roty_matrix(roty):
    c = np.cos(roty)
    s = np.sin(roty)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def get_vertices(dim, bottom_center=np.array([0.0 ,0.0 ,0.0 ])):
    '''
    dim: length, height, width
    bottom_center: center of bottom face of 3D bounding box

    return: vertices of 3D bounding box (8*3)
    '''
    vertices = bottom_center[None, :].repeat(8, axis=0)
    vertices[:4, 0] = vertices[:4, 0] + dim[0] / 2
    vertices[4:, 0] = vertices[4:, 0] - dim[0] / 2 
    vertices[[0,1,4,5], 1] = vertices[[0,1,4,5], 1]
    vertices[[2,3,6,7], 1] = vertices[[2,3,6,7], 1] - dim[1]
    vertices[[0,2,5,7], 2] = vertices[[0,2,5,7], 2] + dim[2] / 2
    vertices[[1,3,4,6], 2] = vertices[[1,3,4,6], 2] - dim[2] / 2

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
    args = get_opts()

    seq_path = os.path.join(args.base_path, f"{args.segment}")
    datafile = WaymoDataFileReader(seq_path)
    num_frames = len(datafile.get_record_table())
    
    # create folders
    save_dir = args.outpath
    os.makedirs(save_dir, exist_ok=True)
    cams = args.cameras
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
    o3d.io.write_point_cloud(os.path.join(args.outpath, 'ground_lidar.ply'), pcd)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    a, b, c, d = plane_model

    # --- 5) camera calib: use camera_calibrations from frame 0 (calib_frame) ---
    all_campose = {}
    for camera in calib_frame.context.camera_calibrations:
        if camera.name not in args.cameras:
            continue
        c2v = np.array(camera.extrinsic.transform).reshape(4, 4)
        all_campose[camera.name] = c2v @ opengl2waymo

    FRONT = dataset_pb2.CameraName.FRONT
    front_enum = FRONT if FRONT in all_campose else next(iter(all_campose.keys()))
    front_cam_t = all_campose[front_enum][:3, 3]

    height_ground_at_cam = -(a * front_cam_t[0] + b * front_cam_t[1] + d) / c

    front_cam_info = {
        "height": front_cam_t[2] - height_ground_at_cam,
        "rect_mat": None,
    }
    with open(os.path.join(args.outpath, 'front_info.json'), 'w') as f:
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

    cams = args.cameras

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

    combo = np.eye(4)
    combo[:3, :3] = opengl2waymo[:3, :3] @ M

    start_timestamp = None
    global_frame = 0

    # NEW: map global frame index -> frame.pose.transform (4x4)
    frame_v2w_by_global = {}                         # NEW

    # NEW: per-camera list mapping local index -> global frame index
    cam_global_ids = {cam: [] for cam in cams}       # NEW

    for rec in tqdm(it):
        frame = rec

        if not frame.images or not frame.pose.transform:
            continue

        if start_timestamp is None:
            start_timestamp = frame.timestamp_micros / 1e6
        t_abs = frame.timestamp_micros / 1e6
        current_global = global_frame
        global_frame += 1

        # ---- store frame.pose.transform (ego->world) for this global frame ----
        v2w_frame = np.array(frame.pose.transform).reshape(4, 4)   # NEW
        frame_v2w_by_global[current_global] = v2w_frame            # NEW

        # ---- IMAGES (per camera) ----
        for img_pkg in frame.images:
            cam = img_pkg.name
            if cam not in cams:
                continue

            img = cv2.imdecode(np.frombuffer(img_pkg.image, np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            if args.downsample > 1:
                h = int(h // args.downsample)
                w = int(w // args.downsample)
                img = cv2.resize(img, (w, h))

            i_cam = per_cam_idx[cam]
            global_to_local[cam][current_global] = i_cam

            # NEW: record which global frame this local index came from
            cam_global_ids[cam].append(current_global)            # NEW

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
            K[0, 0] = cal.intrinsic[0] / args.downsample
            K[1, 1] = cal.intrinsic[1] / args.downsample
            K[0, 2] = cal.intrinsic[2] / args.downsample
            K[1, 2] = cal.intrinsic[3] / args.downsample
            intr[cam].append(K)

            # Extrinsics (c2v)
            c2v_ctx = np.array(cal.extrinsic.transform).reshape(4, 4)
            extr[cam].append(c2v_ctx)

            per_cam_idx[cam] += 1

        # ---- ego pose for 3D boxes ----
        v2w = v2w_frame  # same as frame.pose.transform we already stored

        # ---- 3D bounding boxes → vehicles dict ----
        for obj in frame.laser_labels:
            type_name = type_list[obj.type]
            height = obj.box.height
            width  = obj.box.width
            length = obj.box.length
            x = obj.box.center_y
            y = obj.box.center_x
            z = obj.box.center_z - height / 2

            t_b2l = np.array([x, y, z, 1]).reshape((4, 1))
            t_b2w = v2w @ t_b2l
            rotation_y = - obj.box.heading  - np.pi / 2

            if type_name in ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
                if obj.id not in vehicles:
                    vehicles[obj.id] = {
                        "rt": [],
                        "timestamp": [],
                        "frame": [],
                    }
                vehicles[obj.id]["rt"].append(np.array(t_b2w[:3, 0].tolist() + [length, height, width, rotation_y]))
                vehicles[obj.id]["timestamp"].append(t_abs)
                vehicles[obj.id]["frame"].append(current_global)


    # ---- Build c2w using aligned lists ----
    for cam in cams:
        c2ws[cam] = []
        for i in range(len(ego_poses[cam])):
            v2w_img = ego_poses[cam][i]
            c2v_ctx = extr[cam][i]
            c2w = v2w_img @ c2v_ctx @ opengl2waymo
            c2ws[cam].append(c2w)
        c2ws[cam] = np.stack(c2ws[cam], axis=0)

    FRONT = dataset_pb2.CameraName.FRONT
    inv_pose = np.linalg.inv(c2ws[FRONT][0])

    # normalize all c2ws so FRONT[0] becomes identity
    for cam in cams:
        poses = c2ws[cam]  # (N,4,4)
        poses = np.einsum('njk,ij->nik', poses, inv_pose)
        c2ws[cam] = poses

    # ---- filter dynamic vehicles ----
    dynamic_id = 0
    for objid, infos in vehicles.items():
        infos['rt'] = np.stack(infos['rt'])
        trans = infos['rt'][:, :3]  # (T,3)
        # transform into normalized world space
        trans = np.einsum('njk,ij->nik', trans[..., None], inv_pose[:3, :3])
        trans = trans[..., 0] + inv_pose[:3, 3]

        movement = np.max(np.max(trans, axis=0) - np.min(trans, axis=0))
        if movement > 1.0:
            infos["rt"][:, :3] = trans
            dynamics[dynamic_id] = infos
            dynamic_id += 1
    
    # ---- post process dynamics: visibility per camera ----
    verts = {}
    rts_per_cam = {cam: {} for cam in cams}

    for dynamic_id, infos in dynamics.items():
        lhw = np.array(infos['rt'][0, 3:6])
        points = get_vertices(lhw)
        trans = infos['rt'][:, 0:3]
        roty = infos['rt'][:, 6]

        seq_visible = False

        for idx, fid_global in enumerate(infos['frame']):
            rt = np.eye(4)
            if FRONT in cams and fid_global in global_to_local[FRONT]:
                i_front = global_to_local[FRONT][fid_global]
                R_front = c2ws[FRONT][i_front][:3, :3]
                cam_roty = SCR.from_matrix(R_front).as_euler('yxz')[0]
            else:
                print('No FRONT camera for frame', fid_global)      

            rt[:3, :3] = roty_matrix(roty[idx] + cam_roty)
            rt[:3, 3] = trans[idx]

            points_w = (rt[:3, :3] @ points.T).T + rt[:3, 3]
            frame_visible = False

            for cam in cams:
                if fid_global not in global_to_local[cam]:
                    continue

                i_local = global_to_local[cam][fid_global]
                c2w = c2ws[cam][i_local]
                w2c = np.linalg.inv(c2w)
                K = intr[cam][i_local]
                h, w = imsize[cam][i_local]

                points_cam = (w2c[:3, :3] @ points_w.T).T + w2c[:3, 3]
                Z = points_cam[:, 2]
                if np.all(Z <= 0):
                    continue

                pts_scr = (K[:3, :3] @ points_cam.T).T + K[:3, 3]
                pts_uv = (pts_scr[:, :2] / Z[:, None]).astype(int)

                valid_mask = (Z > 0) & \
                             (pts_uv[:, 0] >= 0) & (pts_uv[:, 1] >= 0) & \
                             (pts_uv[:, 0] < w) & (pts_uv[:, 1] < h)

                if np.any(valid_mask):
                    frame_visible = True
                    seq_visible = True

                    if i_local not in rts_per_cam[cam]:
                        rts_per_cam[cam][i_local] = {}

                    rts_per_cam[cam][i_local][dynamic_id] = rt.tolist()
                    break

            if frame_visible:
                pass

        if seq_visible:
            verts[dynamic_id] = points.tolist()

    # write meta_data.json
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

            # NEW: get the global frame index for this local camera index
            global_id = cam_global_ids[cam][i]                         # NEW
            frame_v2w = frame_v2w_by_global.get(global_id, None)       # NEW

            info = {
                "rgb_path": f"./images/cam_{cam}/{i:06d}.png",
                "camtoworld": c2ws[cam][i].tolist(),
                "intrinsics": intr[cam][i].tolist(),
                "width": int(w),
                "height": int(h),
                "timestamp": float(timestamps_per_cam[cam][i]),
                "dynamics": rts_per_cam[cam].get(i, {}),
                # NEW: raw ego->world pose at image timestamp (img_pkg.pose.transform)
                "ego_pose_image": ego_poses[cam][i].tolist()
            }

            # NEW: raw ego->world pose from frame.pose.transform (common for this global frame)
            if frame_v2w is not None:
                info["ego_pose_frame"] = frame_v2w.tolist()            # NEW

            meta_data["frames"].append(info)

    with open(os.path.join(save_dir, 'meta_data.json'), 'w') as wf:
        json.dump(meta_data, wf, indent=2)

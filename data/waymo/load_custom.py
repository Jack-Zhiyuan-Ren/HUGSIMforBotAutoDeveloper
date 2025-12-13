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

M = np.array([[ 0.,  0.,  1.],
              [0,  -1,  0.],
              [ 1., 0.,  0.]], dtype=float)

type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

    

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_path', type=str, required=True)
    parser.add_argument('-s', '--segment', type=str, required=True)
    parser.add_argument('-c', '--cameras', nargs='+', type=int, required=True)
    parser.add_argument('-o', '--outpath', type=str, required=True)
    parser.add_argument('--downsample', type=float, default=2) # don't downsamlpe. 
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

    it = iter(datafile)
    try:
        frame0 = next(it)
    except StopIteration:
        raise RuntimeError("Empty TFRecord: no frames found")

    # Camera & laser calibrations only live in frame 0 for your custom data
    ctx_cam_cals = {cal.name: cal for cal in frame0.context.camera_calibrations}
    ctx_laser_cals = {cal.name: cal for cal in frame0.context.laser_calibrations}
    


    # convenience
    FRONT = dataset_pb2.CameraName.FRONT
    cams = args.cameras
        

    all_campose = {}
    for name, camera in ctx_cam_cals.items():
        if name not in cams:
            continue
        c2v = np.array(camera.extrinsic.transform).reshape(4, 4)
    
        # Combine opengl2waymo and M (rotation-only)
        combo = np.eye(4)
        combo[:3, :3] = opengl2waymo[:3, :3] @ M
    
        all_campose[name] = c2v @ combo

    
    
    front_enum = FRONT if FRONT in all_campose else next(iter(all_campose.keys()))
    front_cam_t = all_campose[front_enum][:3, 3]
    
    # If you want a simple height estimate without LiDAR plane fit:
    cam_height = float(front_cam_t[2])  # crude; or use a fixed 1.6 as fallback
    front_cam_info = {"height": cam_height, "rect_mat": None}
    with open(os.path.join(args.outpath, 'front_info.json'), 'w') as f:
        json.dump(front_cam_info, f, indent=2)





    ##########################################################################
    #                     Read all frames infos                         #
    ##########################################################################
    # NEW: per-camera aligned storage (indices start at 0 for first saved image of each cam)
    per_cam_idx         = {cam: 0    for cam in cams}
    ego_poses           = {cam: []   for cam in cams}  # per-image v2w at this camera timestamp
    intr                = {cam: []   for cam in cams}  # per-image intrinsics
    extr                = {cam: []   for cam in cams}  # per-image c2v (from context; append once per saved image)
    imsize              = {cam: []   for cam in cams}  # per-image (h, w)
    timestamps_per_cam  = {cam: []   for cam in cams}  # per-image timestamp (seconds)

    vehicles = {}
    dynamics = {}
    c2ws                = {}                            # filled after the loop
    global_to_local = {cam: {} for cam in cams}

    # Keep your combo if you really need both transforms; otherwise try only opengl2waymo
    combo = np.eye(4)
    combo[:3, :3] = opengl2waymo[:3, :3] @ M
    
    start_timestamp = None
    global_frame = 0
    for rec in tqdm(it):
        frame = rec  # 'it' already yields parsed frames from WaymoDataFileReader
    
        # Skip frames without images or pose (context-only / sparse cases)
        if not frame.images or not frame.pose.transform:
            continue

                
        if start_timestamp is None:
            start_timestamp = frame.timestamp_micros / 1e6
        t_abs = frame.timestamp_micros / 1e6  # absolute seconds; use relative if you prefer
        current_global = global_frame
        global_frame += 1
        # ---- IMAGES (per camera) ----
        for img_pkg in frame.images:
            cam = img_pkg.name
            if cam not in cams:
                continue
    
            # Decode & (optional) downsample
            img = cv2.imdecode(np.frombuffer(img_pkg.image, np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            if args.downsample > 1:
                h = int(h // args.downsample); w = int(w // args.downsample)
                img = cv2.resize(img, (w, h))
    
            # Per-camera index drives file name AND array indices
            i_cam = per_cam_idx[cam]
            global_to_local[cam][current_global] = i_cam
            out_png = os.path.join(save_dir, "images", f"cam_{cam}", f"{i_cam:06d}.png")
            cv2.imwrite(out_png, img)
    
            # Per-image bookkeeping (aligned across lists for this cam)
            imsize[cam].append((h, w))
            timestamps_per_cam[cam].append(t_abs)
    
            v2w_img = np.array(img_pkg.pose.transform).reshape(4, 4)  # vehicle->world at camera timestamp
            ego_poses[cam].append(v2w_img)
    
            # Intrinsics from cached context (append once per saved image)
            cal = ctx_cam_cals[cam]
            K = np.eye(4, dtype=float)
            K[0, 0] = cal.intrinsic[0] / args.downsample
            K[1, 1] = cal.intrinsic[1] / args.downsample
            K[0, 2] = cal.intrinsic[2] / args.downsample
            K[1, 2] = cal.intrinsic[3] / args.downsample
            intr[cam].append(K)
    
            # Extrinsics (c2v) from cached context (append once per saved image)
            c2v_ctx = np.array(cal.extrinsic.transform).reshape(4, 4)
            extr[cam].append(c2v_ctx)
    
            # advance only when we actually saved an image for this camera
            per_cam_idx[cam] += 1
            

        
            
        # ego pose
        v2w = np.array(frame.pose.transform).reshape(4,4)

        # current_global = global_frame
        # global_frame += 1
        # 3d bbox
        for obj in frame.laser_labels:
            type_name = type_list[obj.type]
            height = obj.box.height  # up/down
            width = obj.box.width  # left/right
            length = obj.box.length  # front/back
            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2
            t_b2l = np.array([x,y,z,1]).reshape((4,1))
            t_b2w = v2w @ t_b2l
            rotation_y = -obj.box.heading - np.pi / 2
            if type_name in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
                if obj.id not in vehicles:
                    vehicles[obj.id] = {
                        "rt": [],
                        "timestamp": [],
                        "frame": [],
                    }
                vehicles[obj.id]['rt'].append(np.array(t_b2w[:3, 0].tolist() + [length, height, width, rotation_y]))
                vehicles[obj.id]["timestamp"].append(t_abs)
                vehicles[obj.id]['frame'].append(current_global)



    # Build c2w using aligned lists
    for cam in cams:
        c2ws[cam] = []
        for i in range(len(ego_poses[cam])):
            v2w = ego_poses[cam][i]
            c2v = extr[cam][i]
            c2w = v2w @ c2v @ combo
            c2ws[cam].append(c2w)
        c2ws[cam] = np.stack(c2ws[cam], axis=0)




                
    FRONT = dataset_pb2.CameraName.FRONT
    inv_pose = np.linalg.inv(c2ws[FRONT][0])


    for cam in cams:
        poses = c2ws[cam]  # (N,4,4)
        # Correct: left-multiply inv_pose @ pose
        c2ws[cam] = np.einsum('ij,njk->nik', inv_pose, poses)

    
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

    # --- REPLACE your current "post process dynamic infos" block with this ---
    verts, rts_per_cam = {}, {cam: {} for cam in cams}  # rts grouped per cam/local index
    
    for dynamic_id, infos in dynamics.items():
        lhw = np.array(infos['rt'][0, 3:6])
        points = get_vertices(lhw)              # (8,3) in object space
        trans  = infos['rt'][:, 0:3]            # (T,3) world translations (already normalized earlier)
        roty   = infos['rt'][:, 6]              # (T,) yaw in world
        seq_visible = False
    
        for idx, fid_global in enumerate(infos['frame']):   # fid_global = your current_global
            # Build box world transform at this time step
            rt = np.eye(4)
            # Camera yaw term from extrinsics is not appropriate here—use world yaw only
            # If you do need a camera-relative yaw, compute it from c2ws when you project below.
            rt[:3, :3] = roty_matrix(roty[idx])  # world yaw only
            rt[:3, 3]  = trans[idx]
    
            points_w = (rt[:3, :3] @ points.T).T + rt[:3, 3]   # (8,3) world
            frame_visible = False
    
            # Try projection onto each requested camera at this global frame
            for cam in cams:
                if fid_global not in global_to_local[cam]:
                    continue  # that camera didn't produce an image for this record
    
                i_local = global_to_local[cam][fid_global]   # per-cam index
                c2w = c2ws[cam][i_local]
                w2c = np.linalg.inv(c2w)
                K   = intr[cam][i_local]
                h, w = imsize[cam][i_local]
    
                points_cam = (w2c[:3, :3] @ points_w.T).T + w2c[:3, 3]      # (8,3)
                Z = points_cam[:, 2]
                if np.all(Z <= 0):
                    continue
    
                pts_scr = (K[:3, :3] @ points_cam.T).T + K[:3, 3]           # (8,3)
                pts_uv  = (pts_scr[:, :2] / Z[:, None]).astype(int)         # (8,2)
                valid_mask = (Z > 0) & (pts_uv[:,0] >= 0) & (pts_uv[:,1] >= 0) & \
                             (pts_uv[:,0] < w) & (pts_uv[:,1] < h)
    
                if np.any(valid_mask):
                    frame_visible = True
                    seq_visible   = True
    
                    if i_local not in rts_per_cam[cam]:
                        rts_per_cam[cam][i_local] = {}
    
                    # Store the world-space rt (or camera-space if you prefer)
                    rts_per_cam[cam][i_local][dynamic_id] = rt.tolist()
                    break  # one cam visibility is enough
    
        if seq_visible:
            verts[dynamic_id] = points.tolist()
    # Ensure list lengths are consistent per cam (truncate to the minimum)



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
            info = {
                "rgb_path": f"./images/cam_{cam}/{i:06d}.png",
                "camtoworld": c2ws[cam][i].tolist(),
                "intrinsics": intr[cam][i].tolist(),
                "width": int(w),
                "height": int(h),
                "timestamp": float(timestamps_per_cam[cam][i]),
                "dynamics": rts_per_cam[cam].get(i, {})    # <-- per-cam aligned
            }
            meta_data["frames"].append(info)
        


    with open(os.path.join(save_dir, 'meta_data.json'), 'w') as wf:
        json.dump(meta_data, wf, indent=2)


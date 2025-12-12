# import argparse
# import glob
# import os
# import json
# import numpy as np
# import torch
# from tqdm import tqdm
# from unidepth.models import UniDepthV2
# from PIL import Image
# import json


# def get_opts():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--out', type=str, required=True)
#     return parser.parse_args()

# if __name__ == '__main__':
#     args = get_opts()
    
#     print('loading depth model...')
#     model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", force_download=True)
#     model = model.to("cuda")
#     model.eval()
#     print("Depth model loaded")
    
#     os.makedirs(os.path.join(args.out, 'depth'), exist_ok=True)
#     for cam_pth in glob.glob(os.path.join(args.out, 'images', '*')):
#         cam = os.path.basename(cam_pth)
#         os.makedirs(os.path.join(args.out, 'depth', cam), exist_ok=True)
    
#     with open(os.path.join(args.out, 'meta_data.json')) as f:
#         meta_data = json.load(f)
    
#     for frame in tqdm(meta_data['frames']):
#         im_path = os.path.join(args.out, frame['rgb_path'])
#         K = np.array(frame['intrinsics'])
#         K = torch.from_numpy(K[:3, :3]).float().cuda()
#         image = torch.from_numpy(np.array(Image.open(im_path))).permute(2, 0, 1)
#         prediction = model.infer(image, K)
#         depth = prediction["depth"][0][0].detach().cpu()  # Depth in [m].
        
#         depth_path = os.path.join(
#             args.out,
#             im_path.replace("images", "depth")
#             .replace("./", "")
#             .replace(".jpg", ".pt")
#             .replace(".png", ".pt"),
#         )
        
#         torch.save(depth, depth_path)


#This version accounts for distortion

import argparse
import glob
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from unidepth.models import UniDepthV2
from PIL import Image
import cv2  # <-- added
import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()

# ---- per-camera intrinsics with distortion (downsampled) ----
cam1_intr_ds = np.array(
    [[786.0027, 785.79835, 480.9809, 272.695, 0.0, 0.0, 0.0, 0.0, 0.0]],
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

# map "cam_1"/"cam_2"/"cam_3" -> intrinsics vector
CAM_INTR_MAP = {
    "cam_1": cam1_intr_ds[0],
    "cam_2": cam2_intr_ds[0],
    "cam_3": cam3_intr_ds[0],
}

if __name__ == '__main__':
    args = get_opts()
    
    print('loading depth model...')
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", force_download=True)
    model = model.to("cuda")
    model.eval()
    print("Depth model loaded")
    
    os.makedirs(os.path.join(args.out, 'depth'), exist_ok=True)
    for cam_pth in glob.glob(os.path.join(args.out, 'images', '*')):
        cam = os.path.basename(cam_pth)
        os.makedirs(os.path.join(args.out, 'depth', cam), exist_ok=True)
    
    with open(os.path.join(args.out, 'meta_data.json')) as f:
        meta_data = json.load(f)
    
    for frame in tqdm(meta_data['frames']):
        im_path = os.path.join(args.out, frame['rgb_path'])  # e.g. ./images/cam_2/000008.png

        # -------- pick intrinsics + distortion from cam name --------
        cam_name = os.path.basename(os.path.dirname(frame['rgb_path']))  # "cam_1", "cam_2", "cam_3"

        if cam_name in CAM_INTR_MAP:
            intr_vec = CAM_INTR_MAP[cam_name]
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = intr_vec
            K_np = np.array([[fx, 0.0, cx],
                             [0.0, fy, cy],
                             [0.0, 0.0, 1.0]], dtype=np.float32)
            dist_np = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        else:
            # fallback: use K from meta_data with no distortion
            K_full = np.array(frame['intrinsics'], dtype=np.float32)
            K_np = K_full[:3, :3]
            dist_np = np.zeros(5, dtype=np.float32)

        # -------- load and undistort image --------
        img_np = np.array(Image.open(im_path))  # H, W, 3 (RGB)
        h, w = img_np.shape[:2]

        # Compute optimal new camera matrix for undistorted image
        newK, roi = cv2.getOptimalNewCameraMatrix(K_np, dist_np, (w, h), alpha=0)

        # Undistort image using K and distortion
        img_undist = cv2.undistort(img_np, K_np, dist_np, None, newK)

        # Convert to torch tensor (C,H,W)
        # image = torch.from_numpy(img_undist).permute(2, 0, 1).float().cuda()
        image = torch.from_numpy(img_np).permute(2, 0, 1).float().cuda() #Because right now the image already come undistorted in load_custom15

        # Use newK as the intrinsics for the undistorted image
        K_torch = torch.from_numpy(newK).float().cuda()

        # -------- depth inference --------
        prediction = model.infer(image, K_torch)
        depth = prediction["depth"][0][0].detach().cpu()  # Depth in [m].
        
        depth_path = os.path.join(
            args.out,
            im_path.replace("images", "depth")
            .replace("./", "")
            .replace(".jpg", ".pt")
            .replace(".png", ".pt"),
        )
        
        torch.save(depth, depth_path)


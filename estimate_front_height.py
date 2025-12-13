import os
import json
import numpy as np
import torch
from imageio.v2 import imread
from tqdm import tqdm

# ===========================================
# CONFIG – EDIT THIS
# ===========================================
# OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"
OUT_DIR = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1083056"
DATASET_TYPE = "waymo"   # "waymo", "nuscenes", "pandaset", "kitti360"
MAX_FRAMES = 50          # limit frames for speed (None for all)
Z_MIN = 3.0              # m, ignore very close noisy depth
Z_MAX = 60.0             # m, ignore extremely far
# ===========================================


def is_front_cam(frame, datatype):
    rgb_path = frame["rgb_path"]
    if datatype == "waymo":
        return "/cam_1/" in rgb_path
    elif datatype == "nuscenes":
        return "/CAM_FRONT/" in rgb_path
    elif datatype == "pandaset":
        return "/front_camera/" in rgb_path
    elif datatype == "kitti360":
        return "/cam_0/" in rgb_path
    else:
        raise NotImplementedError(f"Unknown datatype: {datatype}")


def main():
    out_dir = OUT_DIR
    datatype = DATASET_TYPE

    meta_path = os.path.join(out_dir, "meta_data.json")
    with open(meta_path, "r") as rf:
        meta_data = json.load(rf)

    frames = meta_data["frames"]

    # pick only front-camera frames
    front_frames = [f for f in frames if is_front_cam(f, datatype)]
    if len(front_frames) == 0:
        print("No front-camera frames found – check is_front_cam logic.")
        return

    if MAX_FRAMES is not None:
        front_frames = front_frames[:MAX_FRAMES]

    all_y = []

    for frame in tqdm(front_frames, desc="Processing front frames"):
        intrinsic = np.array(frame["intrinsics"])
        cx, cy, fx, fy = intrinsic[0, 2], intrinsic[1, 2], intrinsic[0, 0], intrinsic[1, 1]

        rgb_path = frame["rgb_path"]
        rgb_abs = os.path.join(out_dir, rgb_path)
        if not os.path.isfile(rgb_abs):
            print("Missing image:", rgb_abs)
            continue

        # depth
        depth_path = os.path.join(
            out_dir,
            rgb_path.replace("images", "depth")
                   .replace("./", "")
                   .replace(".jpg", ".pt")
                   .replace(".png", ".pt"),
        )
        if not os.path.isfile(depth_path):
            print("Missing depth:", depth_path)
            continue

        depth = torch.load(depth_path).numpy()  # (H,W)

        # semantics
        smts_path = os.path.join(
            out_dir,
            rgb_path.replace("images", "semantics")
                   .replace("./", "")
                   .replace(".jpg", ".npy")
                   .replace(".png", ".npy"),
        )
        if not os.path.isfile(smts_path):
            print("Missing semantics:", smts_path)
            continue

        smts = np.load(smts_path)   # (H,W)

        H, W = depth.shape

        # pixel grid
        xs = np.arange(0, W)
        ys = np.arange(0, H)
        xx, yy = np.meshgrid(xs, ys)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        d_flat = depth.reshape(-1)
        smts_flat = smts.reshape(-1)

        # unproject in *camera* frame
        X = (xx - cx) * d_flat / fx
        Y = (yy - cy) * d_flat / fy
        Z = d_flat

        # ground semantics: smts <= 1
        mask_ground = smts_flat <= 1
        # depth range filter
        mask_depth = (Z > Z_MIN) & (Z < Z_MAX)

        mask = mask_ground & mask_depth

        if np.sum(mask) == 0:
            continue

        Y_ground = Y[mask]
        all_y.append(Y_ground)

    if len(all_y) == 0:
        print("No valid ground points collected – check paths / masks.")
        return

    all_y = np.concatenate(all_y)

    print("\n===== Empirical ground height (camera Y) =====")
    print(f"Num ground samples: {all_y.shape[0]}")
    print(f"Y mean   : {np.mean(all_y):.4f}")
    print(f"Y median : {np.median(all_y):.4f}")
    print(f"Y std    : {np.std(all_y):.4f}")

    # small extra: show 5th/95th percentile
    p5  = np.percentile(all_y, 5)
    p95 = np.percentile(all_y, 95)
    print(f"Y 5th pct: {p5:.4f}, 95th pct: {p95:.4f}")


if __name__ == "__main__":
    main()

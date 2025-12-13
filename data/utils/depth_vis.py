# 


#### This version below includes readouts

# import os
# import glob

# import torch
# import numpy as np
# import cv2
# from tqdm import tqdm


# # ================== HARD-CODED CONFIG ==================
# ROOT = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"
# # ROOT = "Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1083056"
# VIS_DIRNAME = "depth_color"   # output folder name under ROOT
# # =======================================================


# def normalize_depth(depth_np):
#     """
#     Normalize depth to [0, 255] for visualization.

#     depth_np: (H, W) float32, in meters.
#     """
#     # Keep only finite values
#     valid = np.isfinite(depth_np)
#     if not np.any(valid):
#         # Fallback: everything invalid, just zeros
#         return np.zeros_like(depth_np, dtype=np.uint8)

#     dmin = depth_np[valid].min()
#     dmax = depth_np[valid].max()

#     if dmax <= dmin + 1e-8:
#         # Almost constant depth, avoid division by zero
#         return np.zeros_like(depth_np, dtype=np.uint8)

#     vis = (depth_np - dmin) / (dmax - dmin)
#     vis = np.clip(vis, 0.0, 1.0)
#     vis = (vis * 255.0).astype(np.uint8)
#     return vis


# if __name__ == "__main__":
#     root = ROOT
#     depth_root = os.path.join(root, "depth")
#     vis_root = os.path.join(root, VIS_DIRNAME)
#     stats_path = os.path.join(root, "depth_stats.txt")

#     if not os.path.isdir(depth_root):
#         raise FileNotFoundError(f"Depth directory not found: {depth_root}")

#     print(f"Reading depth .pt files from: {depth_root}")
#     print(f"Saving colored PNGs under: {vis_root}")
#     print(f"Saving depth stats to: {stats_path}")

#     # Find all .pt files recursively under ROOT/depth
#     pt_files = glob.glob(os.path.join(depth_root, "**", "*.pt"), recursive=True)
#     pt_files = sorted(pt_files)

#     if len(pt_files) == 0:
#         print("No .pt files found. Did you run the UniDepth script first?")
#         raise SystemExit(0)

#     # Open stats file and write header
#     with open(stats_path, "w") as sf:
#         sf.write("file,dtype,shape,min,max\n")

#         for pt_path in tqdm(pt_files, desc="Converting depth to color PNG"):
#             # Load depth tensor
#             depth = torch.load(pt_path)        # shape ~ (H, W)
#             if not isinstance(depth, torch.Tensor):
#                 raise TypeError(f"Expected a torch.Tensor in {pt_path}, got {type(depth)}")

#             # Basic stats
#             d_type = depth.dtype
#             d_shape = tuple(depth.shape)
#             d_min = depth.min().item()
#             d_max = depth.max().item()

#             # Print to console (as you requested)
#             print(pt_path)
#             print(type(depth), depth.shape, depth.dtype)
#             print(d_min, d_max)  # depth range in meters

#             # Save stats to file (relative path for readability)
#             rel_path = os.path.relpath(pt_path, root)
#             sf.write(f"{rel_path},{d_type},{d_shape},{d_min},{d_max}\n")

#             # Convert to numpy for visualization
#             depth_np = depth.cpu().numpy()

#             # Normalize to [0, 255]
#             vis = normalize_depth(depth_np)    # uint8, (H, W)

#             # Apply colormap (PLASMA, JET, etc.)
#             vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)

#             # Build output path: replace "depth" with VIS_DIRNAME and ".pt" with ".png"
#             rel_path_from_depth = os.path.relpath(pt_path, depth_root)  # e.g. cam_1/00001.pt
#             out_path = os.path.join(vis_root, rel_path_from_depth)
#             out_path = out_path.replace(".pt", ".png")

#             # Make sure directory exists
#             os.makedirs(os.path.dirname(out_path), exist_ok=True)

#             # Save PNG
#             cv2.imwrite(out_path, vis_color)

#     print("Done! Colored depth images saved under:")
#     print(f"  {vis_root}")
#     print("Depth stats saved to:")
#     print(f"  {stats_path}")

# This version below includes mean and median in stats

import os
import glob

import torch
import numpy as np
import cv2
from tqdm import tqdm


# ================== HARD-CODED CONFIG ==================
# ROOT = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6"
ROOT = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1083056"
VIS_DIRNAME = "depth_color"   # output folder name under ROOT
# =======================================================


def normalize_depth(depth_np):
    """
    Normalize depth to [0, 255] for visualization.

    depth_np: (H, W) float32, in meters.
    """
    # Keep only finite values
    valid = np.isfinite(depth_np)
    if not np.any(valid):
        # Fallback: everything invalid, just zeros
        return np.zeros_like(depth_np, dtype=np.uint8)

    dmin = depth_np[valid].min()
    dmax = depth_np[valid].max()

    if dmax <= dmin + 1e-8:
        # Almost constant depth, avoid division by zero
        return np.zeros_like(depth_np, dtype=np.uint8)

    vis = (depth_np - dmin) / (dmax - dmin)
    vis = np.clip(vis, 0.0, 1.0)
    vis = (vis * 255.0).astype(np.uint8)
    return vis


if __name__ == "__main__":
    root = ROOT
    depth_root = os.path.join(root, "depth")
    vis_root = os.path.join(root, VIS_DIRNAME)
    stats_path = os.path.join(root, "depth_stats.txt")

    if not os.path.isdir(depth_root):
        raise FileNotFoundError(f"Depth directory not found: {depth_root}")

    print(f"Reading depth .pt files from: {depth_root}")
    print(f"Saving colored PNGs under: {vis_root}")
    print(f"Saving depth stats to: {stats_path}")

    # Find all .pt files recursively under ROOT/depth
    pt_files = glob.glob(os.path.join(depth_root, "**", "*.pt"), recursive=True)
    pt_files = sorted(pt_files)

    if len(pt_files) == 0:
        print("No .pt files found. Did you run the UniDepth script first?")
        raise SystemExit(0)

    # Open stats file and write header
    with open(stats_path, "w") as sf:
        sf.write("file,dtype,shape,min,max,mean,median\n")

        for pt_path in tqdm(pt_files, desc="Converting depth to color PNG"):
            # Load depth tensor
            depth = torch.load(pt_path)        # shape ~ (H, W)
            if not isinstance(depth, torch.Tensor):
                raise TypeError(f"Expected a torch.Tensor in {pt_path}, got {type(depth)}")

            # Basic stats
            d_type = depth.dtype
            d_shape = tuple(depth.shape)
            d_min = depth.min().item()
            d_max = depth.max().item()
            d_mean = depth.mean().item()
            d_median = depth.median().item()

            # Print to console
            print(pt_path)
            print(type(depth), depth.shape, depth.dtype)
            print(f"min={d_min}, max={d_max}, mean={d_mean}, median={d_median}")

            # Save stats to file (relative path for readability)
            rel_path = os.path.relpath(pt_path, root)
            sf.write(
                f"{rel_path},{d_type},{d_shape},{d_min},{d_max},{d_mean},{d_median}\n"
            )

            # Convert to numpy for visualization
            depth_np = depth.cpu().numpy()

            # Normalize to [0, 255]
            vis = normalize_depth(depth_np)    # uint8, (H, W)

            # Apply colormap (PLASMA, JET, etc.)
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)

            # Build output path: replace "depth" with VIS_DIRNAME and ".pt" with ".png"
            rel_path_from_depth = os.path.relpath(pt_path, depth_root)  # e.g. cam_1/00001.pt
            out_path = os.path.join(vis_root, rel_path_from_depth)
            out_path = out_path.replace(".pt", ".png")

            # Make sure directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Save PNG
            cv2.imwrite(out_path, vis_color)

    print("Done! Colored depth images saved under:")
    print(f"  {vis_root}")
    print("Depth stats saved to:")
    print(f"  {stats_path}")


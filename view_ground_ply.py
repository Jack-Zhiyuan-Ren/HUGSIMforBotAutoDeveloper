


# def main():
#     # Load point clouds
#     pcd_waymo = o3d.io.read_point_cloud(WAYMO_PLY)
#     pcd_custom = o3d.io.read_point_cloud(CUSTOM_PLY)

#     print("Waymo cloud:", pcd_waymo)
#     print("Custom cloud:", pcd_custom)

#     # Give them different colors
#     pcd_waymo.paint_uniform_color([0, 0, 1])   # blue
#     pcd_custom.paint_uniform_color([1, 0, 0])  # red

#     # Translate one so they don't overlap (purely for visualization)
#     pcd_custom_translated = pcd_custom.translate((50, 0, 0), relative=False)

#     # Visualize
#     o3d.visualization.draw_geometries(
#         [pcd_waymo, pcd_custom_translated],
#         window_name="Waymo (blue) vs Custom (red)",
#         width=1600,
#         height=900,
#     )

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import os

# TODO: set these paths
WAYMO_PLY = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/1002394/ground_points3d.ply"
# WAYMO_PLY = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/Reconstruction/1006130/point_cloud_vis/iteration_7000/ground.ply"
CUSTOM_PLY = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/waymo_data/311238_part_0_100_v6/ground_points3d.ply"
# CUSTOM_PLY = "/workspace/Jack/HUGSIM/data/samplepoints_in_data/Reconstruction/311238_part_0_100_v6/point_cloud_vis/iteration_14000/ground.ply"
OUT_DIR = "./ground_ply_analysis_v11"
# OUT_DIR = "./points3d_ply_analysis_v3"

def summarize_cloud(pcd, name):
    pts = np.asarray(pcd.points)
    print(f"\n=== {name} ===")
    print(f"Num points: {pts.shape[0]}")

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    means = pts.mean(axis=0)
    stds = pts.std(axis=0)

    print(f"min  (x,y,z): {mins}")
    print(f"max  (x,y,z): {maxs}")
    print(f"mean (x,y,z): {means}")
    print(f"std  (x,y,z): {stds}")
    print(f"z range: {mins[2]} to {maxs[2]} (span {maxs[2] - mins[2]})")

    # Histogram of z (height)
    plt.figure(figsize=(6, 4))
    plt.hist(pts[:, 2], bins=80)
    plt.xlabel("z (height)")
    plt.ylabel("count")
    plt.title(f"{name} - z histogram")
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUT_DIR, f"{name}_z_hist.png"))
    plt.close()


def scatter_projection(pts, name, plane="xz", max_points=50000):
    """
    Save a 2D scatter projection:
      plane="xz" -> x horizontal, z vertical
      plane="xy" -> x horizontal, y vertical
      plane="yz" -> y horizontal, z vertical
    """
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    if plane == "xz":
        a, b = pts[:, 0], pts[:, 2]
        xlabel, ylabel = "x", "z"
    elif plane == "xy":
        a, b = pts[:, 0], pts[:, 1]
        xlabel, ylabel = "x", "y"
    elif plane == "yz":
        a, b = pts[:, 1], pts[:, 2]
        xlabel, ylabel = "y", "z"
    else:
        raise ValueError("plane must be one of: xz, xy, yz")

    plt.figure(figsize=(6, 6))
    plt.scatter(a, b, s=1, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{name} - {plane} projection")
    plt.axis("equal")
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fname = os.path.join(OUT_DIR, f"{name}_{plane}_proj.png")
    plt.savefig(fname)
    plt.close()


def main():
    print("Loading point clouds...")
    pcd_waymo = o3d.io.read_point_cloud(WAYMO_PLY)
    pcd_custom = o3d.io.read_point_cloud(CUSTOM_PLY)

    print("Waymo cloud:", pcd_waymo)
    print("Custom cloud:", pcd_custom)

    # Convert to numpy
    pts_waymo = np.asarray(pcd_waymo.points)
    pts_custom = np.asarray(pcd_custom.points)

    # Summaries + z histogram
    summarize_cloud(pcd_waymo, "waymo")
    summarize_cloud(pcd_custom, "custom")

    # 2D projections: xz (forward vs height), xy (plan view), yz
    for name, pts in [("waymo", pts_waymo), ("custom", pts_custom)]:
        scatter_projection(pts, name, plane="xz")
        scatter_projection(pts, name, plane="xy")
        scatter_projection(pts, name, plane="yz")

    print(f"\nSaved plots to: {OUT_DIR}")
    print("Check the PNGs (z_hist + projections) to compare thickness/shape.")


if __name__ == "__main__":
    main()

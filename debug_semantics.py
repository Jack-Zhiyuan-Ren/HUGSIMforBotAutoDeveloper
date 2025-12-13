#!/usr/bin/env python3
import os
import argparse
import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from scene import load_cameras
from utils.dataset import HUGSIM_dataset, hugsim_collate


def analyze_semantics(cfg, num_images=5, ground_ids=None):
    """
    Debug semantic labels:
      - prints unique IDs in gt_semantic
      - prints fraction of pixels treated as ground by:
          (a) gt_semantic <= 1   (current training rule)
          (b) optional custom ground_ids (if provided)
    """
    # Load cameras and dataset the same way as in training
    train_cams, test_cams, _ = load_cameras(cfg, cfg.data_type, True)
    train_dataset = HUGSIM_dataset(train_cams, cfg.data_type)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=hugsim_collate,
    )

    if ground_ids is not None:
        ground_ids = torch.tensor(ground_ids, dtype=torch.long)

    print(f"Analyzing up to {num_images} training images...\n")

    for idx, batch in enumerate(train_loader):
        if idx >= num_images:
            break

        view_iid, prev_iid, gt_image, gt_semantic, gt_flow, gt_depth, mask = batch

        # gt_semantic shape in training is (B, H, W); B = 1 here
        # keep it as a tensor on CPU
        if isinstance(gt_semantic, (list, tuple)):
            gt_semantic = gt_semantic[0]
        gt_semantic = gt_semantic.squeeze(0)  # (H, W)

        cam = train_cams[view_iid]
        img_name = getattr(cam, "image_name", f"index_{int(view_iid)}")

        print(f"--- Image {idx} ({img_name}) ---")
        unique_ids = torch.unique(gt_semantic)
        print("Unique semantic IDs:", unique_ids.tolist())

        # (a) Current training rule: gt_semantic <= 1
        valid_mask_le1 = (gt_semantic <= 1)
        frac_le1 = valid_mask_le1.float().mean().item()
        print("Ground fraction with (gt_semantic <= 1): "
              f"{frac_le1:.4f}")

        # (b) Optional custom ground IDs
        if ground_ids is not None:
            # torch.isin is available in recent torch versions
            valid_mask_custom = torch.isin(gt_semantic, ground_ids)
            frac_custom = valid_mask_custom.float().mean().item()
            print(f"Ground fraction with ground_ids={ground_ids.tolist()}: "
                  f"{frac_custom:.4f}")

        print()

    print("Done.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Debug semantic IDs and ground masks.")
    parser.add_argument(
        "--base_cfg",
        type=str,
        default="./configs/waymo_gs_base.yaml",
        help="Base config (same as training).",
    )
    parser.add_argument(
        "--data_cfg",
        type=str,
        default="./configs/waymo.yaml",
        help="Data config (same as training).",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="",
        help="Override cfg.source_path if provided.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=5,
        help="Number of training images to inspect.",
    )
    parser.add_argument(
        "--ground_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of class IDs to treat as ground "
             "(e.g. --ground_ids 0 1 2).",
    )

    args = parser.parse_args()

    # Load and merge configs
    cfg = OmegaConf.merge(
        OmegaConf.load(args.base_cfg),
        OmegaConf.load(args.data_cfg)
    )
    if args.source_path:
        cfg.source_path = args.source_path

    print("Using data_type:", cfg.data_type)
    print("Source path:", cfg.source_path)
    if args.ground_ids is not None:
        print("Custom ground IDs:", args.ground_ids)
    print()

    analyze_semantics(cfg, num_images=args.num_images,
                      ground_ids=args.ground_ids)


if __name__ == "__main__":
    main()

"""
Interactive visualization of DMSurface partitions

Shows each partition one by one with damaged and GT views.
Press 'q' to close window and move to next partition.
Press Ctrl+C to exit completely.
"""

import argparse
import os
import glob
import numpy as np
import open3d as o3d


def print_partition_stats(damaged_labels, gt_labels, partition_name, partition_idx, total_partitions):
    """Print bar chart with point counts in terminal."""
    
    dmg_intact = int(np.sum(damaged_labels == 0))
    dmg_damaged = int(np.sum(damaged_labels == 1))
    gt_intact = int(np.sum(gt_labels == 0))
    gt_recon = int(np.sum(gt_labels == 1))
    
    dmg_total = len(damaged_labels)
    gt_total = len(gt_labels)
    
    bar_width = 30
    
    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  DAMAGED PARTITION                                              │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    
    # Damaged - Intact bar
    intact_bar_len = int(dmg_intact / dmg_total * bar_width) if dmg_total > 0 else 0
    intact_pct = dmg_intact / dmg_total * 100 if dmg_total > 0 else 0
    print(f"  │  Intact  ▓{'█' * intact_bar_len}{'░' * (bar_width - intact_bar_len)}▓ {dmg_intact:>5} ({intact_pct:5.1f}%)  │")
    
    # Damaged - Damaged bar
    damaged_bar_len = int(dmg_damaged / dmg_total * bar_width) if dmg_total > 0 else 0
    damaged_pct = dmg_damaged / dmg_total * 100 if dmg_total > 0 else 0
    print(f"  │  Damaged ▓{'█' * damaged_bar_len}{'░' * (bar_width - damaged_bar_len)}▓ {dmg_damaged:>5} ({damaged_pct:5.1f}%)  │")
    
    print(f"  │  Total: {dmg_total}                                                │"[:68] + "│")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  GT PARTITION                                                   │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    
    # GT - Intact bar
    gt_intact_bar_len = int(gt_intact / gt_total * bar_width) if gt_total > 0 else 0
    gt_intact_pct = gt_intact / gt_total * 100 if gt_total > 0 else 0
    print(f"  │  Intact  ▓{'█' * gt_intact_bar_len}{'░' * (bar_width - gt_intact_bar_len)}▓ {gt_intact:>5} ({gt_intact_pct:5.1f}%)  │")
    
    # GT - Reconstructed bar
    gt_recon_bar_len = int(gt_recon / gt_total * bar_width) if gt_total > 0 else 0
    gt_recon_pct = gt_recon / gt_total * 100 if gt_total > 0 else 0
    print(f"  │  Recon   ▓{'█' * gt_recon_bar_len}{'░' * (bar_width - gt_recon_bar_len)}▓ {gt_recon:>5} ({gt_recon_pct:5.1f}%)  │")
    
    print(f"  │  Total: {gt_total}                                                │"[:68] + "│")
    print(f"  └─────────────────────────────────────────────────────────────────┘")


def visualize_partition_pair(damaged_file, gt_file, partition_idx, total_partitions):
    """
    Visualize damaged and GT partition side by side.
    
    Args:
        damaged_file: Path to damaged partition .npz
        gt_file: Path to GT partition .npz
        partition_idx: Current partition index
        total_partitions: Total number of partitions
    """
    # Load data
    damaged_data = np.load(damaged_file)
    damaged_pts = damaged_data['points']
    damaged_labels = damaged_data['labels']
    
    gt_data = np.load(gt_file)
    gt_pts = gt_data['points']
    gt_labels = gt_data['labels']
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"Partition {partition_idx}/{total_partitions}")
    print(f"File: {os.path.basename(damaged_file)}")
    print(f"{'='*80}")
    print(f"Damaged partition:")
    print(f"  Total points: {len(damaged_pts)}")
    print(f"  Intact (label=0): {np.sum(damaged_labels==0)} ({np.sum(damaged_labels==0)/len(damaged_labels)*100:.1f}%)")
    print(f"  Damaged (label=1): {np.sum(damaged_labels==1)} ({np.sum(damaged_labels==1)/len(damaged_labels)*100:.1f}%)")
    
    print(f"\nGT partition:")
    print(f"  Total points: {len(gt_pts)}")
    print(f"  Intact (label=0): {np.sum(gt_labels==0)} ({np.sum(gt_labels==0)/len(gt_labels)*100:.1f}%)")
    print(f"  Reconstruction (label=1): {np.sum(gt_labels==1)} ({np.sum(gt_labels==1)/len(gt_labels)*100:.1f}%)")
    
    # Create point clouds for damaged
    pcd_damaged = o3d.geometry.PointCloud()
    pcd_damaged.points = o3d.utility.Vector3dVector(damaged_pts)
    
    # Color: grey=intact, red=damaged
    colors_damaged = np.zeros((len(damaged_pts), 3))
    colors_damaged[damaged_labels == 0] = [0.7, 0.7, 0.7]  # Grey
    colors_damaged[damaged_labels == 1] = [1.0, 0.0, 0.0]  # Red
    pcd_damaged.colors = o3d.utility.Vector3dVector(colors_damaged)
    
    # Create point clouds for GT
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
    
    # Color: grey=intact, green=reconstruction
    colors_gt = np.zeros((len(gt_pts), 3))
    colors_gt[gt_labels == 0] = [0.7, 0.7, 0.7]  # Grey
    colors_gt[gt_labels == 1] = [0.0, 1.0, 0.0]  # Green
    pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)
    
    # Offset GT to the right for side-by-side view
    bbox = pcd_damaged.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    offset = np.array([extent[0] * 1.5, 0, 0])
    pcd_gt.translate(offset)
    
    # Print stats in terminal
    print_partition_stats(damaged_labels, gt_labels,
                          os.path.basename(damaged_file),
                          partition_idx, total_partitions)
    
    # Visualize 3D point clouds
    print(f"\n  Left: Damaged (grey=intact, red=damaged)")
    print(f"  Right: GT (grey=intact, green=reconstructed)")
    print(f"  Press 'q' to close and show next partition")
    
    o3d.visualization.draw_geometries(
        [pcd_damaged, pcd_gt],
        window_name=f"Partition {partition_idx}/{total_partitions} - {os.path.basename(damaged_file)}",
        width=1600,
        height=900
    )


def main(args):
    # Find all partition files
    print(f"Scanning directory: {args.data_root}")
    
    partition_pairs = []
    
    # Scan all surface directories
    for surface_id in range(1, 21):
        surface_dir = os.path.join(args.data_root, f"{surface_id:03d}", "partitions")
        
        if not os.path.exists(surface_dir):
            continue
        
        # Find all damaged partition files (not GT)
        damaged_files = sorted(glob.glob(os.path.join(surface_dir, "*[0-9][0-9].npz")))
        
        for damaged_file in damaged_files:
            # Skip if it's a GT file
            if '_gt' in damaged_file:
                continue
            
            # Find corresponding GT file
            gt_file = damaged_file.replace('.npz', '_gt.npz')
            
            if os.path.exists(gt_file):
                partition_pairs.append((damaged_file, gt_file))
    
    print(f"\nFound {len(partition_pairs)} partition pairs")
    
    if len(partition_pairs) == 0:
        print("ERROR: No partitions found!")
        print(f"Check that partitions exist in: {args.data_root}/*/partitions/")
        return
    
    # Filter by surface if specified
    if args.surface_id is not None:
        surface_str = f"/{args.surface_id:03d}/"
        partition_pairs = [p for p in partition_pairs if surface_str in p[0]]
        print(f"Filtered to surface {args.surface_id:03d}: {len(partition_pairs)} partitions")
    
    # Filter by damage flag if specified
    if args.only_damaged:
        partition_pairs = [p for p in partition_pairs if '01.npz' in p[0]]
        print(f"Filtered to damaged partitions only: {len(partition_pairs)} partitions")
    
    if len(partition_pairs) == 0:
        print("ERROR: No partitions match the filters!")
        return
    
    # Limit number of partitions if specified
    if args.max_partitions is not None:
        partition_pairs = partition_pairs[:args.max_partitions]
        print(f"Limited to first {args.max_partitions} partitions")
    
    print(f"\nStarting visualization of {len(partition_pairs)} partitions...")
    print("="*80)
    
    # Visualize each partition
    try:
        for idx, (damaged_file, gt_file) in enumerate(partition_pairs, 1):
            visualize_partition_pair(damaged_file, gt_file, idx, len(partition_pairs))
    
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize DMSurface partitions interactively')
    
    parser.add_argument('--data_root', type=str, default='data/DMSurface',
                       help='Root directory containing DMSurface data')
    parser.add_argument('--surface_id', type=int, default=None,
                       help='Visualize only specific surface (1-20)')
    parser.add_argument('--only_damaged', action='store_true',
                       help='Only show partitions with damage (flag=01)')
    parser.add_argument('--max_partitions', type=int, default=None,
                       help='Maximum number of partitions to show')
    
    args = parser.parse_args()
    
    main(args)

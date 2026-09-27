"""
Post-process DMSurface partitions

Removes partition pairs where either damaged or GT partition
has no intact points (label=0). These are invalid for training
since the network needs intact context to reconstruct.
"""

import os
import glob
import numpy as np
import argparse


def process_surface(surface_dir, dry_run=False):
    """
    Check and remove invalid partitions for one surface.
    
    Returns:
        total: Total partition pairs found
        removed: Number of pairs removed
    """
    partition_dir = os.path.join(surface_dir, "partitions")
    
    if not os.path.exists(partition_dir):
        return 0, 0
    
    # Find all damaged partition files (not GT)
    all_files = sorted(glob.glob(os.path.join(partition_dir, "*.npz")))
    damaged_files = [f for f in all_files if '_gt' not in f]
    
    total = 0
    removed = 0
    
    for damaged_file in damaged_files:
        gt_file = damaged_file.replace('.npz', '_gt.npz')
        
        # Skip non-damaged partitions (flag=00, no GT exists)
        if not os.path.exists(gt_file):
            continue
        
        total += 1
        
        # Load and check
        damaged_data = np.load(damaged_file)
        gt_data = np.load(gt_file)
        
        damaged_labels = damaged_data['labels']
        gt_labels = gt_data['labels']
        
        dmg_intact = np.sum(damaged_labels == 0)
        gt_intact = np.sum(gt_labels == 0)
        
        # Remove if either has no intact points
        if dmg_intact == 0 or gt_intact == 0:
            filename = os.path.basename(damaged_file)
            print(f"  Removing {filename} (dmg_intact={dmg_intact}, gt_intact={gt_intact})")
            
            if not dry_run:
                os.remove(damaged_file)
                os.remove(gt_file)
            
            removed += 1
    
    return total, removed


def main(args):
    print(f"Post-processing DMSurface partitions")
    print(f"Data root: {args.data_root}")
    if args.dry_run:
        print("DRY RUN - no files will be deleted")
    print("=" * 60)
    
    total_all = 0
    removed_all = 0
    
    for surface_id in range(1, 21):
        surface_dir = os.path.join(args.data_root, f"{surface_id:03d}")
        
        if not os.path.exists(surface_dir):
            continue
        
        total, removed = process_surface(surface_dir, args.dry_run)
        
        if total > 0:
            print(f"Surface {surface_id:03d}: {removed}/{total} removed")
        
        total_all += total
        removed_all += removed
    
    print("=" * 60)
    print(f"Total: {removed_all}/{total_all} partition pairs removed")
    print(f"Remaining: {total_all - removed_all} valid partition pairs")
    
    if args.dry_run:
        print("\nThis was a DRY RUN. Run without --dry_run to delete files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-process DMSurface partitions')
    
    parser.add_argument('--data_root', type=str, default='data/DMSurface',
                       help='Root directory containing DMSurface data')
    parser.add_argument('--dry_run', action='store_true',
                       help='Only show what would be removed, without deleting')
    
    args = parser.parse_args()
    main(args)

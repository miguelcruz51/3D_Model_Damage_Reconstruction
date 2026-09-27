#!/usr/bin/env python3
"""
Interactive partition visualizer for DMC dataset.
Allows browsing and visualizing partitions with damaged point highlighting.
"""

import os
import glob
import numpy as np
import open3d as o3d


def get_available_components(base_path):
    """Get list of available component folders (001-016)."""
    components = []
    for i in range(1, 17):
        comp_id = f"{i:03d}"
        comp_path = os.path.join(base_path, comp_id)
        if os.path.exists(comp_path):
            components.append(comp_id)
    return components


def get_available_versions(base_path, component):
    """Get list of available versions for a component from partition files."""
    partition_path = os.path.join(base_path, component, "partitions")
    
    if not os.path.exists(partition_path):
        return []
    
    versions = set()
    pattern = os.path.join(partition_path, f"{component}*.npz")
    files = glob.glob(pattern)
    
    for f in files:
        basename = os.path.basename(f)
        if "_gt" not in basename and len(basename) >= 9:
            # Extract version from filename: 001001000000.npz -> version 001
            version = basename[3:6]
            versions.add(version)
    
    return sorted(list(versions))


def get_available_partitions(base_path, component, version):
    """Get list of available partitions for a component/version."""
    partition_path = os.path.join(base_path, component, "partitions")
    
    if not os.path.exists(partition_path):
        return []
    
    # Find all .npz files matching the pattern
    pattern = os.path.join(partition_path, f"{component}{version}*.npz")
    files = glob.glob(pattern)
    
    partitions = []
    for f in files:
        basename = os.path.basename(f)
        # Skip ground truth files
        if "_gt" in basename:
            continue
        # Extract partition info: componentversionpartition_numberdamaged_flag
        # Example: 001001001301.npz -> partition 0013, damaged flag 01
        if len(basename) >= 16:  # 3+3+4+2 = 12 + .npz
            partition_num = basename[6:10]
            damaged_flag = basename[10:12]
            partitions.append({
                'file': basename,
                'partition': partition_num,
                'has_damage': damaged_flag == '01',
                'full_path': f
            })
    
    # Sort by partition number
    partitions.sort(key=lambda x: x['partition'])
    return partitions


def load_npz_with_labels(npz_path):
    """Load point cloud from npz file and extract labels if present."""
    data = np.load(npz_path)
    
    # Typical keys: 'points', 'labels', 'normals', etc.
    points = data['points'] if 'points' in data else data['data']
    labels = data['labels'] if 'labels' in data else None
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    return pcd, labels


def visualize_partition(npz_path, has_damage):
    """Visualize a single partition."""
    pcd, labels = load_npz_with_labels(npz_path)
    
    if has_damage and labels is not None:
        # Color code: grey for good points (0), red for damaged (1)
        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.5, 0.5, 0.5]  # Grey
        colors[labels == 1] = [1.0, 0.0, 0.0]  # Red
        pcd.colors = o3d.utility.Vector3dVector(colors)
        title = f"Partition (Grey=Good, Red=Damaged)"
    else:
        # No damage or no labels, show in default color
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        title = "Partition (No Damage)"
    
    print(f"\n{title}")
    print(f"Points: {len(pcd.points)}")
    if labels is not None:
        damaged_count = np.sum(labels == 1)
        print(f"Damaged points: {damaged_count}")
    
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_with_ground_truth(npz_path, gt_path, has_damage):
    """Visualize partition alongside its ground truth."""
    # Load partition
    pcd, labels = load_npz_with_labels(npz_path)
    
    if has_damage and labels is not None:
        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.5, 0.5, 0.5]  # Grey
        colors[labels == 1] = [1.0, 0.0, 0.0]  # Red
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    # Load ground truth with labels
    gt_pcd, gt_labels = load_npz_with_labels(gt_path)
    
    # Color ground truth: grey for label 0, green for label 1
    if gt_labels is not None:
        gt_colors = np.zeros((len(gt_labels), 3))
        gt_colors[gt_labels == 0] = [0.5, 0.5, 0.5]  # Grey
        gt_colors[gt_labels == 1] = [0.0, 0.7, 0.0]  # Green
        gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)
    else:
        gt_pcd.paint_uniform_color([0.0, 0.7, 0.0])  # Green for GT
    
    # Offset GT for side-by-side view
    gt_points_array = np.asarray(gt_pcd.points)
    bbox = pcd.get_axis_aligned_bounding_box()
    offset = bbox.get_max_bound()[0] - bbox.get_min_bound()[0] + 0.1
    gt_points_array[:, 0] += offset
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points_array)
    
    print(f"\nPartition (left) vs Ground Truth (right)")
    print(f"Partition points: {len(pcd.points)}")
    print(f"GT points: {len(gt_pcd.points)}")
    if gt_labels is not None:
        gt_green_count = np.sum(gt_labels == 1)
        print(f"GT green points (label 1): {gt_green_count}")
    
    o3d.visualization.draw_geometries([pcd, gt_pcd], 
                                      window_name="Partition + Ground Truth")


def main():
    base_path = "/home/miguelcruz/Documents/D-PCN2/data/DMC"
    
    print("=" * 60)
    print("DMC Partition Visualizer")
    print("=" * 60)
    
    while True:
        # Select component
        components = get_available_components(base_path)
        if not components:
            print("No components found!")
            return
        
        print(f"\nAvailable components: {', '.join(components)}")
        component = input("Select component (001-016) or 'q' to quit: ").strip()
        
        if component.lower() == 'q':
            break
        
        if component not in components:
            print("Invalid component!")
            continue
        
        # Select version
        versions = get_available_versions(base_path, component)
        if not versions:
            print(f"No versions found for component {component}!")
            continue
        
        print(f"\nAvailable versions: {', '.join(versions)}")
        version = input("Select version (001-010) or 'b' to go back: ").strip()
        
        if version.lower() == 'b':
            continue
        
        if version not in versions:
            print("Invalid version!")
            continue
        
        # Get partitions
        partitions = get_available_partitions(base_path, component, version)
        if not partitions:
            print(f"No partitions found for {component}/{version}!")
            continue
        
        # Display partitions
        print(f"\nAvailable partitions for {component}/{version}:")
        print("-" * 60)
        for idx, part in enumerate(partitions):
            damage_status = "WITH DAMAGE" if part['has_damage'] else "NO DAMAGE"
            print(f"{idx+1}. Partition {part['partition']} - {damage_status}")
        print("-" * 60)
        
        # Select partition
        selection = input("\nSelect partition number (or 'b' to go back): ").strip()
        
        if selection.lower() == 'b':
            continue
        
        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(partitions):
                print("Invalid selection!")
                continue
        except ValueError:
            print("Invalid input!")
            continue
        
        selected = partitions[idx]
        npz_path = selected['full_path']
        
        # Check for ground truth
        gt_path = npz_path.replace('.npz', '_gt.npz')
        has_gt = os.path.exists(gt_path)
        
        # Visualization options
        print(f"\nVisualization options:")
        print("1. View partition only")
        if has_gt:
            print("2. View partition + ground truth")
        
        vis_choice = input("Select option: ").strip()
        
        if vis_choice == '1':
            visualize_partition(npz_path, selected['has_damage'])
        elif vis_choice == '2' and has_gt:
            visualize_with_ground_truth(npz_path, gt_path, selected['has_damage'])
        else:
            print("Invalid option!")


if __name__ == "__main__":
    main()

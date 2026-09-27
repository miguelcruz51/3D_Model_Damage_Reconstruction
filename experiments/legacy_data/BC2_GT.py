"""
BC2_GT.py - Enhanced preprocessing to generate GT patches

This script extends BC2.py to generate:
1. Damaged patches (with labels 0=intact, 1=damaged) - same as BC2.py
2. GT patches (with labels 0=intact, 1=reconstructed) - NEW!

Works in ORIGINAL coordinate space (no normalization).

GT patches are created by:
- Using the aligned nominal and damaged point clouds
- Identifying intact points (label=0) from damaged
- Identifying reconstruction targets (label=1) from nominal (points missing in damaged)
- Combining them into GT patch
- Saving with "_gt" suffix

Output structure:
  data/DMC/001/001001.npz  <- damaged patch (label 1=damaged points)
  data/DMC/001/001001_gt.npz  <- GT patch (label 1=reconstruction targets)
"""

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import os
import glob

def manual_perturbation(mesh, rotation_deg=(0, 0, 0), translation=(0, 0, 0)):
    """
    Applies manual rotation (in degrees) and translation to a mesh.
    rotation_deg: tuple of 3 floats (rx, ry, rz) in degrees
    translation: tuple of 3 floats (tx, ty, tz)
    """
    # Convert degrees to radians
    rotation_rad = np.radians(rotation_deg)
    
    # Rotation matrix
    R = mesh.get_rotation_matrix_from_xyz(rotation_rad)
    mesh.rotate(R, center=np.mean(np.asarray(mesh.points), axis=0))
    
    # Apply translation
    mesh.translate(translation)
    
    return mesh


def process_pair_with_gt(nominal_path, damaged_path,
                         align_methods=("icp",), vis_flags=("align",),
                         dbscan_eps=1, dbscan_min=30, threshold=2.5):
    """
    Process one STL pair - exactly like BC2.py but returns more info for GT generation.
    
    Args:
        nominal_path: Path to nominal STL
        damaged_path: Path to damaged STL
        align_methods: Alignment methods to use
        vis_flags: Visualization flags
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min: DBSCAN min points
        threshold: Distance threshold for damage detection
        
    Returns:
        dmg_pcd: Damaged point cloud (aligned)
        damage_pcd: Damage region point cloud
        nom_pcd: Nominal point cloud (aligned) - for GT generation
    """
    
    # Load STLs and sample large clouds (exactly like BC2.py)
    nom = o3d.io.read_triangle_mesh(nominal_path).compute_vertex_normals()
    dmg = o3d.io.read_triangle_mesh(damaged_path).compute_vertex_normals()
    nom_pcd = nom.sample_points_uniformly(200000)
    dmg_pcd = dmg.sample_points_uniformly(200000)
    dmg_pcd.paint_uniform_color([0, 0, 1])
    nom_pcd.paint_uniform_color([0, 1, 0.2])

    dmg_pcd = manual_perturbation(
        dmg_pcd,
        rotation_deg=(0, 0, 0),     # Rotate 10 degrees around X axis
        translation=(-20, 0, 0)     # Move 5mm on X axis
    )

    o3d.visualization.draw_geometries([nom_pcd, dmg_pcd], window_name="Initial Clouds")

    # quit()


    # Apply ICP alignment if requested
    if "icp" in align_methods:
        print("ICP Alignment")
        icp_result = o3d.pipelines.registration.registration_icp(
            nom_pcd, dmg_pcd, 1.5, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        print("Alignment Result:", icp_result.fitness)
        nom_pcd.transform(icp_result.transformation)
        print("ICP Completed!")

    # Visualize aligned clouds
    if "align" in vis_flags:
        o3d.visualization.draw_geometries([nom_pcd, dmg_pcd], window_name="Aligned Clouds")

    # Extract missing points (damage detection) - exactly like BC2.py
    nominal_points = np.asarray(nom_pcd.points)
    damaged_points = np.asarray(dmg_pcd.points)
    
    nom_tree = cKDTree(nominal_points)
    dmg_tree = cKDTree(damaged_points)
    
    # Find missing points in damaged model
    missing_in_damaged = [i for i, point in enumerate(nominal_points)
                         if dmg_tree.query(point, 1)[1] < len(damaged_points) and 
                            np.linalg.norm(point - damaged_points[dmg_tree.query(point, 1)[1]]) > threshold]
    
    # Find missing points in nominal model
    missing_in_nominal = [i for i, point in enumerate(damaged_points)
                         if nom_tree.query(point, 1)[1] < len(nominal_points) and 
                            np.linalg.norm(point - nominal_points[nom_tree.query(point, 1)[1]]) > threshold]

    print(f"Points missing in damaged model: {len(missing_in_damaged)}")
    print(f"Points missing in nominal model: {len(missing_in_nominal)}")
    
    # Create damage surface point cloud
    non_planar_surface_points = damaged_points[missing_in_nominal]
    non_planar_surface = o3d.geometry.PointCloud()
    non_planar_surface.points = o3d.utility.Vector3dVector(non_planar_surface_points)
    non_planar_surface.paint_uniform_color([1, 0, 0])

    # DBSCAN clustering
    labels_non_planar_surface = np.array(
        non_planar_surface.cluster_dbscan(dbscan_eps, dbscan_min, print_progress=False)
    )
    print("DBSCAN Labels:", labels_non_planar_surface)

    # Keep all clusters except noise (-1)
    valid_indices = np.where(labels_non_planar_surface != -1)[0]
    
    if len(valid_indices) > 0:
        non_planar_surface = non_planar_surface.select_by_index(valid_indices)
    else:
        print("Warning: No valid clusters found. Using all missing points.")

    # Create final damage point cloud
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(np.asarray(non_planar_surface.points))
    surface_pcd.paint_uniform_color([1, 0, 0])

    return dmg_pcd, surface_pcd, nom_pcd


def save_labeled_point_cloud_with_gt(dmg_pcd, damage_pcd, nom_pcd, full_id, out_dir, 
                                     match_radius=2.0):
    """
    Save both damaged patch and GT patch.
    
    Damaged patch: label 1=damaged, 0=intact
    GT patch: label 1=reconstruction targets, 0=intact
    
    Args:
        dmg_pcd: Damaged point cloud (aligned)
        damage_pcd: Damage region point cloud
        nom_pcd: Nominal point cloud (aligned)
        full_id: File ID (e.g., "001001")
        out_dir: Output directory
        match_radius: Radius for damage labeling
    """
    os.makedirs(out_dir, exist_ok=True)

    # Get points
    damaged_points = np.asarray(dmg_pcd.points)
    damaged_normals = np.asarray(dmg_pcd.normals)
    nominal_points = np.asarray(nom_pcd.points)
    nominal_normals = np.asarray(nom_pcd.normals)
    damage_region_points = np.asarray(damage_pcd.points)

    # ========================================================================
    # STEP 1: Create DAMAGED patch (same as BC2.py)
    # ========================================================================
    
    # Create KDTree for damaged region
    damage_tree = cKDTree(damage_region_points)
    
    # Label: 1 if point is close to damage region, else 0
    _, idx = damage_tree.query(damaged_points, distance_upper_bound=match_radius)
    damaged_labels = (idx != len(damage_region_points)).astype(np.int32)

    # Save damaged patch
    np.savez(os.path.join(out_dir, f"{full_id}.npz"),
             points=damaged_points, normals=damaged_normals, labels=damaged_labels)

    print(f"Saved damaged patch: {full_id}.npz")
    print(f"  Total points: {len(damaged_points)}")
    print(f"  Damaged (label=1): {np.sum(damaged_labels==1)} ({np.sum(damaged_labels==1)/len(damaged_labels)*100:.1f}%)")
    print(f"  Intact (label=0): {np.sum(damaged_labels==0)} ({np.sum(damaged_labels==0)/len(damaged_labels)*100:.1f}%)")
    
    # Check if there's any damage
    if np.sum(damaged_labels == 1) == 0:
        print(f"No damage found in {full_id}, skipping GT generation")
        return

    # ========================================================================
    # STEP 2: Create GT patch
    # ========================================================================
    
    print(f"Generating GT patch for {full_id}...")
    
    # Build KDTree on damaged points
    damaged_tree = cKDTree(damaged_points)
    
    # For each nominal point, check if it exists in damaged
    # Points that are FAR from damaged are reconstruction targets
    distances, _ = damaged_tree.query(nominal_points, k=1)
    
    # Reconstruction targets: nominal points missing in damaged
    # Use a SMALLER threshold than match_radius to avoid gaps
    # This ensures smooth transition between intact and reconstruction regions
    reconstruction_threshold = match_radius * 0.5  # Half of match_radius to reduce gap
    reconstruction_mask = distances > reconstruction_threshold
    
    print(f"  Nominal points: {len(nominal_points)}")
    print(f"  Reconstruction targets (missing in damaged): {np.sum(reconstruction_mask)}")
    
    # Get intact points from damaged (label=0)
    intact_mask = damaged_labels == 0
    intact_points = damaged_points[intact_mask]
    intact_normals = damaged_normals[intact_mask]
    
    # Get reconstruction targets from nominal (points missing in damaged)
    reconstruction_points = nominal_points[reconstruction_mask]
    reconstruction_normals = nominal_normals[reconstruction_mask]
    
    print(f"  Intact points from damaged: {len(intact_points)}")
    print(f"  Reconstruction targets from nominal: {len(reconstruction_points)}")
    
    # Combine: intact (label=0) + reconstruction targets (label=1)
    gt_points = np.vstack([intact_points, reconstruction_points])
    gt_normals = np.vstack([intact_normals, reconstruction_normals])
    gt_labels = np.hstack([
        np.zeros(len(intact_points), dtype=np.int32),      # 0 for intact
        np.ones(len(reconstruction_points), dtype=np.int32) # 1 for reconstruction targets
    ])
    
    # Save GT patch with "_gt" suffix
    gt_filename = f"{full_id}_gt.npz"
    np.savez(os.path.join(out_dir, gt_filename),
             points=gt_points, normals=gt_normals, labels=gt_labels)
    
    print(f"Saved GT patch: {gt_filename}")
    print(f"  Total points: {len(gt_points)}")
    print(f"  Intact (label=0): {np.sum(gt_labels==0)} ({np.sum(gt_labels==0)/len(gt_labels)*100:.1f}%)")
    print(f"  Reconstruction targets (label=1): {np.sum(gt_labels==1)} ({np.sum(gt_labels==1)/len(gt_labels)*100:.1f}%)")


def partition_with_gt(dmg_pcd, nom_pcd, labels, full_id, out_dir,
                     grid_size=6, partition_size=2048, match_radius=2.0, visualize=False):
    """
    Partition point cloud spatially and generate GT patches for each partition.
    
    For each partition:
    - Save damaged partition: {full_id}{partition_id}{damage_flag}.npz
    - Save GT partition: {full_id}{partition_id}{damage_flag}_gt.npz
    
    Args:
        dmg_pcd: Damaged point cloud
        nom_pcd: Nominal point cloud (aligned)
        labels: Point labels (0=intact, 1=damaged)
        full_id: File ID
        out_dir: Output directory
        grid_size: Grid size for spatial partitioning
        partition_size: Number of points per partition
        match_radius: Radius for reconstruction target identification
        visualize: Whether to visualize partitions
    """
    os.makedirs(out_dir, exist_ok=True)

    damaged_points = np.asarray(dmg_pcd.points)
    damaged_normals = np.asarray(dmg_pcd.normals)
    nominal_points = np.asarray(nom_pcd.points)
    nominal_normals = np.asarray(nom_pcd.normals)

    # Get spatial bounds
    min_bound = damaged_points.min(axis=0)
    max_bound = damaged_points.max(axis=0)
    grid_step = (max_bound - min_bound) / grid_size

    partition_id = 0

    for xi in range(grid_size):
        for yi in range(grid_size):
            for zi in range(grid_size):
                # Compute bounds for this grid cell
                min_corner = min_bound + [xi, yi, zi] * grid_step
                max_corner = min_corner + grid_step

                # Find damaged points inside this box
                mask = np.all((damaged_points >= min_corner) & (damaged_points < max_corner), axis=1)
                cell_points = damaged_points[mask]
                cell_normals = damaged_normals[mask]
                cell_labels = labels[mask]

                if len(cell_points) == 0:
                    continue

                # Downsample or pad to partition_size
                if len(cell_points) >= partition_size:
                    idx = np.random.choice(len(cell_points), partition_size, replace=False)
                else:
                    idx = np.random.choice(len(cell_points), partition_size, replace=True)

                pts_part = cell_points[idx]
                nrms_part = cell_normals[idx]
                lbls_part = cell_labels[idx]

                # Set damage flag
                damage_flag = "01" if np.any(lbls_part == 1) else "00"
                partition_id_str = f"{partition_id:04d}"
                filename = f"{full_id}{partition_id_str}{damage_flag}"

                # Save damaged partition
                np.savez(os.path.join(out_dir, f"{filename}.npz"),
                         points=pts_part, normals=nrms_part, labels=lbls_part)

                print(f"Saved partition {partition_id}: {filename}.npz")

                # Generate GT partition if there's damage
                if damage_flag == "01":
                    print(f"  Generating GT for partition {partition_id}...")
                    
                    try:
                        # Find nominal points in the same region
                        nom_mask = np.all((nominal_points >= min_corner) & (nominal_points < max_corner), axis=1)
                        nom_cell_points = nominal_points[nom_mask]
                        nom_cell_normals = nominal_normals[nom_mask]
                        
                        if len(nom_cell_points) == 0:
                            print(f"  Warning: No nominal points in this region, skipping GT")
                            partition_id += 1
                            continue
                        
                        # Build KDTree on damaged partition points
                        damaged_tree = cKDTree(pts_part)
                        
                        # Find reconstruction targets (nominal points missing in damaged)
                        # Use smaller threshold to reduce gap between intact and reconstruction
                        distances, _ = damaged_tree.query(nom_cell_points, k=1)
                        reconstruction_threshold = match_radius * 0.2  # Reduce gap
                        reconstruction_threshold =  0.05  # Reduce gap
                        reconstruction_mask = distances > reconstruction_threshold
                        
                        # Get intact points from damaged partition
                        intact_mask = lbls_part == 0
                        intact_points = pts_part[intact_mask]
                        intact_normals = nrms_part[intact_mask]
                        
                        # Get reconstruction targets from nominal
                        reconstruction_points = nom_cell_points[reconstruction_mask]
                        reconstruction_normals = nom_cell_normals[reconstruction_mask]
                        
                        print(f"  Intact: {len(intact_points)}, Reconstruction: {len(reconstruction_points)}")
                        
                        # Combine
                        gt_points = np.vstack([intact_points, reconstruction_points])
                        gt_normals = np.vstack([intact_normals, reconstruction_normals])
                        gt_labels = np.hstack([
                            np.zeros(len(intact_points), dtype=np.int32),
                            np.ones(len(reconstruction_points), dtype=np.int32)
                        ])
                        
                        # Resample to partition_size
                        if len(gt_points) > partition_size:
                            indices = np.random.choice(len(gt_points), partition_size, replace=False)
                        else:
                            indices = np.random.choice(len(gt_points), partition_size, replace=True)
                        
                        gt_points = gt_points[indices]
                        gt_normals = gt_normals[indices]
                        gt_labels = gt_labels[indices]
                        
                        # Save GT partition
                        gt_filename = f"{filename}_gt.npz"
                        np.savez(os.path.join(out_dir, gt_filename),
                                 points=gt_points, normals=gt_normals, labels=gt_labels)
                        
                        print(f"  Saved GT partition: {gt_filename}")
                        
                    except Exception as e:
                        print(f"  Error generating GT for partition {partition_id}: {e}")

                partition_id += 1

    print(f"Saved {partition_id} partitions for {full_id}")


if __name__ == "__main__":
    print("Starting BC2_GT batch processing...")
    print("This will generate both damaged patches and GT patches")
    print("="*80)

    # Paths - UPDATE THESE FOR YOUR DATA
    # Example for component 001:
    path = '/home/miguelcruz/Documents/D-PCN2/data/Damaged_Comp/016'
    nominal_stl_path = '/home/miguelcruz/Documents/D-PCN2/data/Damaged_Comp/016/016_nom.STL'
    output_dir = "data/DMC/016"

    # Get all damaged STL files
    files = sorted(glob.glob(os.path.join(path, "[0-9][0-9][0-9][0-9][0-9][0-9].STL")))

    # Filter out already processed files
    files_to_process = []
    for filepath in files:
        filename = os.path.basename(filepath)
        full_id = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{full_id}.npz")
        
        if not os.path.exists(output_file):
            files_to_process.append(filepath)
        else:
            print(f"Skipping {filename} (already processed)")

    files = files_to_process
    
    print(f"\nFound {len(files)} damaged STL files to process")
    if len(files) == 0:
        print("All files already processed!")
        exit(0)
    
    print("="*80)

    for filepath in files:
        filename = os.path.basename(filepath)
        full_id = os.path.splitext(filename)[0]
        damaged_path = os.path.join(path, filename)
        
        print(f"\nProcessing {filename}...")

        try:
            # Process pair to get damaged, damage region, and nominal
            dmg_pcd, damage_pcd, nom_pcd = process_pair_with_gt(
                nominal_stl_path, damaged_path,
                align_methods=("icp",),
                vis_flags=("align",),
                dbscan_eps=20, dbscan_min=10, threshold=1.8
            )
            
            # Save labeled point cloud with GT
            save_labeled_point_cloud_with_gt(
                dmg_pcd, damage_pcd, nom_pcd,
                full_id=full_id,
                out_dir=output_dir,
                match_radius=2.0
            )

            # Visualize damaged and GT patches
            print("\nVisualizing patches...")
            print("Close each window to continue...")
            
            # Load saved patches
            damaged_data = np.load(os.path.join(output_dir, f"{full_id}.npz"))
            damaged_points = damaged_data['points']
            damaged_labels = damaged_data['labels']
            
            gt_file = os.path.join(output_dir, f"{full_id}_gt.npz")
            if os.path.exists(gt_file):
                gt_data = np.load(gt_file)
                gt_points = gt_data['points']
                gt_labels = gt_data['labels']
                
                # Visualization 1: Damaged patch (grey=intact, red=damaged)
                pcd_damaged = o3d.geometry.PointCloud()
                pcd_damaged.points = o3d.utility.Vector3dVector(damaged_points)
                colors_damaged = np.zeros((len(damaged_points), 3))
                colors_damaged[damaged_labels == 0] = [0.6, 0.6, 0.6]  # Grey for intact
                colors_damaged[damaged_labels == 1] = [1, 0, 0]  # Red for damaged
                pcd_damaged.colors = o3d.utility.Vector3dVector(colors_damaged)
                
                o3d.visualization.draw_geometries(
                    [pcd_damaged],
                    window_name=f"{full_id} - Damaged Patch: Grey=Intact, Red=Damaged",
                    width=1400, height=900
                )
                
                # Visualization 2: GT patch (grey=intact, green=reconstruction targets)
                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
                colors_gt = np.zeros((len(gt_points), 3))
                colors_gt[gt_labels == 0] = [0.6, 0.6, 0.6]  # Grey for intact
                colors_gt[gt_labels == 1] = [0, 1, 0]  # Green for reconstruction targets
                pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)
                
                o3d.visualization.draw_geometries(
                    [pcd_gt],
                    window_name=f"{full_id}_gt - GT Patch: Grey=Intact, Green=Reconstruction Targets",
                    width=1400, height=900
                )

            # Ask if user wants to partition
            partition = input("\nPartition the point cloud? (y/n): ").strip().lower() == 'y'
            
            if partition:
                labels = np.load(os.path.join(output_dir, f"{full_id}.npz"))["labels"]
                partition_with_gt(
                    dmg_pcd, nom_pcd, labels,
                    full_id=full_id,
                    out_dir=os.path.join(output_dir, "partitions"),
                    grid_size=6,
                    partition_size=8192,
                    match_radius=2.0,
                    visualize=False
                )

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("BC2_GT processing complete!")
    print(f"Output directory: {output_dir}")
    print("\nFiles generated:")
    print("  - {id}.npz: Damaged patches")
    print("      points: XYZ coordinates (original scale)")
    print("      normals: Normal vectors")
    print("      labels: 0=intact, 1=damaged")
    print("  - {id}_gt.npz: GT patches")
    print("      points: XYZ coordinates (original scale)")
    print("      normals: Normal vectors")
    print("      labels: 0=intact (from damaged), 1=reconstruction targets (from nominal)")

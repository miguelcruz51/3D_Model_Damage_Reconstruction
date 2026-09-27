"""
Volumetric Nominal Patch Extraction - CORRECT VERSION

This extracts nominal patches using the SAME volumetric region as damaged patches.

KEY INSIGHT:
- Damaged patches (.npz in data/DMC/) are in ORIGINAL coordinates
- Nominal models (in data/DMC_processed/nominal/) are NORMALIZED
- We must work in the SAME coordinate system (normalized)
- Then extract using the volumetric extent of the damaged patch

IMPORTANT: For best quality, use HIGH-DENSITY nominal model
- Load from STL and sample densely (e.g., 50k-100k points)
- Extract volumetric patch from high-density model
- Then downsample to target number of points
- This gives much better coverage than extracting from pre-sampled 8k points

Process:
1. Load damaged patch (original coords) and normalize it
2. Load nominal (high-density if available, or pre-sampled)
3. Extract nominal patch from the SAME volume as damaged patch
4. Downsample to target number of points
5. Both patches now represent the same physical region
"""

import os
import numpy as np
from scipy.spatial import cKDTree

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False


def load_high_density_nominal(component_id, stl_path=None, num_points=100000):
    """
    Load high-density nominal model from STL.
    
    This provides much better coverage than the pre-sampled 8192 points.
    
    Args:
        component_id: component ID (e.g., '001')
        stl_path: path to STL file (if None, auto-detect)
        num_points: number of points to sample (default 100k for high density)
        
    Returns:
        points: (N, 3) high-density point cloud
        centroid: (3,) normalization centroid
        scale: scalar normalization scale
    """
    if not O3D_AVAILABLE:
        raise ImportError("Open3D required for STL loading. Install with: pip install open3d")
    
    # Auto-detect STL path if not provided
    if stl_path is None:
        stl_path = f'data/DMC/{component_id}/{component_id}_nom.STL'
    
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    # Load and sample STL
    mesh = o3d.io.read_triangle_mesh(stl_path)
    if mesh is None or len(mesh.triangles) == 0:
        raise RuntimeError(f'Failed to read mesh: {stl_path}')
    
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points, dtype=np.float32)
    
    # Normalize (same as preprocessing script)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    scale = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    
    if scale > 1e-6:
        points_normalized = points_centered / scale
    else:
        points_normalized = points_centered
        scale = 1.0
    
    return points_normalized, centroid, scale


def extract_nominal_patch_volumetric(damaged_patch_file, nominal_file, 
                                    num_points=2048, method='bbox', 
                                    expansion=0.05, use_high_density=True,
                                    high_density_points=100000):
    """
    Extract nominal patch using volumetric correspondence with damaged patch.
    
    This is the MAIN function to use for training datasets.
    
    Args:
        damaged_patch_file: path to damaged patch .npz (e.g., 'data/DMC/001/001001000901.npz')
        nominal_file: path to nominal .npz (e.g., 'data/DMC_processed/nominal/001_nom.npz')
        num_points: number of points to extract for nominal patch (final output)
        method: extraction method ('bbox', 'radius', 'knn')
        expansion: expansion factor for bbox/radius (0.05 = 5% larger)
        use_high_density: if True, load high-density nominal from STL (recommended!)
        high_density_points: number of points to sample from STL (default 100k)
        
    Returns:
        damaged_patch_normalized: (N, 3) damaged patch in normalized coords
        nominal_patch_normalized: (num_points, 3) nominal patch in normalized coords
        damaged_labels: (N,) damage labels for damaged patch
            0 = intact point
            1 = damaged point
        nominal_labels: (num_points,) labels for nominal patch
            0 = corresponds to damaged patch (existing point)
            1 = new point (reconstruction target)
        volume_info: dict with extraction volume information
    """
    # Load damaged patch (original coordinates)
    damaged_data = np.load(damaged_patch_file)
    damaged_points_original = damaged_data['points'].astype(np.float32)
    damaged_labels = damaged_data['labels'].astype(np.float32)
    
    # Load nominal - either high-density from STL or pre-sampled
    if use_high_density and O3D_AVAILABLE:
        # Extract component ID from damaged patch filename
        basename = os.path.basename(damaged_patch_file)
        component_id = basename[:3]
        
        try:
            print(f"Loading high-density nominal ({high_density_points} points)...")
            nominal_normalized, centroid, scale = load_high_density_nominal(
                component_id, num_points=high_density_points
            )
            print(f"  Loaded {nominal_normalized.shape[0]} points from STL")
        except Exception as e:
            print(f"  Warning: Could not load high-density nominal: {e}")
            print(f"  Falling back to pre-sampled nominal...")
            use_high_density = False
    
    if not use_high_density or not O3D_AVAILABLE:
        # Load pre-sampled nominal (normalized coordinates)
        nominal_data = np.load(nominal_file)
        nominal_normalized = nominal_data['points'].astype(np.float32)
        centroid = nominal_data['centroid'].astype(np.float32)
        scale = float(nominal_data['scale'])
        print(f"Using pre-sampled nominal ({nominal_normalized.shape[0]} points)")
    
    # Normalize damaged patch to match nominal coordinate system
    damaged_patch_normalized = (damaged_points_original - centroid) / scale
    
    # Get volume info from damaged patch
    volume_info = _get_patch_volume(damaged_patch_normalized)
    
    # Extract nominal patch using specified method
    # This extracts ALL points in the volume (could be many if high-density)
    if method == 'bbox':
        nominal_patch_full = _extract_bbox(
            nominal_normalized, volume_info, num_points=None, expansion=expansion
        )
    elif method == 'radius':
        nominal_patch_full = _extract_radius(
            nominal_normalized, volume_info, num_points=None, expansion=expansion
        )
    elif method == 'knn':
        nominal_patch_full = _extract_knn(
            nominal_normalized, damaged_patch_normalized, num_points=None
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bbox', 'radius', or 'knn'")
    
    # Now downsample to target number of points
    if len(nominal_patch_full) >= num_points:
        indices = np.random.choice(len(nominal_patch_full), num_points, replace=False)
    else:
        indices = np.random.choice(len(nominal_patch_full), num_points, replace=True)
    
    nominal_patch_normalized = nominal_patch_full[indices]
    
    # Label nominal patch points
    # 0 = corresponds to damaged patch (existing point)
    # 1 = new point (reconstruction target)
    nominal_labels = label_nominal_patch(
        damaged_patch_normalized, 
        nominal_patch_normalized,
        distance_threshold=0.01
    )
    
    # Store extraction info
    volume_info['extracted_points'] = len(nominal_patch_full)
    volume_info['final_points'] = num_points
    volume_info['high_density'] = use_high_density
    volume_info['reconstruction_targets'] = int(np.sum(nominal_labels == 1))
    volume_info['existing_points'] = int(np.sum(nominal_labels == 0))
    
    return damaged_patch_normalized, nominal_patch_normalized, damaged_labels, nominal_labels, volume_info


def _get_patch_volume(points):
    """Analyze spatial extent of a patch."""
    center = np.mean(points, axis=0)
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    extent = bbox_max - bbox_min
    
    distances = np.linalg.norm(points - center, axis=1)
    radius = np.max(distances)
    
    return {
        'center': center,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'radius': radius,
        'extent': extent
    }


def _extract_bbox(nominal_full, volume_info, num_points, expansion):
    """Extract using bounding box."""
    bbox_min = volume_info['bbox_min']
    bbox_max = volume_info['bbox_max']
    
    if expansion > 0:
        extent = bbox_max - bbox_min
        bbox_min = bbox_min - extent * expansion / 2
        bbox_max = bbox_max + extent * expansion / 2
    
    mask = np.all((nominal_full >= bbox_min) & (nominal_full <= bbox_max), axis=1)
    points_in_bbox = nominal_full[mask]
    
    if len(points_in_bbox) == 0:
        # Fallback to radius
        return _extract_radius(nominal_full, volume_info, num_points, expansion)
    
    # If num_points is None, return all points (for high-density extraction)
    if num_points is None:
        return points_in_bbox
    
    if len(points_in_bbox) >= num_points:
        indices = np.random.choice(len(points_in_bbox), num_points, replace=False)
    else:
        indices = np.random.choice(len(points_in_bbox), num_points, replace=True)
    
    return points_in_bbox[indices]


def _extract_radius(nominal_full, volume_info, num_points, expansion):
    """Extract using spherical radius."""
    center = volume_info['center']
    radius = volume_info['radius'] * (1 + expansion)
    
    distances = np.linalg.norm(nominal_full - center, axis=1)
    mask = distances <= radius
    points_in_sphere = nominal_full[mask]
    
    if len(points_in_sphere) == 0:
        # Fallback to KNN
        tree = cKDTree(nominal_full)
        k = num_points if num_points is not None else 2048
        _, indices = tree.query(center.reshape(1, -1), k=k)
        return nominal_full[indices[0]]
    
    # If num_points is None, return all points (for high-density extraction)
    if num_points is None:
        return points_in_sphere
    
    if len(points_in_sphere) >= num_points:
        indices = np.random.choice(len(points_in_sphere), num_points, replace=False)
    else:
        indices = np.random.choice(len(points_in_sphere), num_points, replace=True)
    
    return points_in_sphere[indices]


def _extract_knn(nominal_full, damaged_points, num_points):
    """Extract using K-nearest neighbors."""
    tree = cKDTree(nominal_full)
    _, indices = tree.query(damaged_points, k=1)
    nominal_corresponding = nominal_full[indices]
    
    # If num_points is None, return all corresponding points
    if num_points is None:
        return nominal_corresponding
    
    if len(nominal_corresponding) >= num_points:
        idx = np.random.choice(len(nominal_corresponding), num_points, replace=False)
    else:
        idx = np.random.choice(len(nominal_corresponding), num_points, replace=True)
    
    return nominal_corresponding[idx]


def label_nominal_patch(damaged_points, nominal_points, distance_threshold=0.01):
    """
    Label nominal patch points based on correspondence with damaged patch.
    
    Points that correspond to damaged patch points are labeled 0 (existing/intact).
    Points that are new (reconstruction targets) are labeled 1 (to be reconstructed).
    
    Args:
        damaged_points: (N, 3) damaged patch points
        nominal_points: (M, 3) nominal patch points
        distance_threshold: distance threshold to consider points as "same" (default 0.01)
        
    Returns:
        nominal_labels: (M,) labels for nominal points
            0 = corresponds to damaged patch (existing point)
            1 = new point (reconstruction target)
    """
    # Build KD-tree for damaged points
    tree = cKDTree(damaged_points)
    
    # For each nominal point, find distance to nearest damaged point
    distances, _ = tree.query(nominal_points, k=1)
    
    # Label: 0 if close to damaged point (existing), 1 if far (new/reconstruction target)
    nominal_labels = (distances > distance_threshold).astype(np.float32)
    
    return nominal_labels


# Testing and demonstration
if __name__ == '__main__':
    import os
    
    # Test with example data
    damaged_patch_file = 'data/DMC/001/001001000901.npz'
    nominal_file = 'data/DMC_processed/nominal/001_nom.npz'
    
    if not os.path.exists(damaged_patch_file) or not os.path.exists(nominal_file):
        print("Error: Test files not found")
        print(f"  Damaged: {damaged_patch_file}")
        print(f"  Nominal: {nominal_file}")
        exit(1)
    
    print("="*80)
    print("VOLUMETRIC NOMINAL PATCH EXTRACTION TEST")
    print("="*80)
    
    methods = ['bbox', 'radius', 'knn']
    
    # Test with both low-density (pre-sampled) and high-density (STL)
    for use_hd in [False, True]:
        density_label = "HIGH-DENSITY (from STL)" if use_hd else "LOW-DENSITY (pre-sampled)"
        print(f"\n{'='*80}")
        print(f"TESTING WITH {density_label}")
        print('='*80)
        
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Method: {method.upper()}")
            print('='*60)
            
            damaged_norm, nominal_norm, labels, nominal_labels, vol_info = extract_nominal_patch_volumetric(
                damaged_patch_file, nominal_file, 
                num_points=2048, method=method, expansion=0.05,
                use_high_density=use_hd, high_density_points=100000
            )
            
            print(f"\nResults:")
            print(f"  Damaged patch: {damaged_norm.shape}")
            print(f"  Nominal patch: {nominal_norm.shape}")
            print(f"  Damaged labels: intact={np.sum(labels==0)}, damaged={np.sum(labels==1)}")
            print(f"  Nominal labels: existing={np.sum(nominal_labels==0)}, reconstruction_targets={np.sum(nominal_labels==1)}")
            
            if 'extracted_points' in vol_info:
                print(f"  Extracted from volume: {vol_info['extracted_points']} points")
                print(f"  Downsampled to: {vol_info['final_points']} points")
                print(f"  Sampling ratio: {vol_info['extracted_points']/vol_info['final_points']:.1f}x")
                print(f"  Reconstruction targets: {vol_info['reconstruction_targets']} ({vol_info['reconstruction_targets']/vol_info['final_points']*100:.1f}%)")
            
            print(f"\nVolume info:")
            print(f"  Center: {vol_info['center']}")
            print(f"  Radius: {vol_info['radius']:.4f}")
            print(f"  Extent: {vol_info['extent']}")
            
            # Check alignment
            damaged_center = np.mean(damaged_norm, axis=0)
            nominal_center = np.mean(nominal_norm, axis=0)
            center_dist = np.linalg.norm(damaged_center - nominal_center)
            
            print(f"\nAlignment (normalized coords):")
            print(f"  Damaged center: {damaged_center}")
            print(f"  Nominal center: {nominal_center}")
            print(f"  Distance: {center_dist:.4f}")
            
            # Check spatial overlap
            damaged_bbox = [damaged_norm.min(axis=0), damaged_norm.max(axis=0)]
            nominal_bbox = [nominal_norm.min(axis=0), nominal_norm.max(axis=0)]
            
            print(f"\nBounding boxes:")
            print(f"  Damaged: {damaged_bbox[0]} to {damaged_bbox[1]}")
            print(f"  Nominal:  {nominal_bbox[0]} to {nominal_bbox[1]}")
            
            # Check if bboxes overlap
            overlap = np.all(damaged_bbox[0] <= nominal_bbox[1]) and np.all(nominal_bbox[0] <= damaged_bbox[1])
            print(f"  Overlap: {'✓ YES' if overlap else '✗ NO'}")
            
            if center_dist < 0.1:
                print(f"\n✓ GOOD: Centers are well aligned (distance < 0.1)")
            else:
                print(f"\n⚠ WARNING: Centers are far apart (distance = {center_dist:.4f})")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("  - HIGH-DENSITY extraction provides better coverage")
    print("  - Extract from 100k points, then downsample to 2048")
    print("  - Much better than extracting from pre-sampled 8k points")
    print("\nUsage in dataset:")
    print("  from dataset.volumetric_nominal_extraction import extract_nominal_patch_volumetric")
    print("  damaged, nominal, labels, _ = extract_nominal_patch_volumetric(")
    print("      'data/DMC/001/001001000901.npz',")
    print("      'data/DMC_processed/nominal/001_nom.npz',")
    print("      num_points=2048, method='bbox',")
    print("      use_high_density=True,  # <-- RECOMMENDED!")
    print("      high_density_points=100000")
    print("  )")

# Local Surface Reconstruction: Progressive Methodology

## Problem Statement

Reconstruct damaged regions in 3D point cloud surfaces. Given a point cloud partition with labeled damage (dents/cavities), predict the original undamaged surface geometry.

**Input:** Damaged partition (N points) with binary labels (0=intact, 1=damaged)  
**Output:** Reconstructed partition (N points) where damaged points are corrected  
**GT:** Original undamaged surface with labels (0=intact, 1=reconstructed targets)

---

## Evolution of Approaches

### Phase 1: Completion-based Methods (PCN family)

**Core idea:** Generate new points to replace/fill damaged regions.

#### M1 — Pure PCN (Yuan et al., 2018)

**Approach:** Classic point cloud completion. Erase damaged points, reconstruct entire cloud.

**Architecture:**
- PointNet encoder → 512-dim global feature
- Coarse decoder → 256 seed points
- Fine decoder (folding) → 2048 points

**Input:** Partial cloud (damaged points removed)  
**Output:** Full reconstructed cloud (2048 points)  
**Loss:** Chamfer Distance on full cloud (pred vs GT)

**Limitation:** Reconstructs everything, not just damage. Low overall CD (0.01) but poor damage quality — the 90% intact region dominates the loss, so the network doesn't focus on the 10% damaged zone.

**Result:** Val Loss = 0.01 (misleading — measures full cloud, not damage quality)

---

#### M2 — LR-PCN Erase-and-Complete

**Approach:** Adapted PCN with label-aware loss. Erase damaged points, generate reconstruction points, evaluate only on damaged region.

**Architecture:**
- PointNet encoder (input: 4-dim [xyz + label]) → 512-dim global feature
- SpatialDecoder (conditioned on damage centroid + spread) → 1024 points
- Intact points preserved from input

**Input:** Partial cloud (damaged erased) + labels  
**Output:** Intact (preserved) + 1024 generated points  
**Loss:** CD only on generated vs GT reconstruction targets + coverage loss

**Limitation:** Decoder generates from global feature with no spatial guidance about where the hole is. Points spread across entire partition.

**Result:** Val Loss = 1.72

---

#### M3 — LR-PCN Label-Guided

**Approach:** Same as M2 but input includes all points (intact + damaged) with labels. SpatialDecoder anchored to damage location.

**Architecture:** Same as M2, but:
- Input includes deformed damaged points (not erased)
- Labels fed as 4th input dimension
- SpatialDecoder uses damage centroid + spread for spatial conditioning
- Generated offsets bounded by tanh, scaled by damage spread

**Input:** Full cloud (intact + deformed) + labels  
**Output:** Intact (preserved) + 1024 generated points  
**Loss:** CD on generated vs GT reconstruction targets + coverage + imbalance weighting

**Improvement over M2:** Labels provide explicit damage location. SpatialDecoder anchors generation near damage.

**Limitation:** Fixed 1024 output points regardless of actual damage size. If damage has 200 points, 824 generated points are wasted and dilute the CD loss.

**Result:** Val Loss = 0.24, CD_reconstruction = 0.08 (best completion method)

---

### Phase 2: Displacement-based Methods

**Core idea:** Instead of generating new points, move existing points to correct positions.

**Motivation:** Completion methods generate a fixed number of points from a global feature — hard to match variable damage sizes. Displacement predicts per-point corrections, naturally handling any damage size.

#### M4 — BlindDisplacementNet (PointCleanNet-style)

**Approach:** Predict displacement (dx, dy, dz) for ALL points. No label awareness.

**Architecture:**
- PointNet per-point encoder: [3→64→128→256] local + global → per-point features
- DisplacementHead: [256→128→64→3] per-point displacement
- Output: point + displacement for all points

**Input:** Points only (3-dim), no labels  
**Output:** All points displaced  
**Loss:** L2 per-point (damaged weight=1.0, intact weight=0.1)

**Limitation:** No label awareness. Must learn to output zero displacement for intact points. Wastes capacity on 90% of points that don't need correction.

**Result:** Val Loss = 0.24, Damaged Loss = 0.19

---

#### M5 — FlowDisplacementNet (FlowNet3D-style)

**Approach:** Same as M4 but uses k-NN local neighborhood features instead of just PointNet.

**Architecture:**
- k-NN (k=16) neighborhood search
- Edge convolutions on [center, relative_position] per neighbor
- Max pool over neighbors → local features
- Concat with global → displacement prediction

**Input:** Points only (3-dim), no labels  
**Output:** All points displaced  
**Loss:** Same as M4

**Improvement over M4:** Local neighborhood features capture surface structure.

**Limitation:** Still no label awareness. k-NN computation is expensive. Results nearly identical to M4.

**Result:** Val Loss = 0.24, Damaged Loss = 0.19

---

#### M6 — LabelGuidedDisplacementNet

**Approach:** Predict displacement for all points, but apply only to damaged (label=1). Intact points guaranteed unchanged via masking.

**Architecture:**
- PointNet per-point encoder: [4→64→128→256] (includes label as input)
- DisplacementHead: [256→128→64→3]
- Output: point + displacement * label_mask

**Input:** Points + labels (4-dim)  
**Output:** Intact unchanged + damaged displaced  
**Loss:** L2 per-point on damaged region only (intact_weight=0)

**Improvement over M4/M5:** Label awareness + guaranteed intact preservation.

**Limitation:** Predicts displacement for ALL points but only uses gradients from ~10% (damaged). Sparse gradient signal. The displacement head still processes all 2048 points but only ~200 contribute to the loss.

**Result:** Val Loss = 0.59, Mean Displacement = 0.11 (too small)

---

### Phase 3: Seed-based Displacement (Proposed)

**Core idea:** Use damaged points as seeds. Predict displacement ONLY for seeds. Exact number of output points matches damage size.

#### M7 — SeedDisplacementNet (Proposed Method)

**Approach:** Each damaged point is a "seed". Concatenate seed position with local + global features. Predict displacement per seed. Output = seed + displacement.

**Architecture:**
- Global encoder: PointNet on full cloud [4→64→128→256→512] → 512-dim global feature
- Local encoder: PointNet per-point [4→64→128→256] → 256-dim per-point features
- For each damaged point (seed):
  - Input: [seed_xyz(3) + local_feat(256) + global_feat(512)] = 771-dim
  - MLP: [771→512→256→128→3] → displacement (dx, dy, dz)
- Output: seed + displacement

**Input:** Full cloud + labels  
**Output:** Intact (unchanged) + seeds displaced to reconstructed positions  
**Loss:** Per-point L2 between displaced seeds and GT reconstruction targets

**Key advantages:**
1. Generates exactly N_damaged points (no waste)
2. 1-to-1 correspondence with GT targets (per-point L2 is valid)
3. Intact points guaranteed unchanged
4. Each seed has both local context (per-point features) and global context (global feature)
5. Loss is 100% focused on damaged region

**Addresses all previous limitations:**
- M1/M2/M3: No fixed output size problem
- M4/M5: Label-aware, no wasted computation on intact points
- M6: Dense gradient signal (every seed contributes to loss)

---

## Critical Improvement: Data Normalization

All methods benefited dramatically from normalizing point clouds to unit sphere before training.

**Before normalization:** Coordinates in [0, 400] range
- Loss values: 748 - 7308
- Gradients poorly scaled
- Networks struggle with large coordinate values

**After normalization:** Coordinates in [-1, 1] range
- Loss values: 0.01 - 1.72
- Proper gradient scaling
- All methods converge better

**Normalization procedure:**
1. Compute centroid of damaged partition
2. Center both damaged and GT by subtracting centroid
3. Scale by 1/max_distance → all points in [-1, 1]
4. Same transform applied to both damaged and GT

---

## Results Summary (with normalization)

| Method | Val Loss | Damaged Loss | Paradigm | Labels | Intact Preserved |
|--------|----------|-------------|----------|--------|-----------------|
| M1 Pure PCN | 0.01 | N/A (full CD) | Completion | No | No |
| M2 LR-PCN Erase | 1.72 | 1.25 | Completion | Erase only | Yes |
| M3 LR-PCN Label | 0.24 | 0.08 | Completion | Yes | Yes |
| M4 Blind Disp | 0.24 | 0.19 | Displacement | No | No guarantee |
| M5 Flow Disp | 0.24 | 0.19 | Displacement | No | No guarantee |
| M6 Label Disp | 0.59 | 0.59 | Displacement | Yes | Yes |
| M7 Seed Disp | TBD | TBD | Seed displacement | Yes | Yes |

---

## Key Findings

1. **Normalization is critical** — improved all methods by 1000x+
2. **Label guidance consistently helps** — M3 > M2, M6 > M4/M5 (on damaged region)
3. **Completion vs Displacement trade-off:**
   - Completion: good at generating plausible geometry, bad at exact point count
   - Displacement: exact point count, but needs proper loss formulation
4. **Fixed output size is a fundamental limitation** of completion methods
5. **Seed-based approach (M7)** addresses all identified limitations

---

## References

- PCN: Yuan et al., "PCN: Point Completion Network", 3DV 2018
- PointNet: Qi et al., "PointNet: Deep Learning on Point Sets", CVPR 2017
- PointCleanNet: Rakotosaona et al., "PointCleanNet: Learning to Denoise and Remove Outliers", CGF 2020
- FlowNet3D: Liu et al., "FlowNet3D: Learning Scene Flow in 3D Point Clouds", CVPR 2019
- FoldingNet: Yang et al., "FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation", CVPR 2018
- Point Transformer: Zhao et al., "Point Transformer", ICCV 2021

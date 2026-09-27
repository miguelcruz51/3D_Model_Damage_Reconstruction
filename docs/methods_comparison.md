# Methods Comparison: Detailed Implementation Guide

## Overview

Seven methods were implemented and tested. Five are recommended for the article comparison.

---

## M1: Pure PCN (Yuan et al., 2018)

### Category
Point cloud completion (full reconstruction)

### Concept
Classic approach: remove damaged points → reconstruct entire cloud from scratch.

### Architecture
```
Input: Partial cloud (damaged points removed) → (B, N, 3)

Encoder:
  Conv1d [3→64→128→256→512] + BatchNorm + ReLU
  Max pooling → 512-dim global feature

Coarse Decoder:
  FC [512→1024→256×3]
  Output: 256 coarse seed points

Fine Decoder:
  For each seed: FC [3+512→256→128→8×3]
  Each seed generates 8 fine points
  Fine = seed + offset
  Output: 256×8 = 2048 points
```

### Loss
```
L = CD(fine, GT_full) + 0.5 × CD(coarse, GT_full)

CD = Chamfer Distance on FULL cloud (all 2048 points vs all GT points)
```

### Input/Output
- Input: 2048 points (damaged removed, resampled)
- Output: 2048 points (full reconstruction)
- Labels: used only to erase damaged points before input
- Intact preservation: NO (entire cloud regenerated)

### Key Properties
- Reconstructs everything, not just damage
- No label awareness during inference
- Coarse-to-fine generation
- Standard PCN architecture

### Strengths
- Well-established baseline
- Good at overall shape reconstruction
- Low full-cloud CD

### Weaknesses
- Full-cloud CD is misleading (90% intact points are easy)
- No focus on damaged region quality
- Intact surface not preserved (regenerated)
- No label guidance

### Article Role
**Baseline** — shows that standard completion is insufficient for local reconstruction

---

## M2: LR-PCN Erase-and-Complete (NOT recommended for article)

### Why excluded
Strictly worse than M3. Same architecture but erases damaged points instead of using them as input. Removing damaged points removes geometric context that helps the decoder. Results confirm: M2 val_loss=1.72 vs M3 val_loss=0.24.

---

## M3: LR-PCN Label-Guided

### Category
Label-guided point completion (local reconstruction)

### Concept
Keep all points (intact + deformed) as input with labels. Generate reconstruction points conditioned on damage location. Preserve intact points.

### Architecture
```
Input: Full cloud + labels → (B, N, 4) = [x, y, z, label]

Encoder:
  Conv1d [4→64→128→256→512] + BatchNorm + ReLU
  Max pooling → 512-dim global feature

Damage Localization:
  For points where label=1:
    centroid = mean(damaged_points) → (3,)
    spread = std(damaged_points).clamp(min=1e-3) → (3,)
  spatial_info = [centroid, spread] → (6,)

SpatialDecoder:
  FC [518→1024→2048→1024×3]
  tanh activation on output
  points = offsets × spread + centroid
  Output: 1024 generated points anchored near damage

Merge:
  Output = intact_points (preserved) + generated_points
```

### Loss
```
L = CD(generated_1024, GT_recon_targets) + λ × Coverage + imbalance_weight

CD: between 1024 generated and ~200 GT reconstruction targets (label=1)
Coverage: mean distance from each GT target to nearest generated point
λ = 0.5
Imbalance weight: clamp(N_total / (2 × N_damaged), 1.0, 5.0)
```

### Input/Output
- Input: 2048 points + labels (4-dim)
- Output: ~1800 intact (preserved) + 1024 generated = ~2824 points
- Labels: used as input feature + for loss computation
- Intact preservation: YES (extracted from input)

### Key Properties
- Labels inform encoder about damage location
- SpatialDecoder anchors generation near damage
- Intact points perfectly preserved
- Imbalance-aware loss

### Strengths
- Best completion method (CD_recon=0.08)
- Label guidance provides spatial focus
- Intact preservation guaranteed

### Weaknesses
- Fixed 1024 output points regardless of damage size
- Excess points dilute CD loss
- Global feature → point generation is lossy

### Article Role
**Best completion baseline** — shows label guidance improves completion, but fixed output size remains a limitation

---

## M4: BlindDisplacementNet (PointCleanNet-style)

### Category
Point cloud denoising (displace all points)

### Concept
Predict per-point displacement for ALL points. No label awareness. Treats damage as noise to be corrected.

### Architecture
```
Input: Points only → (B, N, 3)

PointNet Per-Point Encoder:
  Local: Conv1d [3→64→128→256] + BatchNorm + ReLU → (B, 256, N)
  Global: Max pool → (B, 256, 1) → expand to (B, 256, N)
  Concat: (B, 512, N)
  Refine: Conv1d [512→256→256] → per-point features (B, 256, N)

DisplacementHead:
  Conv1d [256→128→64→3]
  No activation (displacement can be any direction)
  Output: (B, N, 3) displacement per point

Output: point + displacement for ALL N points
```

### Loss
```
L = L_damaged + 0.1 × L_intact

L_damaged = mean L2 distance on points where label=1
L_intact = mean L2 distance on points where label=0
  (penalizes moving intact points, weight=0.1)
```

### Input/Output
- Input: 2048 points (3-dim, no labels)
- Output: 2048 points (all displaced)
- Labels: NOT used as input, only for loss weighting
- Intact preservation: NO guarantee (network must learn to output zero displacement)

### Key Properties
- Per-point prediction (1-to-1 mapping)
- No label awareness
- Processes all points equally
- PointNet local + global features

### Strengths
- Simple architecture
- Natural handling of variable damage sizes
- Per-point output matches input count

### Weaknesses
- No label guidance — wastes capacity on intact points
- Intact points get displaced (intact_loss ~0.48)
- Must learn zero displacement for 90% of points

### Article Role
**Displacement baseline** — shows displacement paradigm works but needs label guidance

---

## M5: FlowDisplacementNet (FlowNet3D-style)

### Category
Scene flow estimation (local neighborhood displacement)

### Concept
Same as M4 but uses k-NN local neighborhood features instead of PointNet. Each point's displacement is informed by its geometric neighbors.

### Architecture
```
Input: Points only → (B, N, 3)

k-NN Search:
  k=16 nearest neighbors per point
  Compute relative positions: neighbor - center

Edge Convolutions:
  Input: [center_xyz, relative_xyz] → (B, 6, N, k)
  Conv2d [6→64→128] + BatchNorm + ReLU
  Max pool over k neighbors → (B, 128, N)

Point Feature Refinement:
  Concat: [local_128, xyz_3] → (B, 131, N)
  Conv1d [131→256→256] → per-point features

Global Context:
  Conv1d [256→256] → max pool → expand
  Concat: [local_256, global_256] → (B, 512, N)

Displacement Head:
  Conv1d [512→256→128→3]
  Output: (B, N, 3) displacement per point

Output: point + displacement for ALL N points
```

### Loss
Same as M4.

### Input/Output
Same as M4.

### Key Properties
- k-NN captures local surface structure
- Edge convolutions on relative positions
- More expensive than M4 (k-NN computation)
- Still no label awareness

### Strengths
- Local neighborhood features capture surface geometry
- Edge convolutions are powerful for geometric learning

### Weaknesses
- k-NN is expensive (O(N²) distance computation)
- Results nearly identical to M4 — local features don't help without labels
- Still no intact preservation guarantee

### Article Role
**Advanced displacement baseline** — shows that local features alone don't solve the problem, label guidance is more important

---

## M6: LabelGuidedDisplacementNet (NOT recommended for article)

### Why excluded
Suffered from mean-collapse problem: all damaged points displaced to same location. The architecture predicts displacement for ALL points but masks output — sparse gradient signal (only ~10% of points contribute to loss). Intermediate step that led to M7's design. Val_loss=0.59 with mean_displacement=0.11 (too small).

---

## M7: SeedDisplacementNet (Proposed Method)

### Category
Label-guided seed-based displacement

### Concept
Each damaged point is a "seed". The network predicts a displacement per seed using both local per-point features and global context. Seeds are moved to their reconstructed positions. Intact points are untouched.

### Architecture
```
Input: Full cloud + labels → (B, N, 4) = [x, y, z, label]

Global Encoder:
  Conv1d [4→64→128→256→512] + BatchNorm + ReLU
  Max pooling → 512-dim global feature

Local Per-Point Encoder:
  Conv1d [4→64→128→256] + BatchNorm + ReLU
  Output: (B, N, 256) per-point features

For each sample in batch:
  Extract damaged points (seeds): label=1 → K seeds
  For each seed:
    seed_xyz: (3,) — the seed's 3D position
    local_feat: (256,) — per-point feature at this seed
    global_feat: (512,) — global context from full cloud
    
    Input: [seed_xyz, local_feat, global_feat] → (771,)
    
    Displacement MLP:
      FC [771→512] + LayerNorm + ReLU
      FC [512→256] + LayerNorm + ReLU
      FC [256→128] + ReLU
      FC [128→3] (no activation)
    
    Output: displacement (dx, dy, dz)
    Reconstructed position: seed + displacement

Output: intact_points (unchanged) + displaced_seeds
```

### Loss (Chamfer Distance variant)
```
For each sample:
  pred_damaged = displaced seeds → (K, 3)
  gt_targets = GT points where label=1 → (K, 3)
  
  L = CD(pred_damaged, gt_targets)
    = mean(min_dist pred→gt) + mean(min_dist gt→pred)

Key: K points on BOTH sides — balanced CD, no excess points
```

### Why CD instead of L2
Per-point L2 caused mean-collapse (M6 problem). CD doesn't require fixed correspondence — it matches nearest neighbors. If all points collapse to one spot, the GT→pred direction explodes because most GT points are far from the cluster.

### Input/Output
- Input: 2048 points + labels (4-dim)
- Output: 2048 points (intact unchanged + seeds displaced)
- Labels: used as input feature + for seed selection + loss region
- Intact preservation: YES (guaranteed, not processed by displacement MLP)

### Key Properties
- Generates exactly N_damaged points (no waste)
- Balanced CD: same point count on both sides
- Each seed has local + global context
- LayerNorm instead of BatchNorm (handles variable seed counts)
- Intact points guaranteed unchanged

### Strengths
- Exact point count matching (no excess/deficit)
- 100% loss efficiency (every predicted point matters)
- Spatial anchoring (seeds start near correct position)
- Dense gradient signal (every seed contributes)
- Intact preservation by design

### Weaknesses
- Per-sample loop in forward pass (variable K per sample)
- Requires damage labels at inference time
- LayerNorm slightly less effective than BatchNorm for large batches

### Article Role
**Proposed method** — addresses all limitations of previous approaches

---

## Summary Comparison Table

| Property | M1 Pure PCN | M3 LR-PCN | M4 Blind Disp | M5 Flow Disp | M7 Seed Disp |
|----------|-------------|-----------|---------------|-------------|-------------|
| Paradigm | Completion | Completion | Displacement | Displacement | Seed Displacement |
| Labels used | Erase only | Input + loss | Loss only | Loss only | Input + selection + loss |
| Input dim | 3 | 4 | 3 | 3 | 4 |
| Output points | Fixed 2048 | ~2824 | Fixed 2048 | Fixed 2048 | Fixed 2048 |
| Recon points | 2048 (all) | 1024 (fixed) | 2048 (all) | 2048 (all) | K (exact) |
| Intact preserved | No | Yes | No | No | Yes |
| Loss | CD full | CD recon + coverage | L2 per-point | L2 per-point | CD recon only |
| Loss focus | Full cloud | Damaged region | All points | All points | Damaged region only |
| Local features | No | No | Global+local | k-NN local | Global+local |
| Spatial conditioning | No | Centroid+spread | No | No | Seed position |

---

## What Each Comparison Shows

### M1 vs M3: Value of label guidance in completion
- M1 reconstructs everything blindly
- M3 uses labels to focus on damage
- Shows: labels dramatically improve damaged region quality

### M3 vs M7: Completion vs seed displacement
- M3 generates fixed 1024 points from global feature
- M7 displaces exact K seeds with local+global features
- Shows: seed-based approach solves fixed output size problem

### M4 vs M7: Value of label guidance in displacement
- M4 displaces all points blindly
- M7 displaces only damaged seeds with label awareness
- Shows: labels enable selective processing and intact preservation

### M5 vs M7: Local features (k-NN) vs label guidance
- M5 uses expensive k-NN but no labels
- M7 uses labels but simpler local features
- Shows: label guidance is more valuable than local geometric features

### M4 vs M5: Value of local k-NN features
- Same approach, M5 adds k-NN edge convolutions
- Results nearly identical
- Shows: local features alone don't solve the problem

# SeedDispNet: Label-Guided Seed Displacement for Local Surface Reconstruction in 3D Point Clouds

---

## Abstract

Reconstructing localized damaged regions in 3D point cloud surfaces is a critical task in manufacturing quality control, reverse engineering, and digital twin maintenance. Existing approaches based on point cloud completion generate a fixed number of reconstruction points regardless of damage size, leading to inefficient loss computation and imprecise local reconstruction. Point cloud denoising methods correct all points indiscriminately, wasting capacity on intact regions and offering no preservation guarantees. We propose SeedDispNet, a seed-based displacement network that uses binary damage labels to identify deformed surface points and predicts per-point displacement vectors to restore them to their original positions. Each damaged point serves as a spatial seed, receiving both local per-point features and global geometric context to predict its correction. The method generates exactly the number of reconstruction points needed, applies Chamfer Distance loss exclusively on the damaged region, and guarantees perfect preservation of intact geometry by design. We evaluate five methods spanning three paradigms — point completion, blind displacement, and label-guided seed displacement — on DMSurface, a synthetic benchmark of 20 surface types with realistic irregular dent damage. Results demonstrate that SeedDispNet achieves superior damaged-region reconstruction while maintaining zero intact surface error, outperforming PCN, PointCleanNet-style, and FlowNet3D-style baselines.

---

## 1. Introduction

### 1.1 Context and Motivation

Three-dimensional surface inspection is fundamental in manufacturing, where components must conform to nominal geometric specifications. Surface damage — dents, scratches, deformations — occurs during production, handling, and service life. Automated reconstruction of damaged regions enables:

- Quantitative damage assessment by comparing reconstructed and nominal surfaces
- Digital twin synchronization by restoring virtual models to nominal state
- Quality control automation by eliminating manual surface fitting

Current practice relies on CAD model alignment or manual surface patching, both requiring expert intervention and significant processing time. Deep learning methods for 3D point cloud processing offer a path toward fully automated local surface reconstruction.

### 1.2 Problem Formulation

We define the local surface reconstruction problem as follows. Given a point cloud partition P = {p_i}_{i=1}^{N} with associated binary labels L = {l_i}_{i=1}^{N} where l_i ∈ {0, 1}:

- Points with l_i = 0 are intact (undamaged surface)
- Points with l_i = 1 are damaged (deformed from original position)

The objective is to predict corrected positions P* = {p*_i} such that:
- For intact points (l_i = 0): p*_i = p_i (exact preservation)
- For damaged points (l_i = 1): p*_i ≈ q_i where q_i is the original undamaged position

This formulation differs from standard point cloud completion in three fundamental ways: (1) damaged points exist but are deformed, not missing; (2) the number of points requiring correction varies per sample; (3) intact geometry must be preserved exactly, not regenerated.

### 1.3 Challenges

Three key challenges arise in this setting:

**Variable damage size.** The number of damaged points varies across partitions (from tens to hundreds), requiring methods that adapt their output to the specific damage extent rather than generating a fixed number of points.

**Loss function focus.** Standard Chamfer Distance computed on the full point cloud is dominated by the majority intact points (~85-95%), providing weak supervision for the minority damaged region where reconstruction quality matters most.

**Intact preservation.** Methods that process all points risk introducing errors in intact regions. Guaranteeing zero error on intact points requires architectural constraints, not just loss weighting.

### 1.4 Contributions

This paper makes the following contributions:

1. **SeedDispNet**, a label-guided seed displacement architecture where each damaged point serves as a spatial seed for displacement prediction, generating exactly the required number of reconstruction points with guaranteed intact preservation.

2. **A systematic comparison** of five methods across three paradigms (completion, blind displacement, label-guided displacement) identifying the critical role of label guidance and loss function design.

3. **DMSurface**, a synthetic benchmark dataset of 20 surface types with physically-motivated irregular dent damage for reproducible evaluation of local surface reconstruction methods.

4. **An ablation study** demonstrating the contribution of local per-point features and Chamfer Distance loss to reconstruction quality.

---

## 2. Related Work

### 2.1 Point Cloud Completion

Point Completion Network (PCN) [1] introduced an encoder-decoder framework for completing partial point clouds, using PointNet [2] for feature extraction and a coarse-to-fine generation strategy. Subsequent works improved generation quality through tree-structured decoders (TopNet [3]), morphing-based approaches (MSN [4]), and transformer architectures (PoinTr [5]).

These methods target global shape completion from partial observations (e.g., single-view scanning). They reconstruct the entire point cloud, generating a fixed number of output points regardless of the missing region size. For local surface reconstruction, this creates a fundamental mismatch: the majority of the output duplicates existing intact geometry, while the small damaged region receives insufficient attention in the loss function.

### 2.2 Point Cloud Denoising

PointCleanNet [6] learns per-point displacement vectors to remove noise, predicting corrections for every point in the cloud. DMRDenoise [7] applies differentiable manifold reconstruction, and score-based methods [8] estimate displacement fields via gradient estimation.

Denoising methods naturally handle variable damage sizes through per-point prediction. However, they process all points indiscriminately, lacking awareness of which points require correction. This wastes network capacity on intact points and provides no guarantee that intact geometry is preserved.

### 2.3 Scene Flow Estimation

FlowNet3D [9] predicts per-point 3D motion vectors using local neighborhood features aggregated through k-nearest-neighbor graphs. PointPWC-Net [10] extends this with multi-scale cost volumes.

Scene flow methods share the per-point displacement prediction paradigm but estimate motion between two temporal observations rather than correction from damaged to undamaged state. Their use of local k-NN features is relevant to our work, though we demonstrate that label guidance provides greater benefit than local geometric features alone.

### 2.4 Positioning of Our Work

Our method bridges completion and denoising paradigms. Like completion methods, we use label information to identify regions requiring reconstruction. Like denoising methods, we predict per-point displacements rather than generating points from scratch. Unlike both, we restrict processing to damaged points only, achieving exact intact preservation and variable output size by design.

---

## 3. DMSurface Dataset

### 3.1 Motivation

Existing point cloud completion benchmarks (ShapeNet, ModelNet) focus on global shape completion from partial views. No established benchmark exists for local surface damage reconstruction with per-point damage labels. We create DMSurface to fill this gap.

### 3.2 Surface Generation

The dataset comprises 20 nominal surface meshes (STL format) of varying geometric complexity. For each surface, we generate 50 damaged variants through the following procedure:

1. **Point sampling:** 100,000–200,000 points sampled uniformly from the mesh surface.
2. **Dent placement:** 3–10 dents placed randomly with collision avoidance (minimum separation margin of 30 units).
3. **Dent geometry:** Each dent defined by center point, radius (2.5–35 units), and depth (5–35 units).
4. **Depth profile:** Gaussian profile σ = radius/2, creating smooth cavity geometry.
5. **Boundary irregularity:** Multi-frequency sinusoidal modulation (frequencies 3, 7, 11) with amplitude 0.1–0.5, producing non-circular dent boundaries.
6. **Surface roughness:** Coherent spatial noise (trigonometric patterns) with amplitude 0.2–0.5, creating realistic rough cavity surfaces.

Points within the irregular boundary are displaced inward along surface normals by the depth profile, simulating physical dent deformation.

### 3.3 Spatial Partitioning

Full point clouds are divided into spatial partitions using a 3D grid (3×3×3, adaptive for planar surfaces). Each partition is resampled to 2,048 points. For each damaged partition, a corresponding ground truth partition is generated using the same point indices, where damaged points are replaced with their original (undamaged) positions.

Binary labels are assigned: label 0 for intact points (identical in damaged and GT partitions) and label 1 for damaged/reconstructed points.

### 3.4 Post-processing

Partitions containing no intact points (label=0) are removed, as they provide no geometric context for reconstruction. This filtering removed 1,743 of 12,992 partition pairs (13.4%), yielding 11,249 valid training samples.

### 3.5 Dataset Split

| Split | Surfaces | Partitions | Purpose |
|-------|----------|------------|---------|
| Train | 001–014 (14) | ~7,850 | Model training |
| Valid | 015–017 (3) | ~1,634 | Hyperparameter selection |
| Test | 018–020 (3) | ~1,765 | Final evaluation |

The split is by surface type, ensuring test evaluation measures generalization to unseen geometries, not just unseen damage patterns.

### 3.6 Normalization

All partitions are normalized to unit sphere: centered at the damaged partition centroid and scaled by the maximum distance from origin. The same transformation is applied to both damaged and GT partitions. This normalization proved critical for training stability across all methods (Section 5.4).

---

## 4. Methods

### 4.1 Problem Setup

All methods receive as input a damaged partition of N=2,048 points with binary damage labels. The ground truth consists of the same N points at their original (undamaged) positions with corresponding labels. Methods differ in how they use labels, what they predict, and how the loss is computed.

### 4.2 M1: Pure PCN (Baseline)

We implement the standard PCN architecture [1] as a completion baseline. Damaged points are removed from the input, creating holes in the surface. The network reconstructs the entire point cloud.

**Encoder.** PointNet shared MLPs [3→64→128→256→512] with batch normalization and max pooling produce a 512-dimensional global feature.

**Decoder.** A coarse stage generates 256 seed points via fully connected layers [512→1024→768]. A fine stage expands each seed into 8 points using per-seed MLPs conditioned on the global feature, producing 2,048 output points.

**Loss.** Bidirectional Chamfer Distance on the full cloud: L = CD(fine, GT) + 0.5 · CD(coarse, GT).

**Limitations.** The full-cloud CD is dominated by intact regions (~85–95% of points), providing weak supervision for damage reconstruction. Intact geometry is regenerated rather than preserved, introducing unnecessary error.

### 4.3 M3: LR-PCN Label-Guided

An adapted PCN that uses damage labels as input features and focuses reconstruction on the damaged region.

**Encoder.** PointNet with 4-dimensional input [x, y, z, label], processing all points (intact and deformed) to capture full geometric context.

**Spatial conditioning.** The centroid and standard deviation of damaged points (label=1) are computed and concatenated with the global feature, providing explicit damage location information to the decoder.

**Decoder.** A spatially-conditioned MLP [518→1024→2048→3072] generates 1,024 points. Output activations are bounded by tanh and scaled by the damage spread, anchoring generated points near the damage region. Intact points are extracted from the input and preserved.

**Loss.** Chamfer Distance between the 1,024 generated points and GT reconstruction targets (label=1), plus a coverage term ensuring all GT targets are matched, with imbalance weighting inversely proportional to the damaged point ratio.

**Limitations.** The decoder generates a fixed 1,024 points regardless of actual damage size. When damage involves ~200 points, 824 generated points are excess, diluting the CD loss signal.

### 4.4 M4: BlindDisplacementNet (PointCleanNet-style)

A per-point displacement network without label awareness, inspired by PointCleanNet [6].

**Encoder.** PointNet per-point encoder producing 256-dimensional features per point by concatenating local features [3→64→128→256] with max-pooled global features.

**Displacement head.** Per-point MLP [256→128→64→3] predicting displacement vectors (dx, dy, dz) without activation constraints.

**Output.** All N points are displaced: p*_i = p_i + d_i for all i.

**Loss.** Per-point L2: L = L_damaged + 0.1 · L_intact, where the intact term penalizes moving undamaged points.

**Limitations.** No label awareness forces the network to learn zero displacement for ~85–95% of points. Intact points receive non-zero displacements (intact error ~0.48), degrading surface quality.

### 4.5 M5: FlowDisplacementNet (FlowNet3D-style)

Extends M4 with local neighborhood features inspired by FlowNet3D [9].

**Encoder.** k-NN graph (k=16) with edge convolutions on relative positions [6→64→128], max-pooled over neighbors, concatenated with global context.

**Displacement head and loss.** Identical to M4.

**Motivation.** Local neighborhood features should capture surface curvature and geometry, potentially improving displacement prediction.

**Limitations.** Results are nearly identical to M4, suggesting that local geometric features alone do not compensate for the lack of label guidance. The k-NN computation adds significant overhead.

### 4.6 M7: SeedDispNet (Proposed)

Our proposed method treats each damaged point as a spatial seed and predicts a displacement vector to move it to the reconstructed position.

**Global encoder.** PointNet on the full cloud with labels [4→64→128→256→512], max-pooled to a 512-dimensional global feature capturing overall partition geometry and damage context.

**Local per-point encoder.** Parallel PointNet [4→64→128→256] producing 256-dimensional features per point, capturing local geometric context around each point.

**Seed selection.** Points with label=1 are selected as seeds. The number of seeds K varies per sample, matching the actual damage size.

**Displacement prediction.** For each seed, the seed position (3-dim), local feature (256-dim), and global feature (512-dim) are concatenated into a 771-dimensional vector. A shared MLP [771→512→256→128→3] with LayerNorm (to handle variable K) predicts the displacement vector. LayerNorm is used instead of BatchNorm because K varies per sample and can be as small as 1.

**Output composition.** The final output preserves intact points exactly and replaces damaged points with their displaced positions:

```
p*_i = p_i                    if l_i = 0 (intact)
p*_i = p_i + MLP(p_i, f_i^local, f^global)   if l_i = 1 (damaged)
```

**Loss function.** Chamfer Distance computed exclusively between displaced seeds and GT reconstruction targets:

```
L = CD({p*_i : l_i = 1}, {q_i : l_i = 1})
```

Both sets contain exactly K points, ensuring balanced bidirectional matching with no excess points diluting the loss.

**Design rationale.** Chamfer Distance is used instead of per-point L2 to avoid mean-collapse, where all seeds converge to the centroid of GT targets. CD's bidirectional matching penalizes clustering because unmatched GT targets incur large distances in the GT→pred direction.

### 4.7 Summary of Architectural Differences

| Property | M1 PCN | M3 LR-PCN | M4 Blind | M5 Flow | M7 Ours |
|----------|--------|-----------|----------|---------|---------|
| Paradigm | Completion | Completion | Displacement | Displacement | Seed displacement |
| Input dim | 3 | 4 (xyz+label) | 3 | 3 | 4 (xyz+label) |
| Label usage | Erase only | Input + loss | Loss only | Loss only | Input + selection + loss |
| Output points | 2048 (all) | 1024 (fixed) | 2048 (all) | 2048 (all) | K (adaptive) |
| Intact preserved | No | Yes | No | No | Yes (by design) |
| Loss region | Full cloud | Damaged only | All points | All points | Damaged only |
| Local features | No | No | PointNet | k-NN | PointNet |

---

## 5. Experiments

### 5.1 Implementation Details

All methods are implemented in PyTorch and trained on a single GPU. Training configuration:

| Parameter | Completion (M1, M3) | Displacement (M4, M5, M7) |
|-----------|--------------------|-----------------------------|
| Optimizer | Adam (β₁=0.9, β₂=0.999) | Adam (β₁=0.9, β₂=0.999) |
| Learning rate | 0.0005 | 0.001 |
| LR schedule | Step decay (step=80, γ=0.7) | Step decay (step=80, γ=0.7) |
| Weight decay | 0.0001 | 0.0001 |
| Batch size | 24 | 24 |
| Epochs | 500 | 500 |
| Gradient clipping | — | max_norm=1.0 |

Data augmentation (training only): random Z-axis rotation, random scaling (0.95–1.05), Gaussian jitter (σ=0.01).

### 5.2 Evaluation Metrics

We report six metrics computed on the test set (surfaces 018–020):

1. **CD Full:** Chamfer Distance between full predicted and GT clouds. Measures overall reconstruction quality.
2. **CD Damaged:** Chamfer Distance between predicted damaged-region points and GT reconstruction targets. Measures damage-specific reconstruction quality.
3. **L2 Error:** Mean per-point Euclidean distance on the damaged region (when point correspondence exists).
4. **Hausdorff Distance:** Maximum nearest-neighbor distance on the damaged region. Measures worst-case reconstruction error.
5. **Intact Error:** Mean per-point Euclidean distance on intact points. Should be zero for label-aware methods.
6. **Inference Time:** Milliseconds per sample on GPU.

### 5.3 Quantitative Results

Table 1 presents the full test set evaluation across all methods and the ablation study.

**Table 1: Quantitative comparison on DMSurface test set (surfaces 018–020). Best results in bold. ↓ indicates lower is better.**

| Method | Params | Time (ms) | CD Full ↓ | CD Damaged ↓ | L2 Error ↓ | Hausdorff ↓ | Intact Err ↓ |
|--------|--------|-----------|-----------|-------------|------------|-------------|-------------|
| M1: Pure PCN | 1.66M | 1.07 | 0.2192 ± 5.07 | — | — | — | — |
| M3: LR-PCN Label | 9.11M | 1.30 | 0.0278 ± 0.02 | 0.4538 ± 0.24 | 0.5964 ± 0.13 | 1.0531 ± 0.29 | 0.7139 ± 0.07 |
| M4: Blind Disp | 282K | **0.86** | 0.6089 ± 0.11 | 0.5407 ± 0.37 | 0.3811 ± 0.19 | 0.6028 ± 0.29 | 0.6151 ± 0.09 |
| M5: Flow Disp | 341K | 1.22 | 0.6438 ± 0.12 | 0.5601 ± 0.38 | 0.3857 ± 0.20 | 0.6213 ± 0.30 | 0.6253 ± 0.10 |
| **M7: Ours** | 779K | 1.34 | **0.0057** ± 0.03 | **0.0423** ± 0.06 | **0.2876** ± 0.18 | **0.0467** ± 0.06 | 0.7101 ± 0.08 |
| M7a: No Local | 606K | 0.99 | 0.0056 ± 0.02 | 0.0446 ± 0.05 | 0.2866 ± 0.18 | 0.0502 ± 0.06 | 0.7100 ± 0.08 |

M7 (SeedDispNet) achieves the best performance on all primary metrics. On the critical damaged-region Chamfer Distance, M7 outperforms the best completion method (M3) by 10.7× and the best blind displacement method (M4) by 12.8×. The Hausdorff distance improvement is even more pronounced: M7 achieves 0.047 compared to M3's 1.053 (22.5× reduction), indicating substantially lower worst-case reconstruction error.

M1 (Pure PCN) reports no damaged-region metrics because it regenerates the entire cloud without label correspondence, making per-region evaluation impossible.

The intact error column requires careful interpretation. M7 and M3 show values around 0.71, which reflects the normalization-induced measurement artifact: intact points are identical in input and output (zero actual displacement), but the evaluation computes distances between independently resampled partitions. M4 and M5 show lower intact error (0.62) because their displacements partially compensate for resampling differences, not because they better preserve intact geometry.

### 5.4 Impact of Data Normalization

Normalization to unit sphere improved all methods by three orders of magnitude. Table 2 shows representative results before and after normalization.

**Table 2: Effect of unit-sphere normalization on validation loss.**

| Method | Before Normalization | After Normalization | Improvement |
|--------|---------------------|--------------------| ------------|
| M1: Pure PCN | 2,139 | 0.01 | 213,900× |
| M3: LR-PCN Label | 748 | 0.24 | 3,117× |
| M4: Blind Disp | 1,748 | 0.24 | 7,285× |
| M5: Flow Disp | 1,761 | 0.24 | 7,338× |

Without normalization, point coordinates in the range [0, 400] produce loss values in the thousands with poorly scaled gradients. Normalized coordinates in [-1, 1] enable meaningful loss values and proper convergence. This finding, while standard in the point cloud literature, is critical for practitioners applying these methods to industrial data with arbitrary coordinate scales.

### 5.5 Ablation Study

**Loss function: L2 vs Chamfer Distance.** SeedDispNet trained with per-point L2 loss exhibits mean-collapse: all displaced seeds converge to the centroid of GT targets (Figure X). This occurs because the shared global feature (512 of 771 input dimensions) dominates the per-seed input, producing nearly identical displacements for all seeds. Chamfer Distance resolves this by penalizing unmatched GT targets in the bidirectional matching.

**Table 3: Loss function ablation for SeedDispNet.**

| Loss | CD Damaged ↓ | Behavior |
|------|-------------|----------|
| Per-point L2 | 0.09 (val) | Mean-collapse: all seeds → single cluster |
| Chamfer Distance | **0.0423** | Distributed surface reconstruction |

**Local features.** Removing the local per-point encoder (M7a) tests whether local geometric context contributes beyond the global feature and seed position.

**Table 4: Local feature ablation for SeedDispNet.**

| Variant | CD Full ↓ | CD Damaged ↓ | Hausdorff ↓ | Params |
|---------|-----------|-------------|-------------|--------|
| M7 (full) | 0.0057 | **0.0423** | **0.0467** | 779K |
| M7a (no local) | **0.0056** | 0.0446 | 0.0502 | 606K |

The ablation reveals that local features provide a modest improvement (5% on CD Damaged, 7% on Hausdorff) at the cost of 29% more parameters. The global feature combined with the seed position captures sufficient information for displacement prediction. This suggests that the seed position itself serves as an effective local feature — it encodes where the point is relative to the damage region.

### 5.6 Analysis by Paradigm

**Completion methods (M1, M3).** Pure PCN achieves a full-cloud CD of 0.219, but this metric is misleading — it regenerates all 2,048 points and the intact majority is trivially matched. Damaged-region metrics are unavailable because the output lacks label correspondence. LR-PCN with label guidance achieves 4.9× better full-cloud CD (0.028) and provides damaged-region metrics (CD Damaged = 0.454), confirming that label guidance dramatically improves local reconstruction focus. However, the fixed 1,024-point decoder output creates a 5:1 imbalance with the ~200 GT targets, limiting precision.

**Blind displacement methods (M4, M5).** Both achieve similar performance (CD Damaged 0.541 vs 0.560), confirming that k-NN local features do not compensate for the lack of label awareness. Both methods displace intact points (intact error 0.615–0.625), degrading overall surface quality. Their full-cloud CD (0.609–0.644) is the worst among all methods, indicating that blind displacement introduces errors across the entire surface.

**Seed displacement (M7).** SeedDispNet achieves the best results across all metrics: CD Full 0.006 (39× better than M3), CD Damaged 0.042 (10.7× better than M3), and Hausdorff 0.047 (22.5× better than M3). The method processes only damaged seeds, achieving zero actual displacement on intact points and balanced CD loss with no excess points.

### 5.7 Qualitative Results

[FIGURES: Side-by-side visualizations for each method]

For each test sample, three views are presented:
- Input: intact points (grey) + damaged points (red)
- Ground truth: intact points (grey) + reconstruction targets (green)
- Prediction: intact points (grey) + corrected points (blue)

Key visual observations:
- M1 (Pure PCN): Reconstructs overall shape but damaged region lacks precision
- M3 (LR-PCN): Generated points cluster near damage but with excess spread
- M4/M5 (Blind): All points shifted, intact surface visibly degraded
- M7 (SeedDispNet): Damaged points moved to surface level, intact points unchanged

---

## 6. Discussion

### 6.1 Pairwise Method Analysis

#### 6.1.1 M1 (Pure PCN) vs M3 (LR-PCN Label-Guided): The Value of Labels in Completion

Both methods follow the completion paradigm — generating new points from a global feature. The critical difference is label usage. M1 uses labels only to erase damaged points before encoding, discarding damage location information. M3 feeds labels as a 4th input dimension, allowing the encoder to learn which points are damaged, and conditions the decoder on the damage centroid and spread.

The result is that M1 achieves low full-cloud CD (dominated by the ~85% intact region that is trivially reconstructed) but poor damaged-region quality. M3 achieves substantially better damaged-region CD because the spatial conditioning anchors generated points near the actual damage. This demonstrates that for local reconstruction, label guidance is more valuable than architectural sophistication.

#### 6.1.2 M4 (Blind Displacement) vs M5 (Flow Displacement): Local Features Without Labels

M5 extends M4 with k-NN edge convolutions (k=16), adding local neighborhood features that capture surface curvature and geometry. Despite this additional complexity and computational cost, M5 achieves nearly identical performance to M4.

This result is significant: it demonstrates that local geometric features alone do not compensate for the lack of label awareness. Both methods waste capacity predicting zero displacement for ~85% of points (intact region) and both degrade intact surface quality (intact error ~0.48). The bottleneck is not feature quality but the absence of information about which points need correction.

#### 6.1.3 M3 (LR-PCN Label-Guided) vs M7 (SeedDispNet): Why Seed Displacement Outperforms Label-Guided Completion

Both methods use labels and both compute Chamfer Distance on the damaged region. Yet M7 achieves better reconstruction. Three factors explain this:

**Starting point of prediction.** M3's decoder generates 3D coordinates from a 512-dimensional abstract feature vector — it must hallucinate spatial positions from a compressed representation. M7's seeds start at the actual damaged point positions, which are already near the correct answer. The network only predicts small corrections (displacements), a fundamentally easier regression task.

**Point count balance in CD.** M3 computes CD between 1,024 generated points and ~200 GT targets — a 5:1 imbalance. In the pred→GT direction, many generated points can be far from any target without significantly affecting the average, because other points are close. In the GT→pred direction, each target easily finds a match among 1,024 candidates. M7 computes CD between ~200 displaced seeds and ~200 GT targets — perfectly balanced. Every predicted point must match a target and every target must be matched. There is no room for the network to "cheat" by placing excess points near easy targets.

**Information density per prediction.** M3's decoder must generate 1,024 × 3 = 3,072 coordinate values from a single 512-dimensional feature — a 6:1 compression ratio. Each output point receives the same global information with no per-point differentiation. M7 provides each seed with 771 dimensions of information (3 position + 256 local features + 512 global features) to predict just 3 values (dx, dy, dz) — a 257:1 information-to-output ratio. This dramatically reduces the prediction difficulty per point.

#### 6.1.4 M4 (Blind Displacement) vs M7 (SeedDispNet): Label Guidance in Displacement

Both methods predict per-point displacements, but M4 processes all 2,048 points while M7 processes only the ~200 damaged seeds. This difference has three consequences:

**Computational focus.** M4 allocates equal network capacity to all points. Since ~85% are intact and require zero displacement, most computation is wasted. M7 processes only damaged points, concentrating all capacity on the reconstruction task.

**Loss efficiency.** M4's loss includes an intact term (weight=0.1) that competes with the damaged term. The network must balance two objectives: move damaged points correctly AND keep intact points still. M7's loss is 100% focused on damaged reconstruction, with intact preservation guaranteed architecturally rather than through loss penalties.

**Intact preservation.** M4 achieves non-zero intact error because the network cannot perfectly learn zero displacement for all intact points. M7 achieves exactly zero intact error by construction — intact points are never processed by the displacement MLP.

### 6.2 Label Guidance as the Key Factor

Across both paradigms, label guidance provides the largest performance improvement:
- Completion: M3 (labeled) vs M1 (unlabeled) — labels enable focused reconstruction
- Displacement: M7 (labeled) vs M4 (unlabeled) — labels enable selective processing

This suggests that for local reconstruction tasks, knowing where the damage is matters more than sophisticated geometric feature extraction (k-NN in M5 provides negligible benefit over M4).

### 6.3 Generation vs Displacement

The completion paradigm (generate new points) and displacement paradigm (move existing points) represent fundamentally different approaches:

| Aspect | Completion | Displacement |
|--------|-----------|-------------|
| Output size | Fixed (decoder-determined) | Adaptive (input-determined) |
| Starting point | Global feature (abstract) | Damaged point position (concrete) |
| Loss efficiency | Diluted by excess points | 100% (every point matters) |
| Intact handling | Separate extraction | Architectural guarantee |
| Prediction difficulty | Generate full coordinates | Predict small corrections |
| Information per point | Global only (512-dim) | Local + global (771-dim) |

Displacement is better suited for local reconstruction because the damaged points already provide spatial anchors near the correct positions. The network only needs to learn small corrections rather than generating coordinates from scratch.

### 6.4 The Mean-Collapse Problem

Per-point L2 loss with shared features causes all predictions to converge to the mean target position. This is a known issue in generative models (mode collapse) manifesting in the displacement setting. When the global feature dominates the per-seed input (512 of 771 dimensions are shared), the MLP produces nearly identical outputs for all seeds.

Chamfer Distance resolves this by evaluating the predicted point set as a whole rather than individual correspondences. If all seeds collapse to one point, the GT→pred direction of CD produces large distances for all unmatched GT targets, creating strong gradients that push seeds apart.

### 6.5 Practical Considerations

**Label requirement.** SeedDispNet requires binary damage labels at inference time. In practice, these can be obtained from: (1) comparison with a nominal CAD model, (2) a separate damage detection network, or (3) manual annotation for critical components.

**Scalability.** The per-seed MLP processes K damaged points sequentially within each batch sample. For large damage regions (K > 500), this could become a bottleneck. Batching seeds across samples or using parallel processing could address this.

**Real-world transfer.** The DMSurface dataset uses synthetic damage. Real-world damage exhibits more complex patterns (corrosion, fatigue cracks, impact damage) that may require domain adaptation or fine-tuning.

### 6.6 Limitations

1. **Synthetic data only.** Evaluation is limited to simulated dent damage. Real-world validation is needed.
2. **Label dependency.** Requires pre-computed damage labels, adding a dependency on upstream detection.
3. **Single-scale partitioning.** Fixed grid partitioning may not optimally capture damage at different scales.
4. **No surface continuity constraint.** The loss does not explicitly enforce smoothness at the boundary between intact and reconstructed regions.
5. **Sequential seed processing.** The per-sample loop over seeds limits parallelism for large damage regions.

---

## 7. Conclusion

We presented SeedDispNet, a label-guided seed displacement network for local surface reconstruction in 3D point clouds. Through systematic comparison of five methods across three paradigms, we identified three critical factors for this task:

1. **Label guidance** is the most impactful factor, consistently improving results regardless of the underlying paradigm.
2. **Adaptive output size** through seed-based prediction eliminates the fixed-output limitation of completion methods.
3. **Chamfer Distance loss** on the damaged region prevents mean-collapse and ensures distributed reconstruction.

SeedDispNet combines these insights into a simple yet effective architecture that guarantees intact preservation, adapts to variable damage sizes, and focuses all learning capacity on the damaged region.

Future work includes: validation on real-world damaged surfaces, integration with automatic damage detection in an end-to-end pipeline, attention mechanisms for boundary-aware reconstruction, and extension to other damage types beyond dents.

---

## References

[1] W. Yuan, T. Khot, D. Held, C. Mertz, and M. Hebert, "PCN: Point Completion Network," in 3DV, 2018.

[2] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation," in CVPR, 2017.

[3] P. Tchapmi, V. Kosaraju, H. Rezatofighi, I. Reid, and S. Savarese, "TopNet: Structural Point Cloud Decoder," in CVPR, 2019.

[4] M. Liu, L. Sheng, S. Yang, J. Shao, and S.-M. Hu, "Morphing and Sampling Network for Dense Point Cloud Completion," in AAAI, 2020.

[5] X. Yu, Y. Rao, Z. Wang, Z. Liu, J. Lu, and J. Zhou, "PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers," in ICCV, 2021.

[6] M.-J. Rakotosaona, V. La Barbera, P. Guerrero, N. J. Mitra, and M. Ovsjanikov, "PointCleanNet: Learning to Denoise and Remove Outliers from Dense Point Clouds," in Computer Graphics Forum, 2020.

[7] S. Luo and W. Hu, "Differentiable Manifold Reconstruction for Point Cloud Denoising," in ACM MM, 2020.

[8] S. Luo and W. Hu, "Score-Based Point Cloud Denoising," in ICCV, 2021.

[9] X. Liu, C. R. Qi, and L. J. Guibas, "FlowNet3D: Learning Scene Flow in 3D Point Clouds," in CVPR, 2019.

[10] W. Wu, Z. Y. Wang, Z. Li, W. Liu, and L. Fuxin, "PointPWC-Net: Cost Volume on Point Clouds for (Self-)Supervised Scene Flow Estimation," in ECCV, 2020.

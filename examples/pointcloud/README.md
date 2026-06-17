# PointCloud — view-invariant SSL on 3D point clouds (ModelNet40)

**Question.** Can a two-view SSL objective learn a **view-invariant** shape
representation on an *unordered, irregular* modality (3D point clouds), and how
does linear-probe accuracy degrade as we demand more **rotation invariance**
(none → z → SO(3))?

Point clouds have no temporal frames, so the objective is **two-view VICReg** (the
image-JEPA / audio / EEG recipe), *not* a predictive JEPA. The two SSL views are
two independent augmented samplings + rotations of the *same* object, so VICReg
learns a representation invariant to how the object was sampled and oriented.

## Data
**ModelNet40** in the canonical PointNet HDF5 release (`modelnet40_ply_hdf5_2048`):
9840 train / 2468 test shapes, **2048 (x, y, z) points** each, **40 classes**, at
`/lustre/work/pdl17890/udl806719/datasets/modelnet40/modelnet40_ply_hdf5_2048`.
Each shape → two augmented views, each `[3, n_points=1024]`: random **subsample**,
random **rotation** (`rotate = so3 | z | none`), random **scale** (0.8–1.25),
Gaussian **jitter** (σ=0.01), then **unit-sphere normalize** (center + scale).

## Layout
```
eb_jepa/datasets/pointcloud/   dataset.py (provided loader) + data_config.yaml
examples/pointcloud/
  main.py     SSL pretraining — TODO: build_encoder() + build_ssl()
  eval.py     downstream probe — TODO: probe() + metric
  configs/    train.yaml, eval.yaml
```

## What you implement (the `# TODO`s)
1. `main.py:build_encoder` — a **PointNet** encoder over `[B, 3, N]`: a shared
   per-point MLP of 1×1 `Conv1d`s (`3→64→64→128→out_dim`) + a symmetric **max-pool**
   → permutation-invariant global feature. Expose `.represent()` and `.out_dim`.
2. `main.py:build_ssl` — the **two-view VICReg** objective: two views →
   `encoder.represent` → eb_jepa `Projector` → eb_jepa `VICRegLoss` (invariance +
   variance + covariance). The invariance term makes the feature *view-invariant*;
   the var/cov terms prevent collapse.
3. `eval.py:probe` — the frozen-feature linear probe → **40-way accuracy** on the
   official ModelNet40 test split, compared to a random-encoder floor (chance 2.5%).

Everything else (data loading, augmentation, training loop, feature extraction) is
provided. Reuse the eb_jepa core (`Projector`, `VICRegLoss`) — do not duplicate.

## Why this track
The max-pool gives the encoder **permutation invariance** for free (point order is
meaningless). **Rotation invariance**, by contrast, is not built in — it must be
*learned* from the augmented views. The expected (well-known) result is that
accuracy drops monotonically `none → z → SO(3)`: the more rotation invariance the
two views demand, the harder the global feature is to keep linearly separable.

## Run
```bash
python -m examples.pointcloud.main --fname examples/pointcloud/cfgs/train.yaml
python -m examples.pointcloud.eval --ckpt <.../latest.pth.tar>
# view-invariance study: rerun pretraining with data.rotate=none and data.rotate=z
```

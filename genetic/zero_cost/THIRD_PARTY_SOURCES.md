# Method and implementation sources

The code in this folder is adapted to the project's common proxy interface and
was rewritten to avoid external runtime dependencies. The mathematical and
implementation references used are listed below.

## Apache-2.0 reference implementations

- Zero-Cost Proxies for Lightweight NAS:
  https://github.com/mohsaied/zero-cost-nas
  - SynFlow, SNIP, GradNorm, Plain, L2Norm, Fisher, GraSP, JacobCov.
- ZiCo:
  https://github.com/SLDGroup/ZiCo
  - ZiCo gradient aggregation.

## Other public reference implementations

- Zen-NAS:
  https://github.com/idstcv/ZenNAS
  - Zen-Score computation. The project model does not expose
    `forward_pre_GAP`, so this package captures the last Conv2d feature map.
- NAS Without Training:
  https://github.com/BayesWatch/nas-without-training
  - NASWOT binary activation kernel and log-determinant score.
- SWAP-NAS (AFL-3.0 repository):
  https://github.com/pym1024/SWAP
  - Sample-wise activation pattern definition. This package uses an independent
    memory-safe pattern-hashing implementation.
- MeCo (MIT repository):
  https://github.com/HamsterMimi/MeCo
  - Minimum eigenvalue of feature-map correlation matrices.

## Project-local baseline

- `macs` uses standard Conv2d and Linear forward hooks to count
  multiply-accumulate operations per input sample. It is a complexity baseline,
  not claimed as a new zero-cost method.

These implementations are for passive experimental comparison. When reporting
results, cite the original papers associated with each method.

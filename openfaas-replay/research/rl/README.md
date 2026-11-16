# R366 companion research source

These are source snapshots taken from the experiment directories used to prepare the R366 eight-function OpenFaaS replay. They are included to make the runner's code dependency chain local and reviewable; they are not model or data releases.

## Runtime import chain

```text
R366 live controller
├── R329 trace replay driver
│   └── R328 eight-function mechanism
│       └── R187 simulator and shared paper code under easr-simulation/
├── R361 training helpers
│   ├── R328 mechanism
│   ├── R329 trace replay driver
│   └── R336 GRL adapter and target helpers
└── OpenFaaS eight-function deployment helper
```

The R328, R329, R361, and R336 files were copied from their matching `Other/research/...` experiment directories. In R328 and R366, only source-root and private-artifact path wiring was changed for this package. The R361 protocol copy removes the machine-specific run location. The source SHA-256 values and original experiment paths are recorded in the repository-level `SOURCE_MANIFEST.json`.

## What remains private or excluded

- Azure trace CSVs, generated request windows, synthetic profile manifests, run outputs, and all checkpoints/model weights are excluded. Recreate inputs using the documented generators where applicable.
- R366 reads the frozen R361 PPO/GRL checkpoints and target manifest, plus the frozen R363 RF snapshots and held-out audit from `R366_PRIVATE_ARTIFACT_ROOT`.
- The R363 calibration-training script is not included because it contains the RF threshold calibration sweep, which is part of the sensitivity material currently withheld. R366 does not import that script; it consumes the frozen R363 outputs.
- R336's target-preparation source is present to complete the R361 helper-source lineage, but its private weights and generated targets are not included.

This source package therefore makes the runtime code path inspectable but does not reproduce the trained decisions without the author's private artifacts. See the parent [`README.md`](../../README.md) and [`PROTOCOL.md`](../../PROTOCOL.md) for setup and experimental limits.

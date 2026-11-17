# Serverless function fusion research

This repository contains research source code for the serverless function-fusion paper.

**For viewing and reference only. All rights reserved.** See [LICENSE](LICENSE).

## Included

- [`easr-simulation/`](easr-simulation/README.md): source snapshots for the two-resource-pool simulation pipeline associated with R218.
- [`openfaas-replay/`](openfaas-replay/README.md): the R366 OpenFaaS replay controller, its R328/R329/R361 companion source, R336 GRL adapter helpers, and the shared scale-8 deployment helper. Frozen model and audit artifacts remain private.

## Excluded for now

- Parameter-sensitivity sweeps and their configurations/results.
- Experiment outputs, frozen profile/trace copies, model checkpoints and trained weights. Model weights are author-trained research artifacts and are intentionally private; the repository may expose the training code and protocol without exposing those weights.
- Server logs, server addresses, credentials, and local environment files.

Synthetic inputs are generated from included scripts where possible, and external traces are cited with their preprocessing recipe instead of copied wholesale. The shared four-function simulator has been extracted to `workflow_core.py`; the R116 similarity/threshold probe and its grid are excluded. Protocol files required by the staged simulation drivers are included. R187 profiles and R366's synthetic workflow and Azure-derived request windows have generation recipes, while exact R218/R366 frozen-checkpoint results still depend on intentionally private historical artifacts. See [DATA_AND_MODEL_POLICY.md](DATA_AND_MODEL_POLICY.md) for that boundary.

## Attribution and reproducibility

The GRL baseline is based on [*Structure-Aware Serverless Orchestration via Graph Reinforcement Learning: Joint Optimization of Function Fusion and Placement*](https://ieeexplore.ieee.org/abstract/document/11657977). Our R181 implementation of that method is the code carried into R188; it is not code released by the paper's authors or a full reproduction of their original experiments. The simulation and replay sources are not turnkey reproductions of results obtained with private checkpoints and historical manifests. These boundaries are recorded in [PUBLIC_SOURCE_AUDIT.md](PUBLIC_SOURCE_AUDIT.md) and [DATA_AND_MODEL_POLICY.md](DATA_AND_MODEL_POLICY.md).


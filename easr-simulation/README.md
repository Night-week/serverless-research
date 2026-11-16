# EASR simulation source snapshot

This directory contains the two-resource-pool simulation source lineage and its experiment protocols. Generated profiles, outputs, model weights, checkpoints, and parameter-sensitivity probes are excluded. The trained weights are author-produced research artifacts and are intentionally not part of the source package. Readers can generate fresh R187 profiles and train new models; this will not reproduce exact results obtained with private frozen checkpoints or historical manifests.

## Source map

- `research/rl/r187_twenty_function_scale/`: 20-function workflow model, event engine, PPO/RF training and evaluation source. `generate_profiles.py` creates its deterministic synthetic profile manifest without starting training.
- `research/rl/r188_twenty_function_grl/`: GRL-Orch baseline based on [*Structure-Aware Serverless Orchestration via Graph Reinforcement Learning: Joint Optimization of Function Fusion and Placement*](https://ieeexplore.ieee.org/abstract/document/11657977). R188 carries forward our R181 same-environment implementation; it is not the authors' source or a full reproduction of their original experiments.
- `research/rl/overnight_scale_20261028/suite.py`: shared simulation and baseline helpers.
- `research/rl/r190_twenty_function_confirm500/` and `r194_configuration_coverage/`: confirmation and configuration-coverage drivers.
- `research/rl/r199_live_snapshot_review/`, `r209_grl_response_objective/`, and `r211_remaining_review_queue/`: frozen-policy evaluation, response-objective adapter, and rolling-transition source.
- `research/rl/r218_final_frozen_confirmation/`: final confirmation driver and baseline comparison/summarization scripts.
- `research/rl/r211_remaining_review_queue/rolling_transition.py`: the rolling-transition execution semantics used by R218. The separate response-delay sensitivity runner is not included.
- `CloudEdge_Function_Reconfiguration_Paper_20261026/code/`: shared policy primitives and `workflow_core.py`, the reusable four-function simulator extracted from the former R116 module. The R116 similarity/threshold probe and its grid are excluded.

R187 profiles can be regenerated from source with `python generate_profiles.py --output-dir outputs` in its directory. This creates a new deterministic profile manifest and bounded historical plans; it does not recreate private R190/R194/R215 manifests or frozen model files. The R218 driver is an evaluator for those frozen artifacts and cannot reproduce its exact published decisions without them. Do not describe this source-only package as an exact or turnkey reproduction of R218. See the staged protocol files and repository-level [`DATA_AND_MODEL_POLICY.md`](../DATA_AND_MODEL_POLICY.md) for the reproduction boundary.

Python package versions used by the internal source set are listed in [`requirements.txt`](requirements.txt). No installation or experiment was run while staging these files.


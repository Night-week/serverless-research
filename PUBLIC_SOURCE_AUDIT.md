# Public-source staging audit

## Scope

This document records the source staging audit. It includes two separate tracks: the two-resource-pool simulation source lineage and the eight-function OpenFaaS replay harness.

## Release-preparation changes

- Extracted the reusable four-function simulator primitives from the old R116 file into `easr-simulation/CloudEdge_Function_Reconfiguration_Paper_20261026/code/workflow_core.py`. The R116 similarity probe and threshold grid are not included.
- Updated R151/R187 and scale-suite imports and source bindings to use `workflow_core.py`.
- Extracted only the R211 rolling-transition semantics needed by R218 into `r211_remaining_review_queue/rolling_transition.py`; the response-delay sensitivity runner and its delay sweep are excluded.
- Added the R151/R152/R153, R187/R188, R190/R194, R209, and R218 protocol documents required to explain the staged source. Removed the private execution host and absolute server path from the overnight protocol copy.
- Added `r187_twenty_function_scale/generate_profiles.py` for generating a fresh deterministic profile manifest without starting model training.
- Added the R328/R329/R361 companion source snapshots and R336 GRL adapter helpers from their matching experiment directories; updated R328/R366 source and private-artifact paths for this repository layout. R366 now reads frozen checkpoints, target manifests, RF snapshots, and held-out audit through `R366_PRIVATE_ARTIFACT_ROOT`. Replaced the private image reference with `OPENFAAS_FUNCTION_IMAGE` and the machine-specific node label with a generic label.
- Renamed the staged configuration-manifest path to `full_config_manifest.json` across its source consumers to use neutral terminology; the control and scoring logic is unchanged.
- Added the R366 Azure sequence and synthetic workflow generators. The raw Azure CSV, generated streams, profiles, outputs, and trained weights remain excluded.

## Data, models, and reproducibility limits

- No model weights, raw trace copies, generated experiment outputs, logs, or secret values are staged. OpenFaaS obtains its Basic Auth password from the cluster secret at runtime.
- R187 can generate fresh synthetic configurations and train new models. It does not recreate private historical R190/R194/R215 manifests or the frozen model decisions used by R218.
- R366 can recreate the Azure-derived request windows and the synthetic eight-function fixture. Its runtime companion source is included. Exact replay still requires the matching private R361/R363 checkpoints, target manifests, RF snapshots, and held-out audit under `R366_PRIVATE_ARTIFACT_ROOT`. The R363 calibration-training script and threshold sweep are excluded with sensitivity materials. The controller checks frozen artifact identities and does not silently substitute another model.
- The Azure Functions 2019 trace is cited with its preprocessing recipe; it is not redistributed. The R366 use of the trace is an ingress-rate proxy for a synthetic eight-function diamond, not an Azure workflow DAG.
- The GRL-Orch adapter is our local implementation based on the [Structure-Aware paper](https://ieeexplore.ieee.org/abstract/document/11657977), not the paper authors' source code or a reproduction of their original OpenWhisk experiments.

## Static audit snapshot (2026-11-16)

- The inventory lists 59 files (35 Python source files); every listed file exists and its SHA-256 matches. The inventory excludes itself.
- The R366 runtime import chain resolves to the staged R328/R329/R361/R336 source plus R187 and shared paper code in this tree.
- No model/checkpoint, raw CSV/archive, generated trace, or result artifact is present. No literal credential value or private server hostname/path was found in the release scan; the OpenFaaS helper reads its password from the cluster secret at runtime.
- Completed a bilingual terminology scan and context review of all staged files; no flagged content remained. The source inventory and release text use neutral labels.
- No experiment, generator, or test suite was run during this packaging pass. The two profile generators therefore remain unexecuted and should be checked before the package is described as validated.

## Release terms and follow-up

1. See [LICENSE](LICENSE) for the viewing and reference terms. Third-party materials retain their respective terms.
2. Run and review the two R366 input-generation scripts before claiming the generated files reproduce the frozen historical inputs. The Azure trace-preparation script is adapted from R332; the synthetic profile generator is a new deterministic release helper, not copied from a historical R366 generator.
3. The R218 and R366 source bundles are not turnkey reproductions of results obtained with private checkpoints and historical manifests. Preserve that limitation in the paper/repository description.

The file/hash inventory is maintained in `SOURCE_MANIFEST.json`. It intentionally excludes itself because a self-hash would change the file recursively.

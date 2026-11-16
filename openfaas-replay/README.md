# OpenFaaS replay source snapshot

This directory contains the R366 fixed-controller, cross-window replay runner, its R328/R329/R361 companion source snapshots, and the shared 8-function OpenFaaS helper. The companion code was copied from the corresponding experiment directories; only release-specific path wiring and host-detail cleanup were applied. No run results, Azure request-window data, workflow profiles, checkpoints, or model weights are bundled. The trained weights are author-produced research artifacts and are intentionally private. The replay runner does not train the frozen R361/R363 models, so exact frozen-controller decisions require the private artifacts described below.

The helper does not contain a registry-specific image reference. Set `OPENFAAS_FUNCTION_IMAGE` to an immutable image reference before deployment. Runtime source dependencies are included in this repository: R366 imports the local R329 and R361 drivers, R329 loads the local R328 mechanism, and these use the R187 simulator under `easr-simulation/`. R361's GRL adapter helpers are included under the R336 source snapshot. R366's frozen checkpoints, target manifests, RF snapshots, and held-out audit are private artifacts; place them in the directory layout shown below and set `R366_PRIVATE_ARTIFACT_ROOT`. Missing artifacts cause a clear error rather than a controller substitution. The R363 calibration-training script and its threshold sweep are intentionally omitted with the sensitivity material; the replay consumes only the already-frozen R363 outputs.

The replay needs an authorized K3s/OpenFaaS environment and runtime access to its Basic Auth secret. It was measured on one K3s node with cached images; Edge and Cloud are logical labels there. This source does not model cold image transfer or cross-node networking. The Azure-derived request windows should be cited and recreated from their upstream source using the extraction recipe below; this repository does not redistribute the trace. See the repository-level [`DATA_AND_MODEL_POLICY.md`](../DATA_AND_MODEL_POLICY.md).

Install the shared Python dependencies from the repository root with `python -m pip install -r easr-simulation/requirements.txt`. The host must also provide `kubectl`, a configured K3s context, and an OpenFaaS endpoint reachable by the replay client.

The helper obtains the OpenFaaS Basic Auth password from the cluster secret at runtime. No password value is stored here. Review cluster access and image settings before use.

The R366 Azure-derived request windows can be regenerated without committing the data. Download the official [Azure Functions 2019 archive](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz), extract `invocations_per_function_md.anon.d01.csv`, then run:

```sh
python tools/prepare_azure_sequences.py \
  --input /path/to/invocations_per_function_md.anon.d01.csv \
  --output research/real_platform/openfaas_scene1_rl_live_20261105/r366_same_model_cross_window_replay_20261114/inputs/azure_sequence_streams.json
```

Regenerate the synthetic eight-function workflow profile and initial layouts with:

```sh
python tools/prepare_r366_profile.py \
  --output research/real_platform/openfaas_scene1_rl_live_20261105/r366_same_model_cross_window_replay_20261114/inputs/profile_manifest.json
```

Set `R366_PROFILE_MANIFEST` to that file when running the controller. The frozen R361/R363 models and target manifests remain private dependencies, so this does not recreate their trained decisions.

Before starting a replay, configure the private frozen-artifact tree and deployment image. Its required layout is:

```text
private-dependencies/
└── rl/
    ├── r361_scene1_8fn_live_models_20261112/outputs/
    │   ├── manifest.json
    │   ├── live_targets.json
    │   ├── seed_631/selected.pt
    │   ├── seed_631/rf_live_snapshot.joblib
    │   ├── grl_native/seed_631/selected.pt
    │   └── grl_response/seed_631/selected.pt
    └── r363_scene1_8fn_rf_real_ready_calibration_20261113/outputs/
        ├── heldout_mid_cv_p50.json
        └── seed_631/rf_live_snapshot.joblib
```

The expected model hashes are locked in the R366 runner. Use only the matching frozen artifacts for a seed. Example environment setup:

```sh
export R366_PRIVATE_ARTIFACT_ROOT=/path/to/private-dependencies
export R366_PROFILE_MANIFEST=/path/to/profile_manifest.json
export OPENFAAS_FUNCTION_IMAGE=registry.example/function-image:immutable-tag
export R366_SEED=631
export R366_STREAM_ID=mid_cv_p50_q25_minute0024
```

Use `R366_METHOD` to select one of the methods supported by the runner. The included runner is bound to the R361/R363 held-out target and checkpoint hashes; a different model requires a newly generated target/audit bundle and a corresponding source change. This boundary is intentional because the authors' trained weights are not being released.

The included research source snapshots and their dependency map are listed in [`research/rl/README.md`](research/rl/README.md). The upstream dataset is released under CC BY. Cite Shahrad et al., “Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider,” USENIX ATC 2020; see the official [Azure Public Dataset project](https://github.com/Azure/AzurePublicDataset) and [dataset archive](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz).

The 8-function diamond is a synthetic workflow fixture (profile `326001`), not a DAG from Azure. Its profile manifest and initial layouts can be regenerated with `tools/prepare_r366_profile.py`. The runner still requires the private R361/R363 checkpoints and target manifests, so this source does not reproduce the authors' frozen decisions by itself. Images were cached on one K3s node, and Edge/Cloud are logical labels there; the replay does not measure cold image transfer or cross-node networking. See [`PROTOCOL.md`](PROTOCOL.md) for the protocol and limits.


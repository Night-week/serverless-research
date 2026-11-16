# R366 OpenFaaS replay protocol

## Workload and request streams

- Workflow: a synthetic eight-function diamond, profile `326001`; the Azure trace does not contain workflow DAGs.
- Arrival source: Azure Functions 2019, file `invocations_per_function_md.anon.d01.csv`, which provides per-function invocation counts per minute.
- The `mid_cv_p50` function sequence provides four minute-count windows. R366 uses `mid_cv_p50_q25_minute0024` (39 requests) and `mid_cv_p50_q75_minute0294` (57 requests). Within-minute arrival times are generated with deterministic uniform jitter; they are a replay proxy, not original per-request timestamps.
- The four candidate windows were compared using only the first 15 seconds of arrival data. The q75 window was selected for stronger pre-decision burst indicators, without using replay outcomes.
- Regenerate request windows with `tools/prepare_azure_sequences.py`; download and cite the upstream dataset rather than checking in the CSV or generated JSON.

## Controller and execution

- Frozen controller: EASR seed 631 PPO plus calibrated RF; model weights remain private and are not included. The six runs use the same controller and alternate q25/q75 three times each.
- Decision time: 15 seconds after replay start. The compared paths are the controller's observed keep and Whole32 decisions for these two windows.
- Platform: single-node K3s/OpenFaaS CE. Edge and Cloud are logical labels on the same host. The image is cached; the experiment does not measure cold image transfer or cross-node communication.
- The replay records request response, transition pause, queueing/drain behavior, Ready time, and checksums. The client-side queue estimates are not Kubernetes internal queue telemetry.

## Reproduction limits

The R366 profile manifest and initial layouts can be regenerated with `tools/prepare_r366_profile.py`, and request windows can be regenerated with `tools/prepare_azure_sequences.py`. Runtime source dependencies R328/R329/R361 and the R336 GRL adapter helpers are included under `research/`; the import chain is documented in `research/rl/README.md`. The controller still depends on separately frozen R361/R363 model and target artifacts, which are not part of this source release. The R363 calibration-training script and its threshold sweep are omitted with the sensitivity materials. This is not a turnkey reproduction of the authors' frozen decisions. The Azure function series is used as an ingress proxy for a synthetic workflow, not as a production workflow trace.

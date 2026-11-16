# R329: Azure production-trace-driven request replay

Date frozen: 2026-11-05  
Status: protocol frozen before evaluating any R328 model on these inputs.  
Execution host: authorized OpenFaaS experiment server; no model training runs locally.

## Question

Does the completed R328 Scenario 1 system retain its response-time benefit when the 8-function workflow is driven by request-arrival patterns derived from a production FaaS trace, rather than R328's piecewise-Poisson stream?

## Input trace and semantics

- Source: the public Azure Functions 2019 `invocations_per_function_md.anon.d01.csv` file already present in the research workspace. It has one invocation count per function per minute for one day.
- The file does not expose per-request timestamps or DAG membership. Each selected Azure function's minute-count series is therefore used as a **workflow-ingress proxy** for the existing synthetic 8-function DAG profiles. This is trace-driven workload replay, not an actual Azure workflow trace.
- Function selection is outcome-blind: among the top 1% of non-empty functions by daily invocation count, retain functions active in at least 90% of minutes; choose the minimum-CV (steady), median-CV (variable), and maximum-CV (bursty) series.
- Four minute windows per selected series are selected at the 25th, 50th, 75th, and 95th percentiles of minute counts. Ties are resolved by the earliest unused minute index.
- Each series is downscaled by the smallest integer divisor that makes its maximum minute count at most 480 requests/minute (8 requests/second). Counts are rounded to the nearest integer. This keeps the replay inside the measured testbed range while preserving the trace's relative rate shape.
- The released trace has minute counts only. The retained count is distributed uniformly at random within its minute. The immediately preceding minute is treated as the 60-second history. This interpolation and scaling are explicitly recorded in every stream manifest.
- Every ingress starts the same 8-function DAG in the corresponding held-out R326 profile. DAG shape, function work, memory, links, and resource budgets remain those of R328's nine held-out OpenFaaS profiles.

## Evaluation

- Reuse the 100 completed R328 policy/RF pairs (seeds 601–700); do not retrain or tune thresholds/features on the Azure trace.
- Replay the same 12 generated minute streams for every policy, initial layout, and profile.
- Compare R328's full system, system without the pressure override, local PPO search, `whole32`, and keep-layout baseline.
- Preserve R328's simulator semantics for admission, node/link contention, draining, deployment delay, response time, and resource legality.
- Report mean/P95 response, deployment and drain time, configuration changes, route rates, request completion, and memory legality. Cluster paired intervals by the 12 trace-window IDs; the three underlying function series are additionally called out as a small trace sample.
- These are simulator results under trace-driven ingress. They do not claim that a production DAG or exact per-request production timestamps were replayed.

## OpenFaaS validation

After simulation, replay the same trace windows at the same downscaled rates through the existing 8-function fused/split OpenFaaS harness on the single-node K3s host. Use one predeclared diamond profile to keep the live run bounded. Measure invocation latency, errors, dispatch lag, service readiness, and teardown separately. Existing pilot functions stay untouched. The function image is already present on the node, so deployment timing is a warm-image result; no image-cache eviction or cold-pull claim is made. This validates the live invocation and deployment path, not deployment of the RF/PPO controller.


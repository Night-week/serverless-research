# R328: reduced Scenario 1 mechanism validation at eight functions

## Question

Does the complete Scenario 1 controller still make useful decisions after reducing the function count and the two logical pool budgets to the scale used by the OpenFaaS validation?

## Matched setup

- Eight functions across two logical pools: edge-like 2 cores / 512 MiB and cloud-like 6 cores / 4096 MiB.
- Chain, fork/join diamond, and binary-tree DAGs; 108 generated profiles split into 48 training, 12 validation, and 48 held-out confirmation profiles.
- Four reproducibly sampled legal initial layouts per profile, plus legal all-edge or all-cloud layouts when those are distinct and feasible.
- A held-out replay adds the exact nine eight-function profiles previously invoked on OpenFaaS. Those profiles are projected from R326's 20-function DAGs; their original speed, bandwidth, and RTT settings are restored from the source manifest, and their resource budgets are reduced to 2/6 cores and 512/4096 MiB. They are never used for PPO or RF fitting.
- Three PPO seeds (601, 631, 659), each trained from scratch for 32,768 charged calls. The 20-function checkpoint is not reused because its state/action coordinates do not match eight functions.
- PPO observes the current state and may make at most two legal local actions, with action 0 as STOP. The second action is evaluated after the simulated 15-second transition interval.
- RF uses only training profiles and the R326/R199 live-snapshot feature definition. Its label is positive when the best-of-32 historical whole-layout plan improves mean response by at least 3% over the local PPO result on the same request stream.
- At evaluation time, the router selects the whole plan when RF predicts positive. The existing pressure guard also selects the whole plan when the PPO-selected candidate's maximum normalized CPU/WAN pressure exceeds 1.0.
- The whole-layout target is the best-scoring legal layout in a 32-candidate pool on a separate historical request stream. It receives no confirmation stream or future capacity information.

## Evaluation

Every method starts from the same profile, initial layout, decision time (15 s), and request stream. We compare the full RF + pressure-guard + bounded-PPO system with RF-only routing, local PPO, the historical whole32 target, and keeping the current layout. The simulator charges transition deployment and drain costs in response results. Candidate generation/scoring and controller inference wall time are reported separately so they are not hidden inside the response metric.

Primary results use the 48 generated confirmation profiles; the nine OpenFaaS-profile replays are a separate transfer check. Paired uncertainty resamples profiles as clusters. Results from the two datasets are never pooled.

## Boundary

The nine OpenFaaS profiles are replayed through the simulator after their real-platform workload definitions are loaded. This does not run a closed-loop RF/PPO controller on OpenFaaS. The existing real-platform run verifies service deployment and DAG invocation only; it used one physical K3s node, logical pool budgets, and a benchmark image that does not forward computed outputs. This experiment also does not measure WAN transfer or image-pull costs. Any positive result supports the mechanism at the reduced simulation scale, not a claim of live multi-pool deployment performance.

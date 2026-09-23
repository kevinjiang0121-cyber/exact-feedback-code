# ICLR Strong-Findings Spotlight Gate

Date: 2026-07-31  
Status: frozen before linked analysis

## Scientific question

Can a checkpoint's response policy, identified only from controlled discovery
states, predict its behavior in an independent long-horizon exact-feedback loop?

The test is deliberately stronger than correlating seven model-level summary
rows. It applies each frozen discovery curve to case-disjoint Combined-480
transitions and evaluates prediction at the case/trajectory level.

## Frozen linked panel

The primary panel contains checkpoints with both a discovery response curve and
an independently generated Combined-480 asset under the closest available
common-plain protocol:

- `llama31_8b_base`
- `llama31_8b_instruct`
- `qwen3_1_7b`
- `qwen3_4b`
- `qwen3_8b`
- `qwen3_14b`
- `gemma2_9b_it`

`qwen3_8b_base` remains part of the aligned Base/Instruct system-identification
contrast, but is excluded from this linked test because it has no frozen
Combined-480 endpoint.

All 72 system-identification case IDs are excluded from every closed-loop asset.
Only nonzero transitions whose required action lies within the discovery support
`[-50, 50]` words enter the primary transition analysis.

## Frozen predictor

For closed-loop error `e_t = observed_length - target_length`, the required
linguistic action is `c_t = -e_t`. For each checkpoint, linearly interpolate its
frozen discovery median-action curve at `c_t`, adding the physically required
origin `(0, 0)`. No confirmation value or closed-loop outcome is used to fit the
curve.

The predicted next error is

`e_hat_(t+1) = e_t + a_hat_checkpoint(c_t)`.

Comparators are:

1. the frozen pooled discovery curve;
2. zero action;
3. ideal identity action `a(c)=c`;
4. checkpoint-level parameter count, family, and one-shot performance for
   explicitly exploratory aggregate comparisons.

## Primary estimands

1. Closed-loop action MAE for checkpoint-specific versus pooled curves.
2. Relative MAE reduction `(MAE_pooled - MAE_checkpoint) / MAE_pooled`.
3. Prediction of realized transition category: zero action, correct direction,
   contraction, and overshoot.
4. Prediction of case-level closed-loop regime: eventual exact capture,
   terminal joint success, stationary failure, and oscillation.

Uncertainty is clustered by case ID and stratified by source family. The same
resampled case IDs are applied across checkpoints whenever available.

## Decision levels

### Strong spotlight gate

All must hold:

1. aggregate checkpoint-specific action MAE is below pooled MAE with a 95%
   case-clustered interval excluding zero;
2. the lower confidence bound of aggregate relative MAE reduction exceeds 5%;
3. checkpoint-specific point estimates improve over pooled in at least five of
   seven linked checkpoints;
4. discovery-derived simulated regimes achieve Spearman `rho >= 0.70` against
   observed checkpoint rankings for at least two of three outcomes: terminal
   joint success, oscillation, and stationary failure;
5. the result is not driven solely by either Llama checkpoint.

Passing authorizes the main claim that a cheap local policy assay predicts
expensive long-horizon behavior and supports a spotlight-oriented manuscript.

### Poster-plus gate

If the aggregate MAE interval excludes zero and at least four of seven
checkpoints improve, but one or more materiality/regime conditions fail, claim
only that local policies transfer predictively to closed-loop transitions. Keep
the stronger regime law descriptive.

### No linked uplift

If the aggregate interval includes zero or fewer than four checkpoints improve,
do not claim local-to-closed-loop prediction. Retain the already confirmed
identify--contrast--boundary paper and report this linked result as a limitation
or appendix negative.

## Oral gate

Oral-level positioning is not implied by the spotlight gate. It requires a
compact continuous-state dynamical result that correctly predicts multiple
closed-loop regimes on held-out checkpoints or a separately authorized,
prospective cross-assay/intervention confirmation. Standard control terminology
or additional model rows do not satisfy this condition.

## Output contract

The analysis must write:

- a machine-readable JSON report with input hashes;
- per-checkpoint transition and regime tables;
- case-clustered bootstrap intervals;
- a concise acceptance document stating which claim level is authorized;
- no manuscript changes until the verdict is known.


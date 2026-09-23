# ICLR Unified Output-Recurrence and Basin-Escape Audit V2

Date: 2026-07-31  
Status: frozen after V1 state-definition audit and before V2 results

## Why V1 is invalid

The deployed loop appends every prior assistant response and verifier message
to the chat history.  Therefore equal `(text, error, missing anchors)` at two
rounds does not imply equal controller input.  V1 incorrectly called this a
repeated full state and attached a deterministic-cycle implication.  The
contradiction was directly observable: 211 trajectories succeeded after a V1
"failed attractor."

V1 outputs and its exact script snapshot remain preserved for audit but are
not paper evidence:

- `experiments/unified_attractor_audit_v1/`
- `scripts/analyze_unified_attractor_audit_v1_invalid.py`

## Corrected object of study

Define the observable **output state**

`o_t = (SHA256(text_t), error_t, sorted(missing_anchors_t))`.

Repeated `o_t` under a growing conversation history is an output recurrence,
not a repeated Markov state.  It shows that additional exact feedback and
additional context can return the system to a previous draft, but it does not
mathematically guarantee future periodicity.

The primary population, integrity checks, twelve checkpoints, basin/escape
definitions, bootstraps, and model-level permutation tests remain those frozen
in `ICLR_UNIFIED_ATTRACTOR_AUDIT_PROTOCOL_20260731.md`.

## Frozen recurrence definitions

- **Output no-op:** two consecutive output states are identical.
- **Any output recurrence:** `o_t = o_(t-k)` for any `k >= 1`.
- **Persistent terminal fixed output:** a failed trajectory whose last three
  output states are identical.
- **Persistent terminal period-k output cycle:** for `k in {2,3,4}`, the last
  `2k` output states form two identical blocks of length `k`, with at least two
  distinct states in the block.  Use the smallest qualifying `k`.
- **Persistent terminal output recurrence:** persistent fixed output or
  persistent period-2--4 output cycle.

Periods 5--7 and single unconfirmed returns are reported as sensitivities but
do not enter the primary persistent-recurrence metric.

## Frozen landmark prediction

Use revision 4 as a fixed landmark.  Include only trajectories still active
after revision 4 (`rounds >= 6`), so every included case was exposed to the
same opportunity to stop by that point and had at least one later revision.

Exposure is any exact output recurrence among revisions 0--4.  Outcome is
final failure at revision 8.  Estimate the source-stratified, equal-checkpoint
risk difference

`P(final failure | recurrence by r4) - P(final failure | no recurrence by r4)`.

Report the per-checkpoint estimate and the 10,000 shared-case bootstrap
interval.  This is a prognostic association, not a causal effect.

## Updated instability burden

For model-level analyses define instability burden as the case rate of

`persistent terminal output recurrence OR primary basin escape`.

For the seven-checkpoint calibration bridge, use the same definition on the
same 408 case-disjoint trajectories.  The frozen bridge criterion remains a
negative full-panel slope and at least 20% lower LOOCV MAE than the intercept
baseline.

## Decision levels

### Strong output-dynamics account

All must hold:

1. persistent terminal output recurrence occurs in at least three independent
   model lineages;
2. it covers at least 10% of pooled failed trajectories;
3. the revision-4 landmark failure-risk difference has a 95% interval whose
   lower bound exceeds 10 percentage points;
4. at least eight of twelve checkpoint-level landmark differences are
   positive;
5. model-level persistent-recurrence burden correlates with final-joint
   success at `rho <= -0.60`, permutation `p < 0.05`;
6. the frozen seven-model calibration bridge passes.

This authorizes a main claim that closed-loop failure is organized by
persistent output recurrences and basin-escape hazards beyond the local median
response policy.

### Robust output-recurrence taxonomy

If conditions 1--5 hold but the calibration bridge fails, claim that recurrent
output patterns are widespread, strongly prognostic, and associated with the
cross-model endpoint, but do not claim they explain the local-policy
calibration residual.

### Descriptive recurrence only

Otherwise retain output recurrences and basin escapes as audited failure modes
without reorganizing the paper around them.

## Claim boundary

Never call an output recurrence a repeated full controller state.  Never claim
that continued retries are mathematically unable to escape, because the chat
history and revision position continue to change.  "Fixed point" and "cycle"
must be qualified as **output** patterns under the finite observed horizon.


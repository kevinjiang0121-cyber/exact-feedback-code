# Frozen protocol: recurrence-conditioned later rescue

Date frozen: 2026-08-03

## Question

Among trajectories still unresolved after an early closed-loop landmark, does
observable output recurrence predict failure to recover later, after
conditioning on the current error state, prior action history, checkpoint, and
source?

This is a read-only derived analysis of the completed 12-open plus 7-API
Combined-480 trajectories. It introduces no generation, model call, or
trajectory exclusion based on outcome.

## Primary estimand

- Landmark: revision 4.
- Risk set: a trajectory is unresolved at revision 4 and has at least one
  observed later revision.
- Exposure: repeat fraction among revisions 1--4, where recurrence is an exact
  return to a previously observed (text, signed error, missing-anchor set)
  state.
- Outcome: joint exact-length plus literal-anchor success during the remaining
  revisions through revision 8.
- Statistical unit: case trajectory.
- Uncertainty: case-ID-clustered sandwich covariance; the same case ID is a
  cluster across checkpoints.

## Adjustment set

Current state:

- signed and absolute error;
- log absolute error;
- target length;
- missing-anchor count;
- source;
- length band.

Prior action history:

- minimum absolute error;
- initial-to-current error improvement;
- contraction count;
- sign-flip count;
- length no-op count;
- largest absolute action;
- near-target basin escape.

Checkpoint fixed effects are included. The primary coefficient is the change
from zero to full repeat fraction.

## Panels

1. Open discovery panel: 12 checkpoints, seven lineages.
2. API external panel: seven models on the identical 480 cases.
3. Combined panel: 19 controllers with checkpoint fixed effects.

The API model definitions, providers, returned models, refusal treatment, case
IDs, and endpoint accounting remain frozen by the completed all-seven audit.
The single stable Gemini provider refusal is excluded from control estimands.

## Primary support gate

The stronger recurrence-conditioned rescue claim is supported only if:

1. the adjusted repeat-fraction odds ratio is below one in both open and API
   panels;
2. the API panel's case-clustered two-sided p-value is below 0.05;
3. the combined panel's p-value is below 0.001;
4. unadjusted recurrence-versus-no-recurrence rescue differences are negative
   in at least five of seven API models and all four sources;
5. the adjusted API coefficient remains negative at revision-3 and revision-5
   landmarks.

Failure of any gate prevents promotion to a cross-panel law. The existing open
recurrence prognosis and history-reset causal results remain valid.

## Secondary analyses

- positive- versus negative-current-error risk sets;
- revision-3 and revision-5 landmarks;
- per-model and per-source exposed/unexposed rescue rates;
- panel-level rank association between recurrence prevalence and later rescue;
- capture-only and capture/recurrence component audit for interpretation.

Secondary analyses do not redefine the primary gate.

## Claim boundary

Passing supports a cross-panel recurrence-conditioned non-recovery relation in
this exact-feedback assay. It does not establish a calibrated universal
probability law, cross-domain equivalence, subjective text quality, or a
portable treatment.

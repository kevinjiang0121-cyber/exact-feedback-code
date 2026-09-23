# ICLR Capture--Recurrence Law Audit

Date: 2026-08-01  
Status: frozen before results

## Question

Across model families, does early output recurrence contain transferable
information about whether an exact-feedback loop will later reach the target,
beyond its current signed error and ordinary progress history?

## Population and landmark

Use the same twelve canonical open-weight Combined-480 checkpoints and audited
raw assets as `ICLR_UNIFIED_OUTPUT_RECURRENCE_PROTOCOL_V2_20260731.md`.

Primary landmark: revision 4.  Include a model-case only if it remains active
after revision 4, operationalized as at least six stored rounds (revisions
0--5).  This excludes cases already successful by revision 4 and guarantees at
least one later opportunity.  The label is `late_rescue`: final joint success
during revisions 5--8.

All features are computed from revisions 0--4 only.  No final-round or
post-landmark information enters a predictor.

## Output recurrence

For fixed case contract define

`o_t = (SHA256(text_t), error_t, sorted(missing_anchors_t))`.

For revisions 0--4 compute:

- recurrence count: number of outputs equal to any earlier output state;
- repeat fraction: recurrence count divided by four transitions;
- any recurrence;
- unique-output ratio;
- longest consecutive identical-output run;
- observed period-2 return.

These are output-history features, not repeated full controller states.

## Frozen nested predictors

All predictors are deterministic L2 logistic regressions with `C=1`, no class
reweighting, training-fold standardization, and one-hot categorical encoding.
No hyperparameter is tuned on held-out families.

### B0: current state

- current signed error, absolute error, and `log1p(abs(error))` at revision 4;
- target length and current missing-anchor count;
- source family and frozen length band.

### B1: current state plus generic progress history

B0 plus:

- minimum absolute error through revision 4;
- initial-to-current absolute-error improvement;
- contraction count;
- sign-flip count;
- length no-op count;
- largest absolute action;
- basin escape by revision 4 under the frozen `20 -> 50` definition.

### B2: progress plus recurrence

B1 plus the six frozen output-recurrence features above.

The primary comparison is B2 versus B1.  It asks whether recurrence transfers
beyond ordinary progress, not merely beyond a weak endpoint baseline.

## Leave-one-family-out evaluation

Families are:

- Llama;
- Gemma;
- GLM;
- Mistral/Ministral;
- Qwen, holding all six Qwen checkpoints out together;
- Granite;
- Falcon-H1.

Train on six families and predict every landmark-risk case in the held family.
Primary metric: Brier score.  Supporting metrics: log loss, AUROC where both
classes occur, and calibration slope/intercept.

Aggregate families with equal weight.  Use 10,000 shared-case,
source-stratified bootstrap replicates on the fixed out-of-family predictions.
Report B2-minus-B1 Brier difference and relative reduction.  A family improves
when its B2 Brier is lower than B1.

## Common conditional effect

Fit one prespecified binomial GLM on the complete landmark panel:

`late_rescue ~ repeat_fraction + B1 numeric controls + model fixed effects + source fixed effects + length-band fixed effects`.

Use case-ID clustered sandwich standard errors, clustering the same case across
checkpoints.  The common recurrence effect is the odds ratio for
`repeat_fraction`; it must be below one.  This is an adjusted association, not
a causal effect.

Also report the unadjusted within-checkpoint late-rescue risk difference for
any recurrence versus no recurrence.  Consistency is directional, not equality
of effect magnitude.

## Early hazard ratio

Using revisions 0--4 for all 480 cases, estimate for each checkpoint:

- early capture hazard: transitions ending in joint success divided by active
  transitions;
- early output-recurrence hazard: transitions returning to an earlier output
  divided by active transitions;
- smoothed capture--recurrence ratio `(captures + 0.5)/(recurrences + 0.5)`.

Test its Spearman association with Combined-480 final-joint success using
100,000 seeded model-label permutations.  Repeat exactly over the five Qwen3
Dense sizes with the exact 5! permutation distribution.

## Frozen robustness

- repeat the landmark analysis at revisions 3 and 5 with otherwise identical
  definitions;
- repeat the common directional contrast within each of the four source
  families;
- report models at outcome ceilings or with fewer than five exposed or
  unexposed landmark cases rather than silently excluding them;
- report escape as a separate feature and never redefine recurrence to absorb
  escape events.

## Decision levels

### Strong cross-family law

All must hold:

1. adjusted repeat-fraction odds ratio is below one with clustered `p < 0.01`;
2. any-recurrence late-rescue risk difference is negative in at least ten of
   twelve checkpoints and all four source families;
3. B2 improves equal-family leave-one-family-out Brier score with a 95%
   interval excluding zero, at least 5% relative reduction, and improvement in
   at least five of seven held families;
4. the early capture--recurrence ratio correlates with final success at
   `rho >= 0.70`, permutation `p < 0.05`;
5. landmarks 3 and 5 preserve the adjusted negative direction and the B2
   predictive advantage.

This authorizes:

> Exact-feedback reliability is governed by competition between target capture
> and output recurrence: after conditioning on current error and generic
> progress, recurrence reduces transferable late-rescue probability, and the
> early capture--recurrence balance predicts checkpoint reliability.

### Consistent prognostic law

If conditions 1, 2, and 4 pass but cross-family B2 materiality or landmark
robustness fails, claim a consistent adjusted/prognostic relationship without
claiming a portable prediction rule.

### No universal law

Otherwise retain the twelve-model recurrence taxonomy and report the
cross-family test as a boundary.  Do not tune the landmark, features, or family
partition after seeing results.

## Outputs

- integrity-linked input manifest;
- landmark case table with only pre-landmark features;
- held-family predictions and per-family metrics;
- common-effect and directional-consistency tables;
- early hazard table and permutation tests;
- landmark/source sensitivity tables;
- machine-readable verdict and concise acceptance document.


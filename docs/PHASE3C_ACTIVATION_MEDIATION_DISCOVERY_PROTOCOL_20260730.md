# Phase 3C activation-mediation discovery protocol

Date frozen: 2026-07-30, before patched generation  
Accelerator: physical GPU2 only

## Question

Do checkpoint-specific activations at the two Gate-B-selected functional
locations mediate Meta Instruct's feedback-conditioned correction phenotype,
or is the effect dominated by the unpatched checkpoint computation and its
interaction with the injected state?

## Frozen scope

Use the byte-identical Base/Meta-Instruct coordinate system, exact Meta chat
serialization, and the 96 case-clustered discovery fixed states from Phase 3A.
The two intervention units are:

1. the complete feedback-message token span at post-block residuals 8--15;
2. the complete feedback-message token span at the post-block-31 residual,
   immediately before the final norm/output map.

The second unit tests whether the output-map necessity found in Phase 3B is
accompanied by checkpoint-specific final residual mediation. It does not
replace or reinterpret the output-weight intervention.

For every state and unit run:

- denoising: clean Meta-Instruct activations injected into Base;
- noising: clean Base activations injected into Meta Instruct.

Clean Base and clean Meta-Instruct rows are reused from Phase 3A. No
single-layer search, steering coefficient, training, Tülu exchange, or Phase
3C confirmation condition is authorized in this discovery stage.

## Intervention semantics

- Donor and recipient receive exactly the same token IDs.
- The feedback-message content span is located as a token subsequence inside
  the exact rendered prompt and patched in full.
- For blocks 8--15, every selected post-block residual is replaced by its
  clean donor counterpart during the prompt prefill only.
- For the output boundary, only the post-block-31 residual is replaced.
- Generated tokens are not patched after prefill.
- Greedy decoding and the frozen `max(128, target * 3)` token ceiling match
  Phase 3A.

## Outcomes and interaction accounting

For each unit and state family (`correction`, `near_target`) report:

- total checkpoint effect:
  `TE = Y(Instruct, Instruct-state) - Y(Base, Base-state)`;
- denoising indirect effect:
  `IE_B = Y(Base, Instruct-state) - Y(Base, Base-state)`;
- noising loss:
  `IE_I = Y(Instruct, Instruct-state) - Y(Instruct, Base-state)`;
- mediated interaction:
  `INT = IE_I - IE_B`.

The primary binary outcome is correct revision direction. Secondary outcomes
are clipped calibration utility `-min(|action_gain - 1|, 10)`, exact length,
anchor retention, and joint success. The independent resampling unit is the
24-case cluster, not the 96 states.

## Frozen discovery selection

A unit-family pair may advance only if:

1. denoising and noising both move direction correctness toward their
   corresponding clean checkpoint contrast;
2. neither direction is sign-reversing;
3. the average of denoising gain and noising loss is positive;
4. calibration utility does not tell a materially contradictory story.

Rank eligible pairs by the mean bidirectional direction effect and select at
most one. Selection authorizes only the preregistered confirmation and
negative-control stage: mismatched-case donor, sign-preserving donor, and a
size-matched neighboring boundary/band.

If no pair is eligible, stop activation localization and report a
checkpoint-level distributed interaction. Do not search the other 30 layers.

## Frozen artifacts

- fixed states SHA-256:
  `0b4c7d6ea80798699e3cff05c6008dc845634f026e98456ce3699162300bad09`
- exact Meta template SHA-256:
  `e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65`
- clean Base and Meta-Instruct rows:
  `experiments/phase3_causal_localization_v1/phase3a_fixed_states`

Phase 3C confirmation remains unauthorized until this discovery is accepted.

# Phase 3D coordinated-controller graft discovery protocol

Date frozen: 2026-07-30, before graft scoring  
Accelerator: physical GPU2 only

## Question

Can the two parameter groups found necessary in Phase 3B transfer
feedback-conditioned exact-length control into the Base checkpoint when
grafted jointly, or are they only necessary inside the full Instruct
checkpoint?

## Coordinate and interface control

All conditions use:

- Meta Llama 3.1 8B Base as the recipient tensor checkpoint;
- Meta Instruct tokenizer, generation metadata, and exact Meta chat template;
- identical BF16 greedy decoding and maximum eight feedback revisions.

The metadata choice is fixed across every cell. It makes the output-map graft
usable with the same `<|eot_id|>` stopping contract under which that map was
trained. The sham condition contains all Base tensors under this same
metadata, so no weight effect is confounded with serialization or stopping
configuration.

## Frozen discovery conditions

1. `base_meta_contract_sham`: all Base tensors;
2. `graft_blocks_08_15`: Instruct blocks 8--15 into Base;
3. `graft_output`: Instruct final norm plus LM head into Base;
4. `graft_blocks_08_15_plus_output`: both selected groups;
5. `graft_noncontiguous_plus_output`: Instruct blocks
   `0,4,8,12,16,20,24,28` plus output, matching the joint graft's layer count
   and output package.

Embeddings and every unselected tensor remain Base. Each checkpoint records
all selected keys, parameter counts, shard modes, and hashes.

## Frozen outcomes

Every condition runs:

- the 96 Phase-3A discovery fixed states;
- the frozen Main-120 closed loop.

Primary proximal outcome: correct revision direction.  
Primary downstream outcome: final joint success.

Secondary outcomes: clipped calibration utility
`-min(|action_gain - 1|, 10)`, exact next revision, recurrence, terminal
self-loop, and terminal absolute error.

All comparisons are paired. Fixed-state intervals resample the 24 case
clusters. Main-120 intervals use source-stratified paired bootstrap.

## Sufficiency and synergy gate

Let `B`, `M`, `O`, `MO`, and `CO` denote sham Base, middle graft, output
graft, joint middle-plus-output graft, and matched-control-plus-output.

The joint graft is eligible for confirmation only when:

1. `MO - B` is positive on direction and final joint, with both 95%
   intervals excluding zero;
2. `MO - CO` is positive on direction and final joint, with both intervals
   excluding zero;
3. factorial synergy `MO - M - O + B` is positive with an interval excluding
   zero on at least one primary outcome and is nonnegative in point estimate
   on the other;
4. calibration utility does not materially contradict the primary outcomes.

If the gate passes, authorize only:

- an alpha-0.5 joint graft;
- Combined-480 confirmation for `B`, `MO`, and `CO`.

If the gate fails, stop the controller-graft route. Do not add layers,
learned adapters, low-rank fitting, or coefficient tuning to this paper.

## Claim boundary

A discovery pass would establish candidate parameter-graft sufficiency and
synergy, not yet a transferable method. The method claim requires monotonic
half-dose behavior and case-disjoint Combined-480 confirmation.

Phase 3D is separate from the completed Phase 3C activation-mediation tree.

## Frozen artifacts

- Main-120 SHA-256:
  `bc36452c8a4d296d80e2cfd49c235a33517be93a0a32849fa576fe87235b0a05`
- fixed states SHA-256:
  `0b4c7d6ea80798699e3cff05c6008dc845634f026e98456ce3699162300bad09`
- exact Meta template SHA-256:
  `e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65`

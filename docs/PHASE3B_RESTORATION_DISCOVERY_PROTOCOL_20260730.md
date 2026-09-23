# Phase 3B parameter-restoration discovery protocol

Date frozen: 2026-07-30, before restoration scoring  
Accelerator: physical GPU2 only

## Question

Which coarse Base-to-Instruct parameter segment is necessary for Meta
Instruct's feedback-conditioned correction profile under exact Meta
serialization?

## Frozen conditions

Recipient checkpoint: Meta Llama 3.1 8B Instruct.  
Donor checkpoint: Meta Llama 3.1 8B Base.

Seven discovery conditions:

1. restore blocks 0--7;
2. restore blocks 8--15;
3. restore blocks 16--23;
4. restore blocks 24--31;
5. restore final norm plus LM head;
6. sham-copy Instruct blocks 0--7 through the same materialization path;
7. restore noncontiguous blocks `0,4,8,12,16,20,24,28`.

Embeddings, tokenizer, exact Meta chat template, generation configuration, and
all unselected Instruct parameters remain fixed. The noncontiguous control
matches the eight-block parameter count of every contiguous block condition.
The output group is intentionally smaller and is interpreted separately.

## Discovery outcomes

Two frozen discovery views are run for every condition:

- proximal: the 96 previously designated discovery fixed states from Phase 3A,
  one greedy next revision per state;
- downstream: the frozen source-stratified Main-120 closed loop, greedy,
  maximum eight revisions.

Primary screen quantity:

`direction loss = sham correct-direction rate - restored-condition rate`

on the 96 fixed states.

Secondary quantities:

- change in median absolute calibration error `|action_gain - 1|`;
- exact next-revision rate;
- final Main-120 joint-success loss relative to sham;
- final exact loss, recurrence, and terminal self-loop.

The independent unit for fixed-state intervals is the 24-case cluster, not 96
states. Main-120 comparisons are paired by case.

## Frozen selection rule

A contiguous band is eligible for the next stage only if:

1. direction loss is positive;
2. Main-120 final-joint loss is nonnegative;
3. the two outcomes do not tell a materially contradictory story.

The noncontiguous control is not eligible for selection. If it is at least as
damaging as a contiguous band on both direction and final joint success, that
band is not treated as localized evidence.

Rank eligible contiguous bands first by direction loss, then by final-joint
loss. Select at most two. Selection authorizes only the previously planned
alpha=0.5 dose and confirmation design; it does not itself establish
necessity.

If no band is eligible, stop the localized-parameter route. Do not begin a
32-layer search.

## Execution and provenance

- Hybrid checkpoints are materialized shard-by-shard.
- Selected tensors come from Base; all remaining tensors come from Instruct.
- Sham selected tensors come from Instruct itself.
- Every selected key, parameter count, shard mode, and shard SHA-256 is stored.
- Untouched shards may be hard-linked to the immutable Instruct checkpoint;
  touched shards are rewritten as safetensors.
- Every behavioral row is independently recounted and anchor-audited.

Phase 3C remains unauthorized until the restoration discovery, dose, and
confirmation gates are evaluated.

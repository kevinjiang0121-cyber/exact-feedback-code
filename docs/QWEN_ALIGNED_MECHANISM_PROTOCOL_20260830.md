# Aligned Qwen causal-mechanism protocol

Date frozen: 2026-08-30  
Experiment: `qwen_aligned_mechanism_v1`  
Status: frozen before generation

## Question

Can the same causal ladder used for aligned Llama distinguish how Qwen3-8B
post-training changes conversion of exact signed feedback into revision
actions?  The confirmatory contrast is that Llama mainly gains direction,
whereas prior Qwen system identification shows lower overshoot together with a
larger zero-action dead zone.

## Models and common interface

- donor/unaligned: Qwen3-8B-Base;
- aligned: Qwen3-8B;
- architecture: 36 transformer blocks, hidden size 4096;
- interface: the Qwen3-8B Instruct tokenizer, chat template, generation
  configuration, and termination token for every condition;
- tensors: Base or Instruct tensors as specified per intervention; embeddings
  are never transplanted.

Q0 requires identical architecture fields, tensor keys and shapes, vocabulary
and merges, and identical rendered text and token IDs for all 288 fixed states.
The Base common-interface checkpoint must reproduce the Base tensor hashes
while using only Instruct interface metadata.  Any failure stops the study.

## Frozen data and outcomes

- fixed states: the existing 72 case-disjoint Phase3A cases, four states per
  case (288 total), with 24 discovery and 48 confirmation cases;
- closed loop: frozen Main120 for discovery and Combined480 for confirmation;
- maximum revisions: eight after the initial draft;
- primary action outcomes: direction correctness, zero action, overshoot,
  absolute gain error, and exact/joint capture;
- primary trajectory outcomes: final joint success, terminal absolute error,
  recurrence, and terminal self-loop;
- intervals: case-clustered for fixed states and source-stratified paired
  bootstrap for closed-loop outcomes.

## Q1: matched residual probes

Capture all 37 residual-stream states (embedding plus 36 blocks) at the
current-count, target, action-magnitude, and final-prompt positions for both
checkpoints.  Ridge alpha is selected using six-fold GroupKFold on the 24
discovery cases at the final layer, then frozen across layers and evaluated on
the 48 confirmation cases.  Final-layer results are primary; best-layer
results are descriptive only.  Probes are diagnostic, not causal.

## Q2: restoration discovery

Stream Base tensors into the Instruct recipient under seven conditions:

- sham Instruct copy;
- nine-block noncontiguous control: blocks 0, 4, 8, 12, 16, 20, 24, 28, 32;
- contiguous blocks 0--8, 9--17, 18--26, or 27--35;
- output package: final norm plus language-model head.

Each condition runs the 96 discovery fixed states and Main120 closed loop.
Select at most two groups.  A group is eligible only if its paired change from
sham is in the Base direction for at least one preregistered Qwen signature
(more zero action or more overshoot), does not improve final success, and is
larger than the matched noncontiguous control on the selecting signature.
Selection ranks the mean of standardized zero-action and overshoot effects,
then terminal absolute-error degradation.  Ties use the earlier group in the
frozen order above.

## Q3: dose and case-disjoint confirmation

For each selected group, evaluate alpha=0.5 and alpha=1.0 restoration.
Discovery dose uses the 96 fixed states and Main120.  Confirmation uses the
192 fixed states and Combined480; the existing Main120 rows may be reused only
as the first audited portion of Combined480.

Necessity passes only when all hold:

1. the full-dose confirmation effect is in the discovery direction;
2. full dose exceeds the noncontiguous control on the selecting signature;
3. alpha=0.5 lies directionally between sham and full dose, allowing sampling
   error but forbidding a sign reversal larger than the full-dose effect;
4. at least one selecting-signature paired 95% interval excludes zero;
5. recount and row-integrity audits pass.

If no group passes, Q4 and Q5 do not run.

## Q4: activation transplantation

For Q3-confirmed transformer groups, patch only the feedback-message token
span during prefill, in both Base-to-Instruct and Instruct-to-Base directions.
The output package is represented by the final transformer block activation;
the final norm/head itself is not an activation site.  Run sham validation and
the 96 discovery states first; open the 192 confirmation states only if both
directions move at least one selecting signature toward the donor without a
larger adverse calibration change.

An activation-mediation claim requires the same-direction bidirectional effect
on case-disjoint confirmation and stable interaction with state family.

## Q5: coordinated controller graft

Using Base tensors under the common Instruct interface, evaluate:

- Base common-interface sham;
- each confirmed transformer group alone;
- output package alone;
- matched noncontiguous plus output;
- each confirmed transformer group plus output.

Discovery uses 96 fixed states and Main120.  A joint graft advances to
Combined480 only if it improves the Qwen selecting signature versus Base and
the matched control without worsening the other primary signature.  Recovery
of final joint success is an outcome, not a prerequisite for learning where
the controller acts.

## Paper decision gate

### Main text

Require Q0 integrity, Q1 case-disjoint representation evidence, at least one
Q3-confirmed parameter group, and either Q4 mediation or Q5 controlled
transfer.  The Qwen result must sharpen a cross-family mechanism contrast and
not merely repeat the Llama paragraph.

The frozen Q1 representation gate is final-layer signed-error sign accuracy
of at least 0.80 in both checkpoints and action-magnitude classification
accuracy of at least 0.60 in at least one checkpoint.  Passing the automated
gate yields a main-text *candidate*; the final semantic non-redundancy check is
reported separately and cannot be manufactured by changing thresholds.

### Appendix

Use when Q0 passes and the complete gated study yields a credible negative,
partial, or redundant result that bounds generality.

### Exclude

Use when Q0 fails, required audits fail, or only discovery/uncontrolled effects
remain.  No paper text is edited before this decision.

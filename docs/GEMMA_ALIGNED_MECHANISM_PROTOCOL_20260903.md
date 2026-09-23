# Aligned Gemma causal-mechanism protocol

Date frozen: 2026-09-03  
Experiment: `gemma_aligned_mechanism_v1`  
Status: frozen before Gemma Base outcomes

## Question

Does Gemma 2 9B post-training produce a feedback-to-action controller with the
Llama-like partial-transfer signature, the Qwen-like representation-without-
transfer signature, or a third causal regime?

This is a discriminating third-family study. It is not a search for a positive
mechanism result. Negative and mixed gates are retained and reported.

## Models and common interface

- Base: `google/gemma-2-9b` at revision
  `33c193028431c2fde6c6e51f29e6f17b60cbfac6`;
- Instruct: the existing `gemma-2-9b-it` snapshot at revision
  `11c9b309abf73637e4b6f9a3fa1e92e615547819`;
- architecture expected before opening outcomes: 42 Gemma2 blocks, hidden size
  3584, vocabulary size 256000;
- interface: Instruct tokenizer, chat template, generation configuration, and
  termination tokens for every condition;
- tensors: Base or Instruct tensors as specified by each intervention;
  embeddings are never transplanted.
- runtime/storage dtype: all conditions load in BF16. The published Base
  checkpoint stores FP32 tensors, so transplanted Base values are explicitly
  cast to the BF16 recipient dtype when materialized; this changes storage
  precision, not the frozen intervention membership.

G0 requires equality of the preregistered architecture fields, tensor-key sets
and tensor shapes, lexical tokenizer assets, and rendered prompt token IDs for
all 288 fixed states. Any mismatch stops causal interventions. Interface-only
metadata differences are allowed because every condition uses the frozen
Instruct interface.

## Frozen data and endpoints

- fixed states: the existing 72 case-disjoint Phase3A cases, four states per
  case, with 96 discovery and 192 confirmation states;
- closed loop: Main120 for discovery and Combined480 for confirmation;
- maximum revisions: eight after the initial draft;
- fixed-state utilities: direction correctness, nonzero action, overshoot
  avoidance, negative absolute gain error, and exact/joint capture;
- trajectory outcomes: final joint success, terminal absolute error,
  recurrence, and terminal self-loop;
- uncertainty: case-clustered bootstrap for fixed states and source-stratified
  paired bootstrap for trajectories.

The existing Gemma-IT response-law confirmation is provenance only. It reports
84.1% direction correctness, 12.2% zero action, and 32.3% overshoot. It does
not determine which Base--IT contrast must be found.

## G1: residual probes

Capture all 43 residual states (embedding plus 42 blocks) at the current-count,
target, action-magnitude, and final-prompt positions. Ridge alpha is selected
by GroupKFold on the 24 discovery cases, frozen across layers, and evaluated
on the 48 confirmation cases. Final-layer results are primary. Probes are
diagnostic and cannot establish causal use.

The representation gate is final-layer signed-error sign accuracy of at least
0.80 in both checkpoints and action-magnitude classification accuracy of at
least 0.60 in at least one checkpoint.

## G2: restoration discovery

Restore Base tensors into the Instruct recipient under:

- Base common-interface reference;
- Instruct sham;
- eleven-block noncontiguous control: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36,
  and 40;
- contiguous blocks 0--9, 10--20, 21--30, or 31--41;
- output package: final norm plus a separately stored language-model head when
  present. Gemma 2 stores no independent `lm_head.weight`; its output head is
  tied to `model.embed_tokens.weight`. Because embeddings are never
  transplanted, the Gemma output package is therefore the final norm only.

Every condition runs 96 discovery fixed states and Main120. Let `u_B-u_I` be
the Base-minus-Instruct contrast over the four preregistered fixed-state
utilities: direction correctness, nonzero action, overshoot avoidance, and
negative absolute gain error. A restoration component is aligned only when
its change from Instruct has the same sign as `u_B-u_I` and its absolute
effect exceeds the matched noncontiguous control for that utility.

A group is eligible when at least one utility is aligned, final joint success
does not improve over Instruct, and integrity passes. Rank by the sum of
clipped component-wise progress fractions toward Base, then by final-success
degradation, then by the frozen group order. Select at most two groups.

## G3: dose and case-disjoint confirmation

For each selected group, evaluate alpha 0.5 and 1.0 restoration on the 192
confirmation states and Combined480. A group is confirmed only when:

1. every selecting utility moves in the discovery direction;
2. full dose exceeds the noncontiguous control on every selecting utility;
3. half dose does not reverse by more than the full-dose effect;
4. at least one selecting-utility 95% interval excludes zero;
5. final joint success does not improve over Instruct; and
6. all row, recount, and trajectory audits pass.

If no group is confirmed, G4 and G5 stop automatically.

## G4: activation mediation

For confirmed transformer groups, patch only the feedback-message span during
prefill in both directions. Discovery opens first. Confirmation opens only if
both directions move at least one selecting utility toward the donor without
a larger adverse calibration change. A mediation claim requires the same
direction on case-disjoint confirmation and no state-family contradiction.

## G5: coordinated controller graft

Using Base tensors under the common Instruct interface, compare Base sham,
each confirmed transformer group, output package, noncontiguous plus output,
and each confirmed group plus output. A joint graft advances from Main120 to
Combined480 only if it moves at least one selecting utility from Base toward
Instruct beyond the matched control and does not reverse another selecting
utility. Direction recovery and terminal capture are reported separately.

## Placement rule

- **Main-text candidate:** G0 integrity, G1 representation, at least one G3
  confirmed group, and G4 mediation or G5 controlled transfer. The result must
  distinguish a cross-family regime rather than repeat the Llama/Qwen text.
- **Appendix:** valid complete gated study with a negative, partial, or
  redundant result.
- **Exclude:** G0 failure, audit failure, or discovery-only uncontrolled
  effects.

No paper or Overleaf file may change before placement adjudication.

## Runtime amendment before G2 outcomes

The first Base closed-loop attempt showed that full eight-revision histories
can exceed Gemma 2's declared 8192-token context (the first observed overflow
was 8281 tokens at revision 6). Truncating history would change the controller
interface, while the original Transformers execution path does not enforce the
configuration limit at request validation. Therefore every Gemma closed-loop
condition retains the full frozen conversation and uses vLLM's explicit
long-context override with `max_model_len=32768`. Fixed-state tests remain
within 8192. This is a matched runtime amendment, not a gate or outcome change;
the override and the pre-outcome failure are retained in runtime/recovery
manifests. Any numerical instability or execution failure remains a failed
audit rather than grounds for another protocol change.

# Phase 3B dose and confirmation protocol

Date frozen: 2026-07-30, before dose scoring or confirmation generation  
Accelerator: physical GPU2 only

## Inputs selected without post-selection expansion

The accepted restoration-discovery screen selected exactly two candidates:

1. Base restoration of Meta Instruct blocks 8--15;
2. Base restoration of Meta Instruct final norm plus LM head.

No neighboring layer search, embedding condition, or Phase 3C activation
intervention is authorized by this protocol.

## Dose test

For each selected candidate, materialize one half-restoration checkpoint:

`theta_half = 0.5 * theta_instruct + 0.5 * theta_base`

for selected tensors only. Every other tensor, tokenizer asset, interface, and
generation setting remains Meta Instruct. Evaluate each half restoration on:

- the frozen 96 discovery fixed states;
- the frozen Main-120 closed loop.

The already completed sham and full-restoration rows are reused byte-for-byte.
For both fixed-state direction loss and Main-120 final-joint loss, strict dose
support means:

`0 < loss_half <= loss_full`.

Failure of strict monotonicity is reported as nonlinearity; it does not
authorize another alpha value.

## Case-disjoint confirmation

Extend exactly four rows to Combined-480 under the exact Meta serialization:

1. sham-copy Meta Instruct reference;
2. selected blocks 8--15 full restoration;
3. selected output full restoration;
4. frozen eight-block noncontiguous control.

Completed Main-120 rows are reused. Only the frozen Replication-120 and
Extension-240 cases are newly generated. The primary confirmation sample is
the 360 newly evaluated cases, which are disjoint from discovery. Combined-480
is the precision estimate.

Primary effect:

`final-joint loss = sham success - restored-condition success`.

Report source-stratified paired bootstrap intervals on the new 360 and all
480 cases. Also report the candidate's excess loss over the noncontiguous
control. Output restoration has a smaller parameter count and is interpreted
as a distinct output-map intervention, not as a size-matched layer-band
comparison.

## Frozen Gate B

A selected intervention supports a necessity claim only when:

1. its discovery effect has the preregistered direction;
2. the new-360 paired 95% interval for final-joint loss excludes zero;
3. its new-360 loss exceeds the noncontiguous control, with a paired 95%
   interval excluding zero;
4. proximal and closed-loop phenotypes are coherent;
5. strict half-dose monotonicity holds on direction loss and final-joint loss.

If an intervention fails only item 5, report a full-restoration necessity
effect with nonlinear dose response, not a graded localized mechanism. If
items 1--4 fail, do not make a localized necessity claim.

Phase 3C remains unauthorized until this gate is evaluated.

## Frozen artifacts

- Primary-120 SHA-256:
  `bc36452c8a4d296d80e2cfd49c235a33517be93a0a32849fa576fe87235b0a05`
- Replication-120 SHA-256:
  `312d3b101613feb2a938c865b41612424c40cd0b5c4afd401ac4959496b8c4ae`
- Extension-240 SHA-256:
  `7e82a1e05c62c438c5c9d60d17e09efb3d5edcc37d39d75d8d03b883a2308c4b`
- Combined-480 SHA-256:
  `6ffebda72f0e7d26797dc31408d8209fe9b4267acae2f787421572ec01daed0a`
- Fixed-state SHA-256:
  `0b4c7d6ea80798699e3cff05c6008dc845634f026e98456ce3699162300bad09`
- Exact Meta template SHA-256:
  `e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65`

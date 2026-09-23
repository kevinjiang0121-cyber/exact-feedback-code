# Revision-32 persistence test

Date frozen: 2026-07-24

## Question

Are revision-8 failures merely slow controllers that close under a larger
unchanged action budget, or do failures persist through revision 32?

## Cohort

For each of the six primary models, select up to 24 failures from the frozen
Combined-240 natural-generation experiment. Eligibility requires:

- final joint failure at revision 8;
- nonzero terminal word-count error;
- complete revisions 0 through 8.

Selection is deterministic and model-specific. Eligible cases are partitioned
by the four source corpora and terminal absolute-error region (`1-5`, `6-20`,
`>20`). A stable-hash round-robin takes one case per nonempty stratum per pass
until 24 cases are selected or eligibility is exhausted. Anchor-only failures
at zero word-count error are excluded and counted in the manifest.

This stratified diagnostic subset is not a prevalence sample and is not used
to re-estimate population-wide revision-32 success.

## Intervention

Reconstruct the byte-identical conversation through revision 8 from the frozen
trajectory. Continue the same model, native chat template, greedy decoding,
deterministic counter, lexical-anchor contract, feedback message, and
full-history conversation through revision 32. Stop at the first joint success.

No rolling context, summarization, prompt change, or repair tool is introduced.
Before every generation, verify that the native context window can accommodate
the rendered prompt plus the frozen maximum generation allowance. Native
context exhaustion is reported as a separate system outcome, not silently
pooled with persistent control failure.

## Endpoints

Primary:

- additional joint closure by revision 32 among the selected revision-8
  failures, with exact counts and 95% Wilson intervals.

Supporting:

- cumulative additional closure at revisions 12, 16, 20, 24, 28, and 32;
- persistent nonclosure at revision 32;
- native context exhaustion;
- no-improvement persistence (no absolute-error improvement over revision 8);
- terminal stasis (the last four generated signed errors are identical);
- persistent oscillation (at least four nonzero-error sign flips after
  revision 8).

All labels and audits are deterministic. The paper may conclude that failures
persist under a 32-revision budget within this sampled protocol. It may not
claim infinite-horizon impossibility.

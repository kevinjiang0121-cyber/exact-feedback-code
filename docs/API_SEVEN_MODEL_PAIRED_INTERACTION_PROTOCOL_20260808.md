# Seven-model API paired and interaction analysis protocol

Date frozen: 2026-08-08  
Status: analysis specification frozen after marginal success rates were observed and before running the paired-bootstrap and interaction tests. This is not described as preregistration.

## Question

Under the same deterministic verifier and matched cases, do controller differences remain stable across exact-length, lexical, and compositional assays, or are they systematically modified by assay and constraint structure?

## Frozen panel

- Models: GPT-5.6 Sol, Claude Opus 5, Gemini 3.6 Flash, GLM-5.2, Qwen3.7 Plus, Llama 3.1 70B, DeepSeek V4 Flash.
- Assays: exact length, lexical constraints, compositional constraints.
- Structures in the two new assays: c05, c07, c12, c09, c10.
- Decoding, verifier, case order, provider/model route, and eight-revision budget remain as recorded in the canonical runs.
- No model-level aggregate revision score will be constructed.

## Integrity and evaluability

- A row is evaluable when `api_error=null`, deterministic rounds/outcomes are present, and either `protocol_complete=true` or the legacy exact-length row predates that field but passes the canonical all-seven audit. An explicit `protocol_complete=false` is never evaluable.
- All model comparisons are paired on common case IDs within an assay or structure.
- The exact-length Gemini provider refusal is retained as a provider event but excluded from model-behavior estimands; comparisons involving Gemini use the common evaluable cases.
- Interaction regressions use the all-seven common-case intersection within each assay to retain a balanced model-by-case panel.

## Outcomes

1. Primary: final deterministic joint success.
2. Secondary: any exact-text recurrence within the trajectory.
3. Descriptive: one-shot joint success and terminal exact-text recurrence.

Recurrence means that the SHA-256 hash of any generated text repeats earlier in the same trajectory.

## Paired bootstrap

- 20,000 resamples, seed 20260808.
- Resample case IDs with replacement within source strata separately for each assay.
- Apply the identical sampled case indices to every model in a paired comparison.
- Report percentile 95% intervals for model rates and pairwise rate differences.
- Report exact two-sided McNemar tests for pairwise binary outcomes.
- Holm-adjust the 21 model-pair tests separately within each assay and outcome family.

## Interaction analysis

### Assay-level

- Fit a balanced linear-probability model with model, assay, and model-by-assay fixed effects.
- Cluster the sandwich covariance by assay-specific case ID.
- The primary omnibus test is a Wald test that all model-by-assay interaction coefficients equal zero.
- Report model-pair difference-in-differences for all three assay pairs; Holm-adjust the 63 interaction contrasts within each outcome.

### Structure-level

- On lexical/compositional cases, fit model, structure, and model-by-structure fixed effects.
- Cluster by structure-specific case ID.
- Test all model-by-structure coefficients jointly.
- Report cell rates and selected largest adjusted interaction contrasts; do not turn them into a model ranking.

## Robustness and interpretation

- Refit the assay-level omnibus model leaving out each controller once.
- Refit the structure-level omnibus model leaving out each structure once.
- Statistical interaction establishes heterogeneous behavior across the frozen case distributions. It does not by itself identify architecture, training data, or an internal neural cause.
- A significant interaction supports constraint-dependent control policies; it does not imply that every individual model pair reverses ordering.

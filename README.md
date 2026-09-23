# Exact Feedback Is Not Control — experimental reproduction

This package contains experimental code, frozen inputs and configuration templates. The separate evidence package contains frozen outputs and measurements. Manuscript source, plotting and typesetting are outside this release.

## Repository scope

This is the public code-only reproduction repository for *Exact Feedback Is Not Control*. It includes experiment runners, analysis implementations, configuration templates, tests, and the frozen inputs required to reproduce the reported protocols. Frozen inputs include initial drafts and selected intervention states; these are required experimental inputs, not an archive of evaluation results.

Experiment outputs, frozen result reports, activation arrays, annotation caches, model weights, and API credentials are not included. Manuscript source and figure-production code are also excluded. Dataset and third-party software terms are documented in THIRD_PARTY_NOTICES.md.

## Quick start

Use Python 3.12 and configure your own model paths or API credentials following **Model experiments** below. Run `python reproduce.py list` to list available experiments. Full local generation requires a compatible Linux/CUDA/vLLM environment. No new GPU/API generation was executed when preparing this repository; validation coverage and limitations are documented in VALIDATION.md.

## Offline analysis

Use Python 3.10 or newer; Python 3.12 is recommended. Install the analysis dependencies and provide an independently obtained evidence directory (not included in this repository):

```bash
pip install -r requirements-analysis.txt
python reproduce.py evidence-check --materials ../evidence
python reproduce.py analyze-evidence --materials ../evidence --output analysis/full --jobs 2
```

No weights, API credentials or GPU are required. `--only history_trigger fixed_draft_crossover` selects analyses; prerequisites are included automatically. Use a new output directory for each execution. Full statistical resampling can take substantial CPU time. See `experiment_index.json` for every experiment-to-analysis mapping and `VALIDATION.md` for the checks actually executed for this release.

## Model experiments

Use Linux with a supported CUDA/vLLM environment for local generation:

```bash
pip install -r requirements.txt
cp config.example.json config.local.json
cp .env.example .env
python reproduce.py list
python reproduce.py check closed_loop_exact --model llama31_8b
python reproduce.py run closed_loop_exact --model llama31_8b --full
python reproduce.py run qwen_aligned --full
python reproduce.py run gemma_aligned --full
python reproduce.py run-all --full --dry-run
```

Download the listed checkpoint revisions and fill `models.*.path`. API endpoint, provider selection and model identifiers use the same config; credentials come from the named environment variable. `run-all --full` excludes paid API calls unless `--include-api` is supplied. Dry-run only constructs commands and may stage inputs. It does not validate inference. Model weights and credentials are never included.

Aligned-pair and fixed-state mechanism pipelines require full mode. Behavioral entries expose `--smoke N` for execution checks; smoke output is not a full reproduction. The reported interventions use frozen state inputs. A reviewer is not asked to regenerate or reselect these states from stochastic upstream runs.

After model runs, `python reproduce.py analyze-paper` performs their statistical analysis. Add `--include-content-contract` for Stanza/NLI annotation, after installing `requirements-annotation.txt` and configuring the two annotation resources. Frozen annotations can instead be inspected through the offline evidence interface.

Historical API identifiers and snapshots describe the executed experiment, not a promise of continued service availability or byte-identical future generation. Probe floating-point results can differ across platforms even with pinned library versions; frozen measurements and the original results are provided separately.

## Annotation resources and verification

```bash
pip install -r requirements-annotation.txt
python tools/prepare_annotation_resources.py --output resources --download
python -m unittest discover -s tests -v
```

Copy the printed annotation paths into `config.local.json`. The preparation command downloads Stanza 1.14.0 resources using the bundled resource manifest and the NLI checkpoint at its immutable revision; it verifies the recorded resource hashes. Omit `--download` to verify an existing installation. It does not execute annotation or any paper experiment.

This repository contains reproduction code and required inputs only. Offline reanalysis requires separately supplied evidence. Keep your weights, API keys, downloaded resources and generated outputs outside version control. Dataset source terms and software attributions are listed in `THIRD_PARTY_NOTICES.md`.

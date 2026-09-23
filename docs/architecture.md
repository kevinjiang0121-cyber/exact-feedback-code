# Execution and analysis architecture

`reproduce.py` is the public interface. `registry.Experiment` declares frozen input files, supported models and protocol adapters. Seventeen concrete `ConfiguredExperiment` subclasses own execution plans for the twenty entries. Crossover and aligned-pair classes additionally validate matched inputs and order interventions. `Command` executes each step. `ModelBackend` resolves model configuration; experiment-specific engines retain their original chat templates and inference semantics.

`ArtifactStore` resolves paths within an evidence root and validates SHA-256 and byte counts. `EvidenceAnalysis` builds a dependency graph of `AnalysisJob` objects. Each job runs an existing statistical implementation in a separate process, using a separate output tree. Bootstrap and permutation counts remain analysis-specific. No GPU/API calls are made by this offline interface.

`analyze-paper` is the compatibility entry for newly generated `runs/`. It materializes expected analysis paths under the reproduction package, computes response transfer before downstream recurrence, and includes crossover, aligned pairs, trigger sensitivity and failure classifications. Frozen-evidence outcome assertions are disabled for fresh failure analyses; they do not force a rerun to match historical success rates.

Low-level engines remain separate where tokenizer behavior, model architecture, backend or intervention semantics differ. Users configure paths and credentials once and do not edit these engines.

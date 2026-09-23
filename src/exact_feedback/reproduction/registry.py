from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    adapter: str
    models: tuple[str, ...]
    data: tuple[str, ...] = ()
    notes: str = ""


MAIN_OPEN = (
    "llama31_8b",
    "gemma2_9b",
    "glm4_9b",
    "ministral_8b",
    "qwen3_1_7b",
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "qwen3_32b",
    "qwen3_30b_a3b",
    "granite_3_3_8b",
    "falcon_h1_7b",
)
API_PANEL = (
    "gpt5_6_sol_api",
    "claude_opus5_api",
    "gemini_3_6_flash_api",
    "glm5_2_api",
    "deepseek_v4_flash_api",
    "qwen3_7_plus_api",
    "llama31_70b_api",
)
TULU_PANEL = (
    "llama31_8b_base",
    "tulu3_8b_sft",
    "tulu3_8b_dpo",
    "tulu31_8b_rlvr",
    "llama31_8b",
)
HISTORICAL_EXACT_SIX = (
    "llama31_8b",
    "gemma2_9b",
    "glm4_9b",
    "ministral_8b",
    "qwen3_8b",
    "qwen3_14b",
)


EXPERIMENTS = {
    x.key: x
    for x in (
        Experiment("qwen_aligned", "Qwen probes, restoration, activation and graft discovery",
                   "aligned_pair", ("qwen3_8b_base", "qwen3_8b"),
                   ("data/phase3a_fixed_states_v1/states.jsonl",)),
        Experiment("gemma_aligned", "Gemma probes, restoration, activation and graft discovery",
                   "aligned_pair", ("gemma2_9b_base", "gemma2_9b"),
                   ("data/phase3a_fixed_states_v1/states.jsonl",)),
        Experiment(
            "fixed_draft_crossover", "Fixed-draft source/reviser comparisons",
            "crossover", ("llama31_8b", "qwen3_14b", "gemma2_9b", "glm4_9b"),
            ("data/controller_crossover_combined240_v1/generation_combined240.jsonl",),
        ),
        Experiment(
            "closed_loop_exact",
            "Eight-revision exact-length loop",
            "exact_local",
            MAIN_OPEN,
            (
                "data/human_generation_combined480_v1/generation_combined480.jsonl",
                "data/human_generation_main120_v1/generation_main120.jsonl",
                "data/human_generation_replication120_v1/generation_replication120.jsonl",
                "data/human_generation_extension240_v1/generation_extension240.jsonl",
                "data/human_generation_extension240_v1/shards2/shard_00.jsonl",
                "data/human_generation_extension240_v1/shards2/shard_01.jsonl",
            ),
        ),
        Experiment(
            "closed_loop_structured",
            "Eight-revision lexical/compositional loops",
            "structured_local",
            MAIN_OPEN,
            (
                "data/multidomain_full480_v1/lexical_constraints_combined480.jsonl",
                "data/multidomain_full480_v1/compositional_constraints_combined480.jsonl",
            ),
        ),
        Experiment(
            "closed_loop_api_exact",
            "Exact-length API-controller panel",
            "exact_api",
            API_PANEL,
            ("data/human_generation_combined480_v1/generation_combined480.jsonl",),
        ),
        Experiment(
            "closed_loop_api_structured",
            "Lexical/compositional API-controller panel",
            "structured_api",
            API_PANEL,
            (
                "data/multidomain_full480_v1/lexical_constraints_combined480.jsonl",
                "data/multidomain_full480_v1/compositional_constraints_combined480.jsonl",
            ),
        ),
        Experiment(
            "response_law",
            "Fixed-draft feedback-to-action response surface",
            "response_surface",
            (
                "llama31_8b_base",
                "llama31_8b",
                "qwen3_8b_base",
                "qwen3_8b",
                "qwen3_1_7b",
                "qwen3_4b",
                "qwen3_14b",
                "gemma2_9b",
                "qwen3_32b",
            ),
            ("data/feedback_policy_response_surface_v1/states.jsonl",),
        ),
        Experiment(
            "prompt_robustness",
            "Feedback-prompt robustness",
            "prompt_robustness",
            ("llama31_8b", "gemma2_9b", "qwen3_14b"),
            (
                "data/prompt_robustness60_v1/cases.jsonl",
                "data/prompt_robustness60_v1/templates.json",
            ),
        ),
        Experiment(
            "decoding_robustness",
            "Sampled-decoding robustness",
            "decoding_robustness",
            ("llama31_8b", "gemma2_9b", "qwen3_14b"),
            ("data/prompt_robustness60_v1/cases.jsonl",),
        ),
        Experiment(
            "persistence_32",
            "Revision-32 persistence audit",
            "persistence",
            (
                "llama31_8b",
                "gemma2_9b",
                "glm4_9b",
                "ministral_8b",
                "qwen3_8b",
                "qwen3_14b",
            ),
        ),
        Experiment(
            "history_reset_exact",
            "State-matched exact-length history reset",
            "history_exact",
            ("llama31_8b", "glm4_9b"),
            ("data/history_reset_closed_loop_state_matched_v1/states.jsonl",),
        ),
        Experiment(
            "history_reset_structured",
            "Cross-constraint state-matched history reset",
            "history_structured",
            ("gemma2_9b", "qwen3_14b", "falcon_h1_7b"),
            ("data/targeted_history_reset_confirmation_v1/states.jsonl",),
        ),
        Experiment(
            "history_partial",
            "GLM partial-history intervention",
            "history_partial",
            ("glm4_9b",),
            ("data/glm_partial_reset_matched_pilot_v1/states.jsonl",),
        ),
        Experiment(
            "residual_probe_capture",
            "Aligned-Llama residual-state capture",
            "state_capture",
            ("llama31_8b_base", "llama31_8b"),
            ("data/phase3a_fixed_states_v1/states.jsonl",),
        ),
        Experiment(
            "parameter_restoration",
            "Base-weight restoration in Instruct",
            "materialize_restoration",
            ("llama31_8b_base", "llama31_8b"),
            (
                "data/phase3a_fixed_states_v1/states.jsonl",
                "data/human_generation_main120_v1/generation_main120.jsonl",
                "data/human_generation_combined480_v1/generation_combined480.jsonl",
            ),
        ),
        Experiment(
            "activation_transplant",
            "Aligned-Llama activation transplantation",
            "activation",
            ("llama31_8b_base", "llama31_8b"),
            ("data/phase3a_fixed_states_v1/states.jsonl",),
        ),
        Experiment(
            "controller_graft",
            "Instruct-to-Base controller graft",
            "materialize_graft",
            ("llama31_8b_base", "llama31_8b"),
            (
                "data/phase3a_fixed_states_v1/states.jsonl",
                "data/human_generation_main120_v1/generation_main120.jsonl",
            ),
        ),
        Experiment(
            "tulu_common_interface",
            "Aligned post-training stages under common interface",
            "tulu_common",
            TULU_PANEL,
        ),
        Experiment(
            "tulu_native_interface",
            "Aligned post-training stages under native interface",
            "tulu_native",
            TULU_PANEL[1:],
        ),
    )
}

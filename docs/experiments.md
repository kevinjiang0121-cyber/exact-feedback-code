# Experiment entries

All entries use `python reproduce.py run <id> --full`; select `--model` for entries with multiple independently run models. Paired mechanism pipelines select both checkpoints internally.

| ID | Experiment | Models |
|---|---|---|
| `qwen_aligned` | Qwen probes, restoration, activation and graft discovery | qwen3_8b_base, qwen3_8b |
| `gemma_aligned` | Gemma probes, restoration, activation and graft discovery | gemma2_9b_base, gemma2_9b |
| `fixed_draft_crossover` | Fixed-draft source/reviser comparisons | llama31_8b, qwen3_14b, gemma2_9b, glm4_9b |
| `closed_loop_exact` | Eight-revision exact-length loop | llama31_8b, gemma2_9b, glm4_9b, ministral_8b, qwen3_1_7b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3_32b, qwen3_30b_a3b, granite_3_3_8b, falcon_h1_7b |
| `closed_loop_structured` | Eight-revision lexical/compositional loops | llama31_8b, gemma2_9b, glm4_9b, ministral_8b, qwen3_1_7b, qwen3_4b, qwen3_8b, qwen3_14b, qwen3_32b, qwen3_30b_a3b, granite_3_3_8b, falcon_h1_7b |
| `closed_loop_api_exact` | Exact-length API-controller panel | gpt5_6_sol_api, claude_opus5_api, gemini_3_6_flash_api, glm5_2_api, deepseek_v4_flash_api, qwen3_7_plus_api, llama31_70b_api |
| `closed_loop_api_structured` | Lexical/compositional API-controller panel | gpt5_6_sol_api, claude_opus5_api, gemini_3_6_flash_api, glm5_2_api, deepseek_v4_flash_api, qwen3_7_plus_api, llama31_70b_api |
| `response_law` | Fixed-draft feedback-to-action response surface | llama31_8b_base, llama31_8b, qwen3_8b_base, qwen3_8b, qwen3_1_7b, qwen3_4b, qwen3_14b, gemma2_9b, qwen3_32b |
| `prompt_robustness` | Feedback-prompt robustness | llama31_8b, gemma2_9b, qwen3_14b |
| `decoding_robustness` | Sampled-decoding robustness | llama31_8b, gemma2_9b, qwen3_14b |
| `persistence_32` | Revision-32 persistence audit | llama31_8b, gemma2_9b, glm4_9b, ministral_8b, qwen3_8b, qwen3_14b |
| `history_reset_exact` | State-matched exact-length history reset | llama31_8b, glm4_9b |
| `history_reset_structured` | Cross-constraint state-matched history reset | gemma2_9b, qwen3_14b, falcon_h1_7b |
| `history_partial` | GLM partial-history intervention | glm4_9b |
| `residual_probe_capture` | Aligned-Llama residual-state capture | llama31_8b_base, llama31_8b |
| `parameter_restoration` | Base-weight restoration in Instruct | llama31_8b_base, llama31_8b |
| `activation_transplant` | Aligned-Llama activation transplantation | llama31_8b_base, llama31_8b |
| `controller_graft` | Instruct-to-Base controller graft | llama31_8b_base, llama31_8b |
| `tulu_common_interface` | Aligned post-training stages under common interface | llama31_8b_base, tulu3_8b_sft, tulu3_8b_dpo, tulu31_8b_rlvr, llama31_8b |
| `tulu_native_interface` | Aligned post-training stages under native interface | tulu3_8b_sft, tulu3_8b_dpo, tulu31_8b_rlvr, llama31_8b |

Offline statistics: `python reproduce.py analyze-evidence --materials <evidence> --output <new-directory>`. Use `--only` with the analysis IDs in `experiment_index.json`; prerequisite analyses are included automatically. `--jobs 2` allows independent CPU analyses to run concurrently.

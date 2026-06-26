# Master Thesis LLM Hallucination Detection

Research code for a cross-model hallucination-detection pipeline over TriviaQA-style question answering outputs.

## Pipeline

The main stages are:

1. `cross_prompting.py` generates paraphrases, checks semantic equivalence, and samples cross-prompted answers.
2. `cross_prompting_fixed.py` samples the official Llama/Qwen baseline answers and combines them into `cp_fixed`.
3. `compute_se.py` computes semantic entropy and selects final answers.
4. `verification_labeling.py` runs RAG verification and assigns hallucination labels.
5. `final_pipeline.py` runs the stage scripts in order.

`baseline.py` is kept as an optional standalone baseline utility for research experiments and is not part of `final_pipeline.py`.

## Lightweight Checks

Run the import-light unit tests before committing cleanup changes:

```powershell
python -m pytest tests/test_config.py tests/test_cli_utils.py tests/test_data_utils.py tests/test_model_loading.py tests/test_baseline.py tests/test_cross_prompting.py tests/test_cross_prompting_fixed.py tests/test_compute_se.py tests/test_semantic_entropy.py tests/test_verification_labeling.py tests/test_final_pipeline.py tests/test_llm_with_rag_verifier.py tests/test_utils.py tests/test_prompt_template.py -q
```

Run syntax checks for the pipeline scripts:

```powershell
python -m py_compile baseline.py cli_utils.py compute_se.py config.py data_utils.py model_loading.py cross_prompting.py cross_prompting_fixed.py final_pipeline.py verification_labeling.py utils.py llm_with_rag_verifier.py semantic_entropy.py prompt_template.py
```

The full pipeline may load local LLMs and write large result files, so only run it when you intentionally want the full experiment:

```powershell
python final_pipeline.py
```

# Master Thesis LLM Hallucination Detection

Research code for a cross-model hallucination-detection pipeline over TriviaQA-style question answering outputs.

The project compares baseline model answers, cross-prompted answers, semantic entropy, and RAG-based verification labels to support hallucination analysis in LLM outputs.

## Repository Map

- `src/final_pipeline.py` runs the main experiment stages in order.
- `src/cross_prompting.py` generates and checks paraphrases, then samples cross-prompted answers.
- `src/cross_prompting_fixed.py` creates the official Llama/Qwen baseline and `cp_fixed` answer format.
- `src/compute_se.py` computes semantic entropy and selects final answers.
- `src/verification_labeling.py` runs RAG verification and assigns hallucination labels.
- `src/semantic_entropy.py` contains semantic clustering and entropy logic.
- `src/llm_with_rag_verifier.py` contains retrieval and verifier logic.
- `src/baseline.py` is an optional standalone baseline utility for additional research runs.
- `tests/` contains lightweight tests that do not load real local models.

## Documentation

- [Pipeline overview](docs/pipeline.md)
- [Data contract](docs/data_contract.md)
- [Running experiments](docs/running_experiments.md)

## Pipeline

The main stages are:

1. `src/cross_prompting.py` generates paraphrases, checks semantic equivalence, and samples cross-prompted answers.
2. `src/cross_prompting_fixed.py` samples the official Llama/Qwen baseline answers and combines them into `cp_fixed`.
3. `src/compute_se.py` computes semantic entropy and selects final answers.
4. `src/verification_labeling.py` runs RAG verification and assigns hallucination labels.
5. `src/final_pipeline.py` runs the stage scripts in order.

`src/baseline.py` is kept as an optional standalone baseline utility and is not part of `src/final_pipeline.py`.

## Environment

Create and activate the Conda environment:

```powershell
conda env create -f environment.yml
conda activate thesis-hallucination
```

The environment file uses the CUDA 12.6 PyTorch build that matches the cleaned pipeline environment. For a CPU-only machine, edit `environment.yml` and replace `torch==2.6.0+cu126` with `torch==2.6.0` before creating the environment.

Check the environment:

```powershell
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Lightweight Checks

Run the import-light unit tests:

```powershell
python -m pytest tests -q
```

Run syntax checks for the pipeline scripts:

```powershell
python -m py_compile src/baseline.py src/cli_utils.py src/compute_se.py src/config.py src/data_utils.py src/model_loading.py src/cross_prompting.py src/cross_prompting_fixed.py src/final_pipeline.py src/verification_labeling.py src/utils.py src/llm_with_rag_verifier.py src/semantic_entropy.py src/prompt_template.py
```

## Running The Pipeline

The full pipeline may load local LLMs and write large result files, so only run it when you intentionally want the full experiment:

```powershell
python src/final_pipeline.py
```

Before running, check `src/config.py` and confirm that local model, dataset, evidence, and output paths are correct.

## Local Files Not Committed

The repository intentionally excludes local or large experiment artifacts:

- `datasets/`
- `models/`
- `results/`
- `.vscode/`
- `.pytest_cache/`
- `__pycache__/`
- secrets such as `access_token.txt` and `.env`


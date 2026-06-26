# Master Thesis LLM Hallucination Detection

Research code for a cross-model hallucination-detection pipeline over TriviaQA-style question answering outputs.

The project compares baseline model answers, cross-prompted answers, semantic entropy, and RAG-based verification labels to support hallucination analysis in LLM outputs.

## Repository Map

- `final_pipeline.py` runs the main experiment stages in order.
- `cross_prompting.py` generates and checks paraphrases, then samples cross-prompted answers.
- `cross_prompting_fixed.py` creates the official Llama/Qwen baseline and `cp_fixed` answer format.
- `compute_se.py` computes semantic entropy and selects final answers.
- `verification_labeling.py` runs RAG verification and assigns hallucination labels.
- `semantic_entropy.py` contains semantic clustering and entropy logic.
- `llm_with_rag_verifier.py` contains retrieval and verifier logic.
- `baseline.py` is an optional standalone baseline utility for additional research runs.
- `tests/` contains lightweight tests that do not load real local models.

## Documentation

- [Pipeline overview](docs/pipeline.md)
- [Data contract](docs/data_contract.md)
- [Running experiments](docs/running_experiments.md)

## Pipeline

The main stages are:

1. `cross_prompting.py` generates paraphrases, checks semantic equivalence, and samples cross-prompted answers.
2. `cross_prompting_fixed.py` samples the official Llama/Qwen baseline answers and combines them into `cp_fixed`.
3. `compute_se.py` computes semantic entropy and selects final answers.
4. `verification_labeling.py` runs RAG verification and assigns hallucination labels.
5. `final_pipeline.py` runs the stage scripts in order.

`baseline.py` is kept as an optional standalone baseline utility and is not part of `final_pipeline.py`.

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
python -m py_compile baseline.py cli_utils.py compute_se.py config.py data_utils.py model_loading.py cross_prompting.py cross_prompting_fixed.py final_pipeline.py verification_labeling.py utils.py llm_with_rag_verifier.py semantic_entropy.py prompt_template.py
```

## Running The Pipeline

The full pipeline may load local LLMs and write large result files, so only run it when you intentionally want the full experiment:

```powershell
python final_pipeline.py
```

Before running, check `config.py` and confirm that local model, dataset, evidence, and output paths are correct.

## Local Files Not Committed

The repository intentionally excludes local or large experiment artifacts:

- `datasets/`
- `models/`
- `results/`
- `.vscode/`
- `.pytest_cache/`
- `__pycache__/`
- secrets such as `access_token.txt` and `.env`



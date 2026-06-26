# Master Thesis LLM Hallucination Detection

Research code for a cross-model hallucination-detection pipeline over TriviaQA-style question answering outputs.

The project compares baseline model answers, cross-prompted answers, semantic entropy, and RAG-based verification labels to support hallucination analysis in LLM outputs.

## Repository Map

- `src/hallucination_detection/pipeline.py` runs the main experiment stages in order.
- `src/hallucination_detection/stages/cross_prompting.py` generates and checks paraphrases, then samples cross-prompted answers.
- `src/hallucination_detection/stages/cross_prompting_fixed.py` creates the official Llama/Qwen baseline and `cp_fixed` answer format.
- `src/hallucination_detection/stages/compute_se.py` computes semantic entropy and selects final answers.
- `src/hallucination_detection/stages/verification_labeling.py` runs RAG verification and assigns hallucination labels.
- `src/hallucination_detection/entropy/semantic_entropy.py` contains semantic clustering and entropy logic.
- `src/hallucination_detection/verification/rag_verifier.py` contains retrieval and verifier logic.
- `src/hallucination_detection/stages/baseline.py` is an optional standalone baseline utility for additional research runs.

## Documentation

- [Pipeline overview](docs/pipeline.md)
- [Data contract](docs/data_contract.md)
- [Running experiments](docs/running_experiments.md)

## Pipeline

The main stages are:

1. `hallucination_detection.stages.cross_prompting` generates paraphrases, checks semantic equivalence, and samples cross-prompted answers.
2. `hallucination_detection.stages.cross_prompting_fixed` samples the official Llama/Qwen baseline answers and combines them into `cp_fixed`.
3. `hallucination_detection.stages.compute_se` computes semantic entropy and selects final answers.
4. `hallucination_detection.stages.verification_labeling` runs RAG verification and assigns hallucination labels.
5. `hallucination_detection.pipeline` runs the stage modules in order.

`hallucination_detection.stages.baseline` is kept as an optional standalone baseline utility and is not part of `hallucination_detection.pipeline`.

## Environment

Create and activate the Conda environment:

```powershell
conda env create -f environment.yml
conda activate thesis-hallucination
python -m pip install -e .
```

The environment file uses the CUDA 12.6 PyTorch build that matches the cleaned pipeline environment. For a CPU-only machine, edit `environment.yml` and replace `torch==2.6.0+cu126` with `torch==2.6.0` before creating the environment.

Check the environment:

```powershell
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Lightweight Checks

Run syntax checks for the package modules:

```powershell
python -m py_compile src/hallucination_detection/config.py src/hallucination_detection/cli_utils.py src/hallucination_detection/data_utils.py src/hallucination_detection/model_loading.py src/hallucination_detection/prompts.py src/hallucination_detection/generation.py src/hallucination_detection/pipeline.py src/hallucination_detection/stages/baseline.py src/hallucination_detection/stages/cross_prompting.py src/hallucination_detection/stages/cross_prompting_fixed.py src/hallucination_detection/stages/compute_se.py src/hallucination_detection/stages/verification_labeling.py src/hallucination_detection/entropy/semantic_entropy.py src/hallucination_detection/verification/rag_verifier.py
```

## Running The Pipeline

The full pipeline may load local LLMs and write large result files, so only run it when you intentionally want the full experiment:

```powershell
python -m hallucination_detection.pipeline
```

Before running, check `src/hallucination_detection/config.py` and confirm that local model, dataset, evidence, and output paths are correct.

## Local Files Not Committed

The repository intentionally excludes local or large experiment artifacts:

- `datasets/`
- `models/`
- `results/`
- `.vscode/`
- `.pytest_cache/`
- `__pycache__/`
- secrets such as `access_token.txt` and `.env`

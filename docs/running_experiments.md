# Running Experiments

This file explains how to set up the environment, run lightweight checks, and run the full experiment pipeline.

## 1. Create The Environment

Create the Conda environment:

```powershell
conda env create -f environment.yml
```

Activate it:

```powershell
conda activate thesis-hallucination
```

Comment:
The provided `environment.yml` uses the CUDA 12.6 PyTorch build. For CPU-only machines, edit the Torch line in `environment.yml` before creating the environment.

## 2. Confirm Python And Torch

```powershell
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected:
- Python 3.12
- Torch import succeeds
- CUDA availability depends on the machine

## 3. Run Lightweight Tests

Run this before committing or archiving:

```powershell
python -m pytest tests -q
```

These tests use fake loaders, fake model functions, and pure helper seams. They should not load the full LLM pipeline.

## 4. Run Syntax Checks

```powershell
python -m py_compile src/baseline.py src/cli_utils.py src/compute_se.py src/config.py src/data_utils.py src/model_loading.py src/cross_prompting.py src/cross_prompting_fixed.py src/final_pipeline.py src/verification_labeling.py src/utils.py src/llm_with_rag_verifier.py src/semantic_entropy.py src/prompt_template.py
```

## 5. Check Runtime Paths

Before running the full pipeline, inspect `src/config.py`.

Important paths:
- `LLM_LLAMA_PATH`
- `LLM_QWEN_PATH`
- `TRIVIAQA_PATH`
- `EVIDENCE_PATH`
- `RETRIEVER_PATH`
- output JSON paths

Comment:
The repository does not commit local models, datasets, evidence files, or generated results.

## 6. Run Individual Stages

Use individual stages when debugging or reproducing part of the experiment:

```powershell
python src/cross_prompting.py
python src/cross_prompting_fixed.py
python src/compute_se.py
python src/verification_labeling.py
```

## 7. Run The Full Pipeline

Only run the full pipeline when local models and datasets are available:

```powershell
python src/final_pipeline.py
```

Warning:
This may be slow and may write large JSON result files.

## 8. Update Existing Environment

If `environment.yml` changes later:

```powershell
conda env update -f environment.yml --prune
```

## 9. Remove And Recreate Environment

```powershell
conda env remove -n thesis-hallucination
conda env create -f environment.yml
conda activate thesis-hallucination
```

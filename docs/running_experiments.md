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
python -m pip install -e .
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

## 3. Run Syntax Checks

```powershell
python -m py_compile src/hallucination_detection/config.py src/hallucination_detection/cli_utils.py src/hallucination_detection/data_utils.py src/hallucination_detection/model_loading.py src/hallucination_detection/prompts.py src/hallucination_detection/generation.py src/hallucination_detection/pipeline.py src/hallucination_detection/stages/baseline.py src/hallucination_detection/stages/cross_prompting.py src/hallucination_detection/stages/cross_prompting_fixed.py src/hallucination_detection/stages/compute_se.py src/hallucination_detection/stages/verification_labeling.py src/hallucination_detection/entropy/semantic_entropy.py src/hallucination_detection/verification/rag_verifier.py
```

## 4. Check Runtime Paths

Before running the full pipeline, inspect `src/hallucination_detection/config.py`.

Important paths:
- `LLM_LLAMA_PATH`
- `LLM_QWEN_PATH`
- `TRIVIAQA_PATH`
- `EVIDENCE_PATH`
- `RETRIEVER_PATH`
- output JSON paths

Comment:
The repository does not commit local models, datasets, evidence files, or generated results.

## 5. Run Individual Stages

Use individual stages when debugging or reproducing part of the experiment:

```powershell
python -m hallucination_detection.stages.cross_prompting
python -m hallucination_detection.stages.cross_prompting_fixed
python -m hallucination_detection.stages.compute_se
python -m hallucination_detection.stages.verification_labeling
```

## 6. Run The Full Pipeline

Only run the full pipeline when local models and datasets are available:

```powershell
python -m hallucination_detection.pipeline
```

Warning:
This may be slow and may write large JSON result files.

## 7. Update Existing Environment

If `environment.yml` changes later:

```powershell
conda env update -f environment.yml --prune
```

## 8. Remove And Recreate Environment

```powershell
conda env remove -n thesis-hallucination
conda env create -f environment.yml
conda activate thesis-hallucination
python -m pip install -e .
```

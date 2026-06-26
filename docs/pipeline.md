# Pipeline Overview

This file explains the main experiment flow. It is meant to help a reader understand the order of scripts without reading every implementation file first.

## Stage Order

The standard pipeline runs in this order:

```text
cross_prompting.py
cross_prompting_fixed.py
compute_se.py
verification_labeling.py
```

`final_pipeline.py` runs these stages in sequence.

## Stage Responsibilities

### 1. Cross-Prompting

Script: `cross_prompting.py`

Purpose:
- Generate paraphrases for verified TriviaQA questions.
- Cross-check paraphrase meaning between Llama and Qwen.
- Ask each model questions generated or validated by the other model.
- Save cross-prompting answer collections.

Important output fields:
- `question_id`
- `original`
- `paraphrases`
- `paraphrases_generated_llama`
- `paraphrases_generated_qwen`
- `answers_responsed_llama`
- `answers_responsed_qwen`

### 2. Fixed Baseline / `cp_fixed`

Script: `cross_prompting_fixed.py`

Purpose:
- Sample standalone baseline answers from Llama and Qwen.
- Match Llama and Qwen records by normalized `question_id`.
- Combine both answer sets into the official `cp_fixed` format.

Comment:
`baseline.py` is kept as an optional research utility. The official baseline stage used by the pipeline is `cross_prompting_fixed.py`.

### 3. Semantic Entropy

Script: `compute_se.py`

Purpose:
- Load generated answer collections.
- Compute semantic clusters and semantic entropy.
- Select a final answer from each answer collection.

Important output fields:
- `semantic_ids`
- `semantic_entropy`
- `final_answer`
- `selected_model`

### 4. RAG Verification And Labeling

Script: `verification_labeling.py`

Purpose:
- Load semantic-entropy outputs.
- Run retrieval-augmented verification against TriviaQA evidence.
- Assign final hallucination labels.

Important output fields:
- `gold_correct`
- `rag_verification`
- `retrieved_chunks`
- `is_hallucinating_true`

## Full Pipeline Command

Run only when local models, datasets, and evidence files are available:

```powershell
python final_pipeline.py
```

The full pipeline can be expensive because it may load local LLMs and write large result files.

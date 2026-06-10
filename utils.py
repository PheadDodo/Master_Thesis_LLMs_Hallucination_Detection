import os
import re
import warnings
import json
import numpy as np
import logging
import unicodedata
from unidecode import unidecode
from ftfy import fix_text
from collections import Counter

import torch
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm


def load_llm(model_path, load_in_8bit=True, dtype=torch.float16, device_map="auto"):
    is_cpu_only = device_map == "cpu"

    if is_cpu_only and load_in_8bit:
        warnings.warn("8-bit quantization requires CUDA; switching load_in_8bit to False for CPU loading.")
        load_in_8bit = False

    if is_cpu_only and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    bnb_config = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )

    pipeline_kwargs = {}
    if is_cpu_only:
        pipeline_kwargs["device"] = -1

    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs
    )
    
    print(f"Loaded model from {model_path} with dtype={dtype}, load_in_8bit={load_in_8bit}, device_map={device_map}")

    return llm, tokenizer, model


# def unload_llm(llm, tokenizer, model):
#     import gc
#     del llm, tokenizer, model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()
#     print("Model unloaded and memory freed.")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} entries to {output_path}")
    


def paraphrase_question(llm, tokenizer, original_q, prompt_template,
                        n=5, max_new_tokens=96, max_tries=3, temperature=0.8,
                        top_p=0.95, do_sample=True):
    paraphrases = []
    seen = {original_q.strip().lower()}

    def is_question_like(text):
        text = text.strip()
        if not text:
            return False

        bad_patterns = [
            r"\banonymous\b",
            r"\bnot verified\b",
            r"\bthu,\b|\bmon,\b|\btue,\b|\bwed,\b|\bfri,\b|\bsat,\b|\bsun,\b",
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        ]
        lowered = text.lower()
        if any(re.search(p, lowered) for p in bad_patterns):
            return False

        if "?" in text:
            return True

        return bool(re.match(r"^(who|what|when|where|why|how|which|whom|whose)\b", lowered))

    for _ in range(max_tries):
        prompt = prompt_template.format(q=original_q, n=n)

        out = llm(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        text = out[0]["generated_text"].strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line in lines:
            cleaned = re.sub(r"^\d+[\).:-]?\s*", "", line).strip()
            if cleaned.lower() in {"paraphrases:", "paraphrase:"}:
                continue
            if not is_question_like(cleaned):
                continue

            key = cleaned.lower()
            if cleaned and key not in seen:
                paraphrases.append(cleaned)
                seen.add(key)

            if len(paraphrases) >= n:
                return paraphrases[:n]

    return paraphrases[:n]


def ask(llm, tokenizer, paraphrases, prompt_template, max_new_tokens=8, temperature=0.8,
                        top_p=0.95, do_sample=True):
    results = []

    for paraphrase in paraphrases:
        prompt = prompt_template.format(q=paraphrase)

        out = llm(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        text = out[0]["generated_text"].strip()
        if not text:
            results.append("")
            continue

        ans = text.splitlines()[0].strip()
        ans = re.sub(r"^(answer\s*:)\s*", "", ans, flags=re.I)
        results.append(ans)

    return results


def check_meaning(llm, tokenizer, original_q, paraphrases, prompt_template, max_new_tokens=3,
                  temperature=0.8, top_p=0.95, do_sample=True):
    results = []

    for paraphrase in paraphrases:
        prompt = prompt_template.format(q0=original_q, q1=paraphrase)

        out = llm(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        text = out[0]["generated_text"].strip()

        verdict_match = re.search(r"verdict\s*:\s*(yes|no)", text, flags=re.I)
        if verdict_match:
            cleaned = verdict_match.group(1).strip()
        else:
            first_line = text.splitlines()[0].strip() if text else ""
            cleaned = re.sub(r"^(answer\s*:)\s*", "", first_line, flags=re.I).strip()

        cleaned = cleaned.rstrip(".?!,;:\"' ")
        results.append(cleaned)

    return results





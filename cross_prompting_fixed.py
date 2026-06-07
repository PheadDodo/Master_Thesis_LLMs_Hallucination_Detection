import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import torch

import config

from utils import load_llm
from utils import load_json
from utils import save_results
from utils import ask
from prompt_template import output_prompt, output_prompt_no_idk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline: sample a single LLM n times per original question"
    )

    parser.add_argument("--llm_llama_path", type=str, default=config.LLM_LLAMA_PATH)
    parser.add_argument("--llm_qwen_path", type=str, default=config.LLM_QWEN_PATH)
    parser.add_argument("--device_map", type=str, default=config.DEVICE_MAP)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=config.TOP_P)
    parser.add_argument("--n", type=int, default=config.N)
    parser.add_argument("--output_path_llama", type=str, default=config.OUTPUT_PATH_LLAMA)
    parser.add_argument("--output_path_qwen", type=str, default=config.OUTPUT_PATH_QWEN)
    parser.add_argument("--output_path_cp_fixed", type=str, default=config.OUTPUT_PATH_CP_FIXED)

    parser.add_argument("--load_in_8bit", dest="load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=config.LOAD_IN_8BIT)

    return parser.parse_args()

def build_cross_prompting_fixed(llama_data, qwen_data):
    cp_fixed = []

    for ex_llama, ex_qwen in zip(llama_data, qwen_data):
        qid = ex_llama["question_id"]
        original_q = ex_llama["original"]
        answers_llama = ex_llama["answers"]
        answers_qwen = ex_qwen["answers"]

        # find corresponding qwen entry
        qwen_entry = next((e for e in qwen_data if e["question_id"] == qid), None)
        if not qwen_entry:
            continue

        cp_entry = {
            "question_id": qid,
            "original": original_q,
            "answers_responsed_llama": answers_llama,
            "answers_responsed_qwen": answers_qwen,
            "answers": answers_llama + answers_qwen,
        }
        cp_fixed.append(cp_entry)

    return cp_fixed


def main():
    args = parse_args()

    # load the single LLM (Llama-3.1-8B-Instruct) for the baseline method
    llm_llama, tokenizer_llama, _ = load_llm(
        args.llm_llama_path,
        load_in_8bit=args.load_in_8bit,
        dtype=torch.float16,
        device_map=args.device_map,
    )
    
    # load the single LLM (Qwen3-8B) for the baseline method
    llm_qwen, tokenizer_qwen, _ = load_llm(
        args.llm_qwen_path,
        load_in_8bit=False,
        dtype=torch.float16,
        device_map="cpu",
    )


    # load TriviaQA verified subset
    data = load_json(config.TRIVIAQA_PATH)
    data = [ex for ex in data["Data"]]
    print(f"Loaded {len(data)} verified question-answer pairs")

    start_time = time.time()

    baseline_results_llama = []
    baseline_results_qwen = []
    for ex in data[:]:  # number of questions to sample for the baseline method
        original_q = ex["Question"]
        question_id = ex["QuestionId"]

        n = args.n # sampling n answers per question for the baseline method

        # ask the single LLM the original question n times
        answers_llama = ask(
            llm=llm_llama,
            tokenizer=tokenizer_llama,
            paraphrases=[original_q] * n,
            prompt_template=output_prompt_no_idk, # try to avoid generating "I don't know" answers for the baseline method
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        answers_qwen = ask(
            llm=llm_qwen,
            tokenizer=tokenizer_qwen,
            paraphrases=[original_q] * n,
            prompt_template=output_prompt_no_idk, # try to avoid generating "I don't know" answers for the baseline method
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        baseline_results_llama.append({
            "question_id": question_id,
            "original": original_q,
            "answers": answers_llama,
        })

        baseline_results_qwen.append({
            "question_id": question_id,
            "original": original_q,
            "answers": answers_qwen,
        })

        print(
            f"[{len(baseline_results_llama)}/{len(data)}] QID={question_id} | n={n} | answers_llama={answers_llama} | answers_qwen={answers_qwen}"
        )
    
    cp_fixed = build_cross_prompting_fixed(baseline_results_llama, baseline_results_qwen)

    save_results(cp_fixed, args.output_path_cp_fixed)
    save_results(baseline_results_llama, args.output_path_llama)
    save_results(baseline_results_qwen, args.output_path_qwen)

    elapsed = time.time() - start_time
    print(f"Cross-prompting fixed completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()




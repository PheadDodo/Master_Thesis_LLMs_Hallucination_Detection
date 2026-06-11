import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time

import config
from data_utils import load_json
from data_utils import save_results
from data_utils import select_verified_examples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Baseline: sample a single LLM n times per original question"
    )

    parser.add_argument("--llm_llama_path", type=str, default=config.LLM_LLAMA_PATH)
    parser.add_argument("--llm_qwen_path", type=str, default=config.LLM_QWEN_PATH)
    parser.add_argument("--device_map", type=str, default=config.DEVICE_MAP)
    parser.add_argument("--qwen_device_map", type=str, default=config.QWEN_DEVICE_MAP)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=config.TOP_P)
    parser.add_argument("--n", type=int, default=config.N)
    parser.add_argument("--output_path_llama", type=str, default=config.OUTPUT_PATH_LLAMA)
    parser.add_argument("--output_path_qwen", type=str, default=config.OUTPUT_PATH_QWEN)
    parser.add_argument("--output_path_cp_fixed", type=str, default=config.OUTPUT_PATH_CP_FIXED)

    parser.add_argument("--load_in_8bit", dest="load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=config.LOAD_IN_8BIT)

    return parser.parse_args(argv)


def build_cross_prompting_fixed(llama_data, qwen_data):
    cp_fixed = []
    qwen_by_id = {ex["question_id"]: ex for ex in qwen_data}

    for ex_llama in llama_data:
        qid = ex_llama["question_id"]
        original_q = ex_llama["original"]
        answers_llama = ex_llama["answers"]

        qwen_entry = qwen_by_id.get(qid)
        if not qwen_entry:
            continue
        answers_qwen = qwen_entry["answers"]

        cp_entry = {
            "question_id": qid,
            "original": original_q,
            "answers_responsed_llama": answers_llama,
            "answers_responsed_qwen": answers_qwen,
            "answers": answers_llama + answers_qwen,
        }
        cp_fixed.append(cp_entry)

    return cp_fixed


def sample_baseline_answers(
    data,
    llm,
    tokenizer,
    ask_fn,
    prompt_template,
    n,
    temperature,
    top_p,
):
    baseline_results = []

    for ex in data:
        original_q = ex["Question"]
        question_id = ex["QuestionId"]
        answers = ask_fn(
            llm=llm,
            tokenizer=tokenizer,
            paraphrases=[original_q] * n,
            prompt_template=prompt_template,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        baseline_results.append({
            "question_id": question_id,
            "original": original_q,
            "answers": answers,
        })

    return baseline_results


def format_cross_prompting_fixed_progress(
    index,
    total,
    n,
    baseline_llama,
    baseline_qwen,
):
    return (
        f"[{index}/{total}] QID={baseline_llama['question_id']} | n={n} | "
        f"answers_llama={baseline_llama['answers']} | "
        f"answers_qwen={baseline_qwen['answers']}"
    )


def sample_cross_prompting_fixed_outputs(
    data,
    llm_llama,
    tokenizer_llama,
    llm_qwen,
    tokenizer_qwen,
    ask_fn,
    prompt_template,
    n,
    temperature,
    top_p,
    progress_fn=None,
):
    baseline_results_llama = []
    baseline_results_qwen = []
    total = len(data)

    for index, ex in enumerate(data, start=1):
        baseline_llama = sample_baseline_answers(
            data=[ex],
            llm=llm_llama,
            tokenizer=tokenizer_llama,
            ask_fn=ask_fn,
            prompt_template=prompt_template,
            n=n,
            temperature=temperature,
            top_p=top_p,
        )[0]
        baseline_qwen = sample_baseline_answers(
            data=[ex],
            llm=llm_qwen,
            tokenizer=tokenizer_qwen,
            ask_fn=ask_fn,
            prompt_template=prompt_template,
            n=n,
            temperature=temperature,
            top_p=top_p,
        )[0]

        baseline_results_llama.append(baseline_llama)
        baseline_results_qwen.append(baseline_qwen)

        if progress_fn is not None:
            progress_fn(index, total, baseline_llama, baseline_qwen)

    cp_fixed = build_cross_prompting_fixed(
        baseline_results_llama,
        baseline_results_qwen,
    )

    return cp_fixed, baseline_results_llama, baseline_results_qwen


def main():
    import torch

    from prompt_template import output_prompt
    from utils import ask
    from utils import load_llm

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
        device_map=args.qwen_device_map,
    )


    # load TriviaQA verified subset
    sample = load_json(config.TRIVIAQA_PATH)
    data = select_verified_examples(sample)
    print(f"Loaded {len(data)} verified question-answer pairs")

    start_time = time.time()

    def print_progress(index, total, baseline_llama, baseline_qwen):
        print(format_cross_prompting_fixed_progress(
            index,
            total,
            args.n,
            baseline_llama,
            baseline_qwen,
        ))

    cp_fixed, baseline_results_llama, baseline_results_qwen = sample_cross_prompting_fixed_outputs(
        data=data,
        llm_llama=llm_llama,
        tokenizer_llama=tokenizer_llama,
        llm_qwen=llm_qwen,
        tokenizer_qwen=tokenizer_qwen,
        ask_fn=ask,
        prompt_template=output_prompt,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        progress_fn=print_progress,
    )

    save_results(cp_fixed, args.output_path_cp_fixed)
    save_results(baseline_results_llama, args.output_path_llama)
    save_results(baseline_results_qwen, args.output_path_qwen)

    elapsed = time.time() - start_time
    print(f"Cross-prompting fixed completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()




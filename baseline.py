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
        description="Optional research baseline: sample Llama and Qwen on original questions"
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

    parser.add_argument("--load_in_8bit", dest="load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=config.LOAD_IN_8BIT)

    return parser.parse_args(argv)


def select_baseline_examples(sample):
    return select_verified_examples(sample)


def sample_baseline_pair(
    ex,
    n,
    llm_llama,
    tokenizer_llama,
    llm_qwen,
    tokenizer_qwen,
    ask_fn,
    prompt_template,
    temperature,
    top_p,
):
    original_q = ex["Question"]
    question_id = ex["QuestionId"]

    answers_llama = ask_fn(
        llm=llm_llama,
        tokenizer=tokenizer_llama,
        paraphrases=[original_q] * n,
        prompt_template=prompt_template,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    answers_qwen = ask_fn(
        llm=llm_qwen,
        tokenizer=tokenizer_qwen,
        paraphrases=[original_q] * n,
        prompt_template=prompt_template,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    return (
        {
            "question_id": question_id,
            "original": original_q,
            "answers": answers_llama,
        },
        {
            "question_id": question_id,
            "original": original_q,
            "answers": answers_qwen,
        },
    )


def format_baseline_progress(index, total, n, baseline_llama, baseline_qwen):
    return (
        f"[{index}/{total}] QID={baseline_llama['question_id']} | n={n} | "
        f"answers_llama={baseline_llama['answers']} | "
        f"answers_qwen={baseline_qwen['answers']}"
    )


def progress_args_from_baseline_records(
    index,
    total,
    baseline_llama,
    baseline_qwen,
):
    return index, total, baseline_llama, baseline_qwen


def run_baseline_records(
    data,
    n,
    llm_llama,
    tokenizer_llama,
    llm_qwen,
    tokenizer_qwen,
    ask_fn,
    prompt_template,
    temperature,
    top_p,
    progress_fn=None,
):
    baseline_results_llama = []
    baseline_results_qwen = []
    total = len(data)

    for index, ex in enumerate(data, start=1):
        baseline_llama, baseline_qwen = sample_baseline_pair(
            ex=ex,
            n=n,
            llm_llama=llm_llama,
            tokenizer_llama=tokenizer_llama,
            llm_qwen=llm_qwen,
            tokenizer_qwen=tokenizer_qwen,
            ask_fn=ask_fn,
            prompt_template=prompt_template,
            temperature=temperature,
            top_p=top_p,
        )

        baseline_results_llama.append(baseline_llama)
        baseline_results_qwen.append(baseline_qwen)

        if progress_fn is not None:
            progress_fn(*progress_args_from_baseline_records(
                index,
                total,
                baseline_llama,
                baseline_qwen,
            ))

    return baseline_results_llama, baseline_results_qwen


def save_baseline_results(llama_results, qwen_results, args, save_fn=save_results):
    save_fn(llama_results, args.output_path_llama)
    save_fn(qwen_results, args.output_path_qwen)


def main():
    import torch

    from prompt_template import output_prompt
    from utils import ask
    from utils import load_llm

    args = parse_args()

    llm_llama, tokenizer_llama, _ = load_llm(
        args.llm_llama_path,
        load_in_8bit=args.load_in_8bit,
        dtype=torch.float16,
        device_map=args.device_map,
    )

    llm_qwen, tokenizer_qwen, _ = load_llm(
        args.llm_qwen_path,
        load_in_8bit=False,
        dtype=torch.float16,
        device_map=args.qwen_device_map,
    )

    sample = load_json(config.TRIVIAQA_PATH)
    data = select_baseline_examples(sample)
    print(f"Loaded {len(data)} verified question-answer pairs")

    start_time = time.time()

    def print_progress(index, total, baseline_llama, baseline_qwen):
        print(format_baseline_progress(
            index,
            total,
            args.n,
            baseline_llama,
            baseline_qwen,
        ))

    baseline_results_llama, baseline_results_qwen = run_baseline_records(
        data=data,
        n=args.n,
        llm_llama=llm_llama,
        tokenizer_llama=tokenizer_llama,
        llm_qwen=llm_qwen,
        tokenizer_qwen=tokenizer_qwen,
        ask_fn=ask,
        prompt_template=output_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        progress_fn=print_progress,
    )

    save_baseline_results(baseline_results_llama, baseline_results_qwen, args)

    elapsed = time.time() - start_time
    print(f"Optional baseline completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()

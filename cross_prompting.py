import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time

import config
from data_utils import append_valid_paraphrases
from data_utils import load_json
from data_utils import save_results
from data_utils import select_verified_examples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cross prompting: generate paraphrases, cross-check meaning, and obtain answers"
    )

    parser.add_argument("--llm_llama_path", type=str, default=config.LLM_LLAMA_PATH)
    parser.add_argument("--llm_qwen_path", type=str, default=config.LLM_QWEN_PATH)
    parser.add_argument("--device_map", type=str, default=config.DEVICE_MAP)
    parser.add_argument("--qwen_device_map", type=str, default=config.QWEN_DEVICE_MAP)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=config.TOP_P)
    parser.add_argument("--n", type=int, default=config.N)
    parser.add_argument("--output_path_cp", type=str, default=config.OUTPUT_PATH_CP)

    parser.add_argument("--load_in_8bit", dest="load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=config.LOAD_IN_8BIT)

    return parser.parse_args(argv)


def questions_for_answering(original_q, valid_paraphrases):
    return valid_paraphrases if valid_paraphrases else [original_q]


def build_cross_prompting_record(
    original_q,
    question_id,
    clean_paraphrases_qwen,
    clean_paraphrases_llama,
    answers_llama,
    answers_qwen,
):
    return {
        "question_id": question_id,
        "original": original_q,
        "paraphrases_generated_llama": clean_paraphrases_llama,
        "paraphrases_generated_qwen": clean_paraphrases_qwen,
        "paraphrases": clean_paraphrases_qwen + clean_paraphrases_llama,
        "answers_responsed_llama": answers_llama,
        "answers_responsed_qwen": answers_qwen,
        "answers": answers_llama + answers_qwen,
    }


def format_cross_prompting_progress(
    index,
    total,
    question_id,
    n,
    answers_llama,
    answers_qwen,
):
    return (
        f"[{index}/{total}] QID={question_id} | n={n} | "
        f"answers_llama={answers_llama} | answers_qwen={answers_qwen}"
    )


def run_cross_prompting_records(
    data,
    llm_llama,
    tokenizer_llama,
    llm_qwen,
    tokenizer_qwen,
    paraphrase_question_fn,
    check_meaning_fn,
    ask_fn,
    paraphrase_prompt,
    output_prompt,
    paraphrases_check_prompt,
    n,
    temperature,
    top_p,
    progress_fn=None,
):
    cleaned_results = []
    total = len(data)

    for index, ex in enumerate(data, start=1):
        original_q = ex["Question"]
        question_id = ex["QuestionId"]

        generated_paraphrases_llama = paraphrase_question_fn(
            llm=llm_llama,
            tokenizer=tokenizer_llama,
            original_q=original_q,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            n=n,
            prompt_template=paraphrase_prompt,
        )
        generated_paraphrases_qwen = paraphrase_question_fn(
            llm=llm_qwen,
            tokenizer=tokenizer_qwen,
            original_q=original_q,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            n=n,
            prompt_template=paraphrase_prompt,
        )

        cross_check_llama = check_meaning_fn(
            llm=llm_llama,
            tokenizer=tokenizer_llama,
            original_q=original_q,
            paraphrases=generated_paraphrases_qwen,
            prompt_template=paraphrases_check_prompt,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        cross_check_qwen = check_meaning_fn(
            llm=llm_qwen,
            tokenizer=tokenizer_qwen,
            original_q=original_q,
            paraphrases=generated_paraphrases_llama,
            prompt_template=paraphrases_check_prompt,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        cleaned_cross_llama = append_valid_paraphrases(
            ex,
            generated_paraphrases_qwen,
            cross_check_llama,
        )
        cleaned_cross_qwen = append_valid_paraphrases(
            ex,
            generated_paraphrases_llama,
            cross_check_qwen,
        )
        clean_paraphrases_qwen = cleaned_cross_llama["paraphrases"][0]
        clean_paraphrases_llama = cleaned_cross_qwen["paraphrases"][0]

        cross_ask_llama = ask_fn(
            llm=llm_llama,
            tokenizer=tokenizer_llama,
            paraphrases=questions_for_answering(original_q, clean_paraphrases_qwen),
            prompt_template=output_prompt,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        cross_ask_qwen = ask_fn(
            llm=llm_qwen,
            tokenizer=tokenizer_qwen,
            paraphrases=questions_for_answering(original_q, clean_paraphrases_llama),
            prompt_template=output_prompt,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        record = build_cross_prompting_record(
            original_q=original_q,
            question_id=question_id,
            clean_paraphrases_qwen=clean_paraphrases_qwen,
            clean_paraphrases_llama=clean_paraphrases_llama,
            answers_llama=cross_ask_llama,
            answers_qwen=cross_ask_qwen,
        )
        cleaned_results.append(record)

        if progress_fn is not None:
            progress_fn(index, total, question_id, cross_ask_llama, cross_ask_qwen)

    return cleaned_results


def save_cross_prompting_results(records, args, save_fn=save_results):
    save_fn(records, args.output_path_cp)


def main():
    import torch

    from prompt_template import paraphrase_prompt, output_prompt, paraphrases_check
    from utils import ask
    from utils import check_meaning
    from utils import load_llm
    from utils import paraphrase_question

    args = parse_args()
    llm_path_llama = args.llm_llama_path
    llm_path_qwen = args.llm_qwen_path
    
    # load Llama-3.1-8B-Instruct
    llm_llama, tokenizer_llama, _ = load_llm(
        llm_path_llama,
        load_in_8bit=args.load_in_8bit,
        dtype=torch.float16,
        device_map=args.device_map
        )
    
    # load Qwen3-8B
    llm_qwen, tokenizer_qwen, _ = load_llm(
        llm_path_qwen,
        device_map=args.qwen_device_map,
        load_in_8bit=False,
        dtype=torch.float16
    )
    
    # load sample data, a verified subset of TriviaQA_rc verified-wikipedia-dev.json, 
    # which is verified by human annotators to have correct question-answer pairs
    sample = load_json(config.TRIVIAQA_PATH)
    data_verified = select_verified_examples(sample)

    print(f"Loaded {len(data_verified)} verified question-answer pairs from TriviaQA_rc_verified-wikipedia-dev.json")
    
    # run cross prompting for each question-answer pair
    start_time = time.time()
    
    def print_progress(index, total, question_id, answers_llama, answers_qwen):
        print(format_cross_prompting_progress(
            index,
            total,
            question_id,
            args.n,
            answers_llama,
            answers_qwen,
        ))

    cleaned_results = run_cross_prompting_records(
        data=data_verified,
        llm_llama=llm_llama,
        tokenizer_llama=tokenizer_llama,
        llm_qwen=llm_qwen,
        tokenizer_qwen=tokenizer_qwen,
        paraphrase_question_fn=paraphrase_question,
        check_meaning_fn=check_meaning,
        ask_fn=ask,
        paraphrase_prompt=paraphrase_prompt,
        output_prompt=output_prompt,
        paraphrases_check_prompt=paraphrases_check,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        progress_fn=print_progress,
    )
    
    #save results, do_sample=True, temperature=0.5, top_p=0.95
    save_cross_prompting_results(cleaned_results, args)
    
    print("Cross prompting completed.")
    elapsed = time.time() - start_time
    print(f"Time elapsed: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    
if __name__ == "__main__":
    main()



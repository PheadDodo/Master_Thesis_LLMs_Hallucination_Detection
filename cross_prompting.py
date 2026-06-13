import argparse
import time

from cli_utils import add_generation_args
import config
from data_utils import append_valid_paraphrases
from data_utils import load_json
from data_utils import save_results
from data_utils import select_verified_examples
from model_loading import add_model_loading_args
from model_loading import load_llama_qwen_models


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cross prompting: generate paraphrases, cross-check meaning, and obtain answers"
    )

    add_model_loading_args(parser, config)
    add_generation_args(parser, config)
    parser.add_argument("--output_path_cp", type=str, default=config.OUTPUT_PATH_CP)

    return parser.parse_args(argv)


def questions_for_answering(original_q, valid_paraphrases):
    return valid_paraphrases if valid_paraphrases else [original_q]


def load_cross_prompting_examples(
    triviaqa_path,
    load_fn=load_json,
    select_fn=select_verified_examples,
):
    sample = load_fn(triviaqa_path)
    return select_fn(sample)


def load_cross_prompting_models(args, load_llm_fn, dtype, qwen_load_in_8bit=None):
    return load_llama_qwen_models(
        args,
        load_llm_fn=load_llm_fn,
        dtype=dtype,
        qwen_load_in_8bit=qwen_load_in_8bit,
    )


def clean_cross_checked_paraphrases(
    ex,
    generated_paraphrases_qwen,
    cross_check_llama,
    generated_paraphrases_llama,
    cross_check_qwen,
    append_fn=append_valid_paraphrases,
):
    cleaned_cross_llama = append_fn(
        ex,
        generated_paraphrases_qwen,
        cross_check_llama,
    )
    cleaned_cross_qwen = append_fn(
        ex,
        generated_paraphrases_llama,
        cross_check_qwen,
    )

    return (
        cleaned_cross_llama["paraphrases"][0],
        cleaned_cross_qwen["paraphrases"][0],
    )


def generate_cross_prompting_paraphrases(
    original_q,
    llm_llama,
    tokenizer_llama,
    llm_qwen,
    tokenizer_qwen,
    paraphrase_question_fn,
    paraphrase_prompt,
    n,
    temperature,
    top_p,
):
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

    return generated_paraphrases_llama, generated_paraphrases_qwen


def cross_check_generated_paraphrases(
    original_q,
    generated_paraphrases_llama,
    generated_paraphrases_qwen,
    llm_llama,
    tokenizer_llama,
    llm_qwen,
    tokenizer_qwen,
    check_meaning_fn,
    paraphrases_check_prompt,
    temperature,
    top_p,
):
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

    return cross_check_llama, cross_check_qwen


def cross_prompt_answers(
    original_q,
    clean_paraphrases_qwen,
    clean_paraphrases_llama,
    llm_llama,
    tokenizer_llama,
    llm_qwen,
    tokenizer_qwen,
    ask_fn,
    output_prompt,
    temperature,
    top_p,
):
    answers_llama = ask_fn(
        llm=llm_llama,
        tokenizer=tokenizer_llama,
        paraphrases=questions_for_answering(original_q, clean_paraphrases_qwen),
        prompt_template=output_prompt,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    answers_qwen = ask_fn(
        llm=llm_qwen,
        tokenizer=tokenizer_qwen,
        paraphrases=questions_for_answering(original_q, clean_paraphrases_llama),
        prompt_template=output_prompt,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    return answers_llama, answers_qwen


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


def process_cross_prompting_example(
    ex,
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
):
    original_q = ex["Question"]
    question_id = ex["QuestionId"]

    generated_paraphrases_llama, generated_paraphrases_qwen = generate_cross_prompting_paraphrases(
        original_q,
        llm_llama,
        tokenizer_llama,
        llm_qwen,
        tokenizer_qwen,
        paraphrase_question_fn,
        paraphrase_prompt,
        n,
        temperature,
        top_p,
    )

    cross_check_llama, cross_check_qwen = cross_check_generated_paraphrases(
        original_q,
        generated_paraphrases_llama,
        generated_paraphrases_qwen,
        llm_llama,
        tokenizer_llama,
        llm_qwen,
        tokenizer_qwen,
        check_meaning_fn,
        paraphrases_check_prompt,
        temperature,
        top_p,
    )

    clean_paraphrases_qwen, clean_paraphrases_llama = clean_cross_checked_paraphrases(
        ex,
        generated_paraphrases_qwen,
        cross_check_llama,
        generated_paraphrases_llama,
        cross_check_qwen,
    )

    cross_ask_llama, cross_ask_qwen = cross_prompt_answers(
        original_q,
        clean_paraphrases_qwen,
        clean_paraphrases_llama,
        llm_llama,
        tokenizer_llama,
        llm_qwen,
        tokenizer_qwen,
        ask_fn,
        output_prompt,
        temperature,
        top_p,
    )

    return build_cross_prompting_record(
        original_q=original_q,
        question_id=question_id,
        clean_paraphrases_qwen=clean_paraphrases_qwen,
        clean_paraphrases_llama=clean_paraphrases_llama,
        answers_llama=cross_ask_llama,
        answers_qwen=cross_ask_qwen,
    )


def progress_args_from_cross_prompting_record(index, total, record):
    return (
        index,
        total,
        record["question_id"],
        record["answers_responsed_llama"],
        record["answers_responsed_qwen"],
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
    process_example_fn=process_cross_prompting_example,
):
    cleaned_results = []
    total = len(data)

    for index, ex in enumerate(data, start=1):
        record = process_example_fn(
            ex=ex,
            llm_llama=llm_llama,
            tokenizer_llama=tokenizer_llama,
            llm_qwen=llm_qwen,
            tokenizer_qwen=tokenizer_qwen,
            paraphrase_question_fn=paraphrase_question_fn,
            check_meaning_fn=check_meaning_fn,
            ask_fn=ask_fn,
            paraphrase_prompt=paraphrase_prompt,
            output_prompt=output_prompt,
            paraphrases_check_prompt=paraphrases_check_prompt,
            n=n,
            temperature=temperature,
            top_p=top_p,
        )
        cleaned_results.append(record)

        if progress_fn is not None:
            progress_fn(*progress_args_from_cross_prompting_record(index, total, record))

    return cleaned_results


def save_cross_prompting_results(records, args, save_fn=save_results):
    save_fn(records, args.output_path_cp)


def main():
    config.configure_runtime_environment()

    import torch

    from prompt_template import paraphrase_prompt, output_prompt, paraphrases_check
    from utils import ask
    from utils import check_meaning
    from utils import load_llm
    from utils import paraphrase_question

    args = parse_args()

    llm_llama, tokenizer_llama, llm_qwen, tokenizer_qwen = load_cross_prompting_models(
        args,
        load_llm_fn=load_llm,
        dtype=torch.float16,
    )
    
    # load sample data, a verified subset of TriviaQA_rc verified-wikipedia-dev.json, 
    # which is verified by human annotators to have correct question-answer pairs
    data_verified = load_cross_prompting_examples(config.TRIVIAQA_PATH)

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



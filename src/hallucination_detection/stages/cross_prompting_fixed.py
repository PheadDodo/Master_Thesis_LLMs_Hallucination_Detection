import argparse
import time

from hallucination_detection import config
from hallucination_detection.cli_utils import add_baseline_output_args
from hallucination_detection.cli_utils import add_generation_args
from hallucination_detection.data_utils import load_json
from hallucination_detection.data_utils import normalized_question_id
from hallucination_detection.data_utils import save_results
from hallucination_detection.data_utils import select_verified_examples
from hallucination_detection.data_utils import triviaqa_rows_from_sample
from hallucination_detection.model_loading import add_model_loading_args
from hallucination_detection.model_loading import load_llama_qwen_models


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Official baseline: sample Llama and Qwen n times per original "
            "question and build cp_fixed outputs"
        )
    )

    add_model_loading_args(parser, config)
    add_generation_args(parser, config)
    add_baseline_output_args(parser, config)
    parser.add_argument(
        "--output_path_cp_fixed",
        type=str,
        default=config.OUTPUT_PATH_CP_FIXED,
    )

    return parser.parse_args(argv)


def _records_by_question_id(records, label):
    if not isinstance(records, list):
        raise ValueError(f"{label} records must be a list")

    records_by_id = {}
    for record in records:
        normalized_record = _normalize_baseline_record(record, label)
        normalized_id = normalized_record["question_id"]

        if normalized_id in records_by_id:
            question_id = record.get("question_id")
            raise ValueError(f"duplicate question_id in {label} records: {question_id}")
        records_by_id[normalized_id] = normalized_record
    return records_by_id


def _normalize_baseline_record(record, label):
    normalized_id = _validate_baseline_record(record, label)
    return {
        **record,
        "question_id": normalized_id,
        "original": _require_non_empty_text(
            record.get("original"),
            f"{label} record original",
        ),
    }


def _validate_baseline_record(record, label):
    if not isinstance(record, dict):
        raise ValueError(f"{label} records must be dictionaries")

    question_id = record.get("question_id")
    normalized_id = "" if question_id is None else str(question_id).strip()
    if not normalized_id:
        raise ValueError(f"{label} record must include a non-empty question_id")

    _require_non_empty_text(record.get("original"), f"{label} record original")

    _validate_answer_list(record.get("answers"), f"{label} record answers")

    return normalized_id


def _validate_answer_list(answers, label):
    if not isinstance(answers, list):
        raise ValueError(f"{label} must be a list")
    if any(not isinstance(answer, str) for answer in answers):
        raise ValueError(f"{label} must be strings")


def _validate_answer_count(answers, label, n):
    if len(answers) != n:
        raise ValueError(f"{label} must contain exactly {n} answers")


def _require_non_empty_text(value, label):
    if value is None:
        raise ValueError(f"{label} must be non-empty")
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")

    text = value.strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _require_processed_pair(processed):
    if (
        not isinstance(processed, (tuple, list))
        or len(processed) != 2
    ):
        raise ValueError("process_example_fn must return llama and qwen records")

    return processed


def _require_positive_n(n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("n must be a positive integer")

    if n <= 0:
        raise ValueError("n must be a positive integer")


def build_cross_prompting_fixed(llama_data, qwen_data):
    cp_fixed = []
    llama_by_id = _records_by_question_id(llama_data, "llama")
    qwen_by_id = _records_by_question_id(qwen_data, "qwen")
    if llama_by_id.keys() != qwen_by_id.keys():
        missing_qwen = sorted(llama_by_id.keys() - qwen_by_id.keys())
        missing_llama = sorted(qwen_by_id.keys() - llama_by_id.keys())
        raise ValueError(
            "llama and qwen records must have matching question_id sets "
            f"(missing_qwen={missing_qwen}, missing_llama={missing_llama})"
        )

    for qid, ex_llama in llama_by_id.items():
        original_q = str(ex_llama["original"]).strip()
        answers_llama = ex_llama["answers"]

        qwen_entry = qwen_by_id[qid]
        if qwen_entry["original"] != original_q:
            raise ValueError(
                "matched llama and qwen records must have matching original"
            )
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


def load_cross_prompting_fixed_examples(
    triviaqa_path,
    load_fn=load_json,
    select_fn=select_verified_examples,
):
    sample = load_fn(triviaqa_path)
    return select_fn(sample)


def load_cross_prompting_fixed_models(args, load_llm_fn, dtype, qwen_load_in_8bit=None):
    return load_llama_qwen_models(
        args,
        load_llm_fn=load_llm_fn,
        dtype=dtype,
        qwen_load_in_8bit=qwen_load_in_8bit,
    )


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
    _require_positive_n(n)

    baseline_results = []

    for ex in triviaqa_rows_from_sample(data):
        original_q = _require_non_empty_text(ex.get("Question"), "TriviaQA row Question")

        question_id = normalized_question_id(ex)
        if not question_id:
            raise ValueError("TriviaQA row must include non-empty QuestionId")

        answers = ask_fn(
            llm=llm,
            tokenizer=tokenizer,
            paraphrases=[original_q] * n,
            prompt_template=prompt_template,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        _validate_answer_list(answers, "answers returned by ask_fn")
        _validate_answer_count(answers, "answers returned by ask_fn", n)

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


def progress_args_from_cross_prompting_fixed_records(
    index,
    total,
    baseline_llama,
    baseline_qwen,
):
    return index, total, baseline_llama, baseline_qwen


def process_cross_prompting_fixed_example(
    ex,
    llm_llama,
    tokenizer_llama,
    llm_qwen,
    tokenizer_qwen,
    ask_fn,
    prompt_template,
    n,
    temperature,
    top_p,
):
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

    return baseline_llama, baseline_qwen


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
    process_example_fn=process_cross_prompting_fixed_example,
):
    _require_positive_n(n)
    data = triviaqa_rows_from_sample(data)
    baseline_results_llama = []
    baseline_results_qwen = []
    total = len(data)
    seen_question_ids = set()

    for index, ex in enumerate(data, start=1):
        processed = process_example_fn(
            ex=ex,
            llm_llama=llm_llama,
            tokenizer_llama=tokenizer_llama,
            llm_qwen=llm_qwen,
            tokenizer_qwen=tokenizer_qwen,
            ask_fn=ask_fn,
            prompt_template=prompt_template,
            n=n,
            temperature=temperature,
            top_p=top_p,
        )
        baseline_llama, baseline_qwen = _require_processed_pair(processed)

        baseline_llama = _normalize_baseline_record(baseline_llama, "llama")
        baseline_qwen = _normalize_baseline_record(baseline_qwen, "qwen")
        llama_question_id = baseline_llama["question_id"]
        qwen_question_id = baseline_qwen["question_id"]
        if llama_question_id != qwen_question_id:
            raise ValueError(
                "processed llama and qwen records must have matching question_id"
            )
        if baseline_llama["original"] != baseline_qwen["original"]:
            raise ValueError(
                "processed llama and qwen records must have matching original"
            )
        _validate_answer_count(baseline_llama["answers"], "llama record answers", n)
        _validate_answer_count(baseline_qwen["answers"], "qwen record answers", n)
        if llama_question_id in seen_question_ids:
            raise ValueError(
                f"duplicate question_id in processed records: {llama_question_id}"
            )
        seen_question_ids.add(llama_question_id)

        baseline_results_llama.append(baseline_llama)
        baseline_results_qwen.append(baseline_qwen)

        if progress_fn is not None:
            progress_fn(*progress_args_from_cross_prompting_fixed_records(
                index,
                total,
                baseline_llama,
                baseline_qwen,
            ))

    cp_fixed = build_cross_prompting_fixed(
        baseline_results_llama,
        baseline_results_qwen,
    )

    return cp_fixed, baseline_results_llama, baseline_results_qwen


def save_cross_prompting_fixed_outputs(
    cp_fixed,
    llama_results,
    qwen_results,
    args,
    save_fn=save_results,
):
    save_fn(cp_fixed, args.output_path_cp_fixed)
    save_fn(llama_results, args.output_path_llama)
    save_fn(qwen_results, args.output_path_qwen)


def main():
    config.configure_runtime_environment()

    import torch

    from hallucination_detection.generation import ask
    from hallucination_detection.generation import load_llm
    from hallucination_detection.prompts import output_prompt

    args = parse_args()

    llm_llama, tokenizer_llama, llm_qwen, tokenizer_qwen = load_cross_prompting_fixed_models(
        args,
        load_llm_fn=load_llm,
        dtype=torch.float16,
    )

    # load TriviaQA verified subset
    data = load_cross_prompting_fixed_examples(config.TRIVIAQA_PATH)
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

    save_cross_prompting_fixed_outputs(
        cp_fixed,
        baseline_results_llama,
        baseline_results_qwen,
        args,
    )

    elapsed = time.time() - start_time
    print(f"Cross-prompting fixed completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")


if __name__ == "__main__":
    main()


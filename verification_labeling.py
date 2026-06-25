import argparse
import time

import config

from cli_utils import positive_int
from data_utils import (
    load_json,
    normalized_question_id,
    save_results,
    triviaqa_rows_from_sample,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run RAG verification labeling and save final hallucination labels"
    )

    # load samples with semantic entropy
    parser.add_argument("--output_path_llama_se", type=str, default=config.OUTPUT_PATH_LLAMA_SE)
    parser.add_argument("--output_path_qwen_se", type=str, default=config.OUTPUT_PATH_QWEN_SE)
    parser.add_argument(
        "--output_path_cp_se",
        type=str,
        default=config.OUTPUT_PATH_CP_SE,
    )
    parser.add_argument(
        "--output_path_cp_fixed_se",
        type=str,
        default=config.OUTPUT_PATH_CP_FIXED_SE,
    )

    parser.add_argument("--evidence_dir", type=str, default=config.EVIDENCE_PATH)
    parser.add_argument("--retriever_path", type=str, default=config.RETRIEVER_PATH)
    parser.add_argument("--verifier_path", type=str, default=config.LLM_VERIFIER_PATH)
    parser.add_argument("--top_k", type=positive_int, default=5)

    # output paths for final results after verification labeling
    parser.add_argument(
        "--output_path_final_llama",
        type=str,
        default=config.OUTPUT_PATH_FINAL_LLAMA,
    )
    parser.add_argument(
        "--output_path_final_qwen",
        type=str,
        default=config.OUTPUT_PATH_FINAL_QWEN,
    )
    parser.add_argument("--output_path_final_cp", type=str, default=config.OUTPUT_PATH_FINAL_CP)
    parser.add_argument(
        "--output_path_final_cp_fixed",
        type=str,
        default=config.OUTPUT_PATH_FINAL_CP_FIXED,
    )

    return parser.parse_args(argv)


def _normalized_rag_verification_label(row):
    if "rag_verification" not in row:
        raise ValueError("Rows must include rag_verification when gold_correct is false")

    if not isinstance(row["rag_verification"], str):
        raise ValueError(
            "Rows must include rag_verification as a string when gold_correct is false"
        )

    rag_verification = row["rag_verification"].strip().upper()
    if not rag_verification:
        raise ValueError("Rows must include non-empty rag_verification when gold_correct is false")

    return rag_verification


def _hallucination_label(row):
    if not isinstance(row, dict):
        raise ValueError("Rows must be dictionaries")

    if "gold_correct" not in row:
        raise ValueError("Rows must include gold_correct")

    if not isinstance(row["gold_correct"], bool):
        raise ValueError("Rows must include gold_correct as a boolean")

    if row["gold_correct"]:
        return 0

    rag_verification = _normalized_rag_verification_label(row)
    return 0 if rag_verification == "ABSTAIN" else 1


def add_hallucination_labels(datasets):
    if not isinstance(datasets, list):
        raise ValueError("datasets must be a list")

    pending_labels = []
    for dataset in datasets:
        if not isinstance(dataset, list):
            raise ValueError("Each dataset must be a list")

        pending_labels.append([_hallucination_label(row) for row in dataset])

    for dataset, labels in zip(datasets, pending_labels):
        for row, label in zip(dataset, labels):
            row["is_hallucinating_true"] = label


def _normalized_generated_question_id(row):
    return str(row.get("question_id") or "").strip()


def _require_list_of_dict_rows(rows, label):
    if not isinstance(rows, list):
        raise ValueError(f"{label} rows must be a list")

    if any(not isinstance(ex, dict) for ex in rows):
        raise ValueError(f"{label} rows must be dictionaries")


def _require_unique_ids(ids, label):
    seen = set()
    duplicates = []
    for question_id in ids:
        if question_id in seen and question_id not in duplicates:
            duplicates.append(question_id)
        seen.add(question_id)

    if duplicates:
        raise ValueError(f"{label} rows must not contain duplicate question_id values")


def select_verification_examples(triviaqa_rows, cp_fixed_sample):
    _require_list_of_dict_rows(cp_fixed_sample, "cp_fixed sample")

    generated_ids = [_normalized_generated_question_id(ex) for ex in cp_fixed_sample]
    missing_question_id = [question_id for question_id in generated_ids if not question_id]
    if missing_question_id:
        raise ValueError("cp_fixed sample rows must include non-empty question_id")
    _require_unique_ids(generated_ids, "cp_fixed sample")

    _require_list_of_dict_rows(triviaqa_rows, "TriviaQA")

    missing_triviaqa_id = [
        ex for ex in triviaqa_rows
        if not normalized_question_id(ex)
    ]
    if missing_triviaqa_id:
        raise ValueError("TriviaQA rows must include non-empty QuestionId")

    sample_ids = set(generated_ids)
    triviaqa_ids = {normalized_question_id(ex) for ex in triviaqa_rows}
    missing_triviaqa_ids = sorted(sample_ids - triviaqa_ids)
    if missing_triviaqa_ids:
        raise ValueError(
            "cp_fixed sample question_id values must exist in TriviaQA rows "
            f"(missing_triviaqa={missing_triviaqa_ids})"
        )

    selected_ids = [
        normalized_question_id(ex)
        for ex in triviaqa_rows
        if normalized_question_id(ex) in sample_ids
    ]
    _require_unique_ids(selected_ids, "TriviaQA")

    return [ex for ex in triviaqa_rows if normalized_question_id(ex) in sample_ids]


def has_paraphrases(sample):
    if not isinstance(sample, list) or not sample:
        return False

    if not isinstance(sample[0], dict):
        return False

    return sample[0].get("paraphrases") is not None


def paraphrase_status(sample):
    if has_paraphrases(sample):
        return True, "Paraphrases are included in the sample."

    return False, "Paraphrases are not included in the sample."


def prepare_verification_run(triviaqa_rows, cp_fixed_sample):
    data = select_verification_examples(triviaqa_rows, cp_fixed_sample)
    paraphrases_check, paraphrases_message = paraphrase_status(cp_fixed_sample)
    return {
        "data": data,
        "paraphrases_check": paraphrases_check,
        "paraphrases_message": paraphrases_message,
    }


def load_verification_inputs(
    args,
    triviaqa_path=config.TRIVIAQA_PATH,
    load_fn=load_json,
):
    triviaqa_sample = load_fn(triviaqa_path)

    return {
        "triviaqa_rows": triviaqa_rows_from_sample(triviaqa_sample),
        "cp_fixed_sample": load_fn(args.output_path_cp_fixed_se),
        "baseline_llama": load_fn(args.output_path_llama_se),
        "baseline_qwen": load_fn(args.output_path_qwen_se),
    }


def load_verification_pipeline(
    args,
    retriever_cls,
    verifier_cls,
    pipeline_cls,
    top_k=None,
):
    if top_k is None:
        top_k = args.top_k

    return pipeline_cls(
        evidence_dir=args.evidence_dir,
        retriever=retriever_cls(args.retriever_path),
        verifier=verifier_cls(args.verifier_path),
        top_k=top_k,
    )


def save_verification_outputs(
    final_data_cp_fixed,
    final_baseline_llama,
    final_baseline_qwen,
    args,
    save_fn=save_results,
):
    save_fn(final_baseline_llama, args.output_path_final_llama)
    save_fn(final_baseline_qwen, args.output_path_final_qwen)
    save_fn(final_data_cp_fixed, args.output_path_final_cp_fixed)


def _require_rag_pipeline_outputs(outputs):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise ValueError(
            "RAG pipeline must return cp_fixed, llama, and qwen result collections"
        )

    final_data_cp_fixed, final_baseline_llama, final_baseline_qwen = outputs
    datasets = [
        final_data_cp_fixed,
        final_baseline_llama,
        final_baseline_qwen,
    ]
    if any(not isinstance(dataset, list) for dataset in datasets):
        raise ValueError(
            "RAG pipeline must return cp_fixed, llama, and qwen result collections"
        )

    lengths = {len(dataset) for dataset in datasets}
    if len(lengths) != 1:
        raise ValueError("RAG pipeline result collections must have the same length")

    return final_data_cp_fixed, final_baseline_llama, final_baseline_qwen


def _require_verification_input_collections(
    data,
    cp_fixed_sample,
    baseline_llama,
    baseline_qwen,
):
    _require_list_of_dict_rows(data, "verification data")
    _require_list_of_dict_rows(cp_fixed_sample, "cp_fixed sample")
    _require_list_of_dict_rows(baseline_llama, "baseline_llama")
    _require_list_of_dict_rows(baseline_qwen, "baseline_qwen")

    lengths = {
        len(data),
        len(cp_fixed_sample),
        len(baseline_llama),
        len(baseline_qwen),
    }
    if len(lengths) != 1:
        raise ValueError("Verification input collections must have the same length")


def run_verification_outputs(
    rag_pipeline,
    data,
    cp_fixed_sample,
    baseline_llama,
    baseline_qwen,
    paraphrases_check,
):
    _require_verification_input_collections(
        data,
        cp_fixed_sample,
        baseline_llama,
        baseline_qwen,
    )

    final_data_cp_fixed, final_baseline_llama, final_baseline_qwen = _require_rag_pipeline_outputs(
        rag_pipeline.run_verification_pipeline(
            data,
            cp_fixed_sample,
            baseline_llama,
            baseline_qwen,
            paraphrases=paraphrases_check,
        )
    )

    add_hallucination_labels([
        final_data_cp_fixed,
        final_baseline_llama,
        final_baseline_qwen,
    ])

    return final_data_cp_fixed, final_baseline_llama, final_baseline_qwen


def main():
    config.configure_runtime_environment()

    from llm_with_rag_verifier import BGERetriever, LLMVerifier, RAGVerificationPipeline

    args = parse_args()

    start_time = time.time()

    llm_rag = load_verification_pipeline(
        args,
        retriever_cls=BGERetriever,
        verifier_cls=LLMVerifier,
        pipeline_cls=RAGVerificationPipeline,
    )

    inputs = load_verification_inputs(args)
    data_all = inputs["triviaqa_rows"]
    cp_fixed_sample = inputs["cp_fixed_sample"]
    baseline_llama = inputs["baseline_llama"]
    baseline_qwen = inputs["baseline_qwen"]

    prepared = prepare_verification_run(data_all, cp_fixed_sample)
    data = prepared["data"]
    print(f"Loaded {len(data)} verified question-answer pairs for verification labeling")

    print(prepared["paraphrases_message"])

    final_data_cp_fixed, final_baseline_llama, final_baseline_qwen = run_verification_outputs(
        rag_pipeline=llm_rag,
        data=data,
        cp_fixed_sample=cp_fixed_sample,
        baseline_llama=baseline_llama,
        baseline_qwen=baseline_qwen,
        paraphrases_check=prepared["paraphrases_check"],
    )

    save_verification_outputs(
        final_data_cp_fixed,
        final_baseline_llama,
        final_baseline_qwen,
        args,
    )
    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed / 60 :.2f} minutes")


if __name__ == "__main__":
    main()


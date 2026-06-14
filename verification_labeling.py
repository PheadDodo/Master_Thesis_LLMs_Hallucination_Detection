import argparse
import time

import config

from cli_utils import positive_int
from data_utils import load_json, save_results, triviaqa_rows_from_sample


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run RAG verification labeling and save final hallucination labels"
    )
    

    # load samples with semantic entropy
    parser.add_argument("--output_path_llama_se", type=str, default=config.OUTPUT_PATH_LLAMA_SE)
    parser.add_argument("--output_path_qwen_se", type=str, default=config.OUTPUT_PATH_QWEN_SE)
    parser.add_argument("--output_path_cp_se", type=str, default=config.OUTPUT_PATH_CP_SE)
    parser.add_argument("--output_path_cp_fixed_se", type=str, default=config.OUTPUT_PATH_CP_FIXED_SE)
    
    parser.add_argument("--evidence_dir", type=str, default=config.EVIDENCE_PATH)
    parser.add_argument("--retriever_path", type=str, default=config.RETRIEVER_PATH)
    parser.add_argument("--verifier_path", type=str, default=config.LLM_VERIFIER_PATH)
    parser.add_argument("--top_k", type=positive_int, default=5)
    
    # output paths for final results after verification labeling
    parser.add_argument("--output_path_final_llama", type=str, default=config.OUTPUT_PATH_FINAL_LLAMA)
    parser.add_argument("--output_path_final_qwen", type=str, default=config.OUTPUT_PATH_FINAL_QWEN)
    parser.add_argument("--output_path_final_cp", type=str, default=config.OUTPUT_PATH_FINAL_CP)
    parser.add_argument("--output_path_final_cp_fixed", type=str, default=config.OUTPUT_PATH_FINAL_CP_FIXED)

    return parser.parse_args(argv)


def add_hallucination_labels(datasets):
    for dataset in datasets:
        for row in dataset:
            row["is_hallucinating_true"] = (
                0
                if row["gold_correct"] or row["rag_verification"] == "ABSTAIN"
                else 1
            )


def select_verification_examples(triviaqa_rows, cp_fixed_sample):
    missing_question_id = [ex for ex in cp_fixed_sample if "question_id" not in ex]
    if missing_question_id:
        raise ValueError("cp_fixed sample rows must include question_id")

    missing_triviaqa_id = [ex for ex in triviaqa_rows if "QuestionId" not in ex]
    if missing_triviaqa_id:
        raise ValueError("TriviaQA rows must include QuestionId")

    sample_ids = {ex["question_id"] for ex in cp_fixed_sample}
    return [ex for ex in triviaqa_rows if ex["QuestionId"] in sample_ids]


def has_paraphrases(sample):
    if not sample:
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


def run_verification_outputs(
    rag_pipeline,
    data,
    cp_fixed_sample,
    baseline_llama,
    baseline_qwen,
    paraphrases_check,
):
    final_data_cp_fixed, final_baseline_llama, final_baseline_qwen = (
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





import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time

import config

from data_utils import load_json, save_results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute semantic entropy and select final answers"
    )
    

    # load samples with semantic entropy
    parser.add_argument("--output_path_llama_se", type=str, default=config.OUTPUT_PATH_LLAMA_SE)
    parser.add_argument("--output_path_qwen_se", type=str, default=config.OUTPUT_PATH_QWEN_SE)
    parser.add_argument("--output_path_cp_se", type=str, default=config.OUTPUT_PATH_CP_SE)
    parser.add_argument("--output_path_cp_fixed_se", type=str, default=config.OUTPUT_PATH_CP_FIXED_SE)
    
    parser.add_argument("--evidence_dir", type=str, default=config.EVIDENCE_PATH)
    parser.add_argument("--retriever_path", type=str, default=config.RETRIEVER_PATH)
    parser.add_argument("--verifier_path", type=str, default=config.LLM_VERIFIER_PATH)
    
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
    sample_ids = {ex["question_id"] for ex in cp_fixed_sample}
    return [ex for ex in triviaqa_rows if ex["QuestionId"] in sample_ids]


def has_paraphrases(sample):
    return sample[0].get("paraphrases") is not None


def main():
    from llm_with_rag_verifier import BGERetriever, LLMVerifier, RAGVerificationPipeline

    args = parse_args()
    
    start_time = time.time()
    
    llm_rag = RAGVerificationPipeline(
    evidence_dir=args.evidence_dir,
    retriever=BGERetriever(args.retriever_path),
    verifier=LLMVerifier(args.verifier_path),
    top_k=5,
    )
    
    data_all = load_json(config.TRIVIAQA_PATH)
    data_all = [ex for ex in data_all["Data"]]
    
    #cp_sample = load_json(args.output_path_cp_se)
    cp_fixed_sample = load_json(args.output_path_cp_fixed_se)
    baseline_llama = load_json(args.output_path_llama_se)
    baseline_qwen = load_json(args.output_path_qwen_se)

    data = select_verification_examples(data_all, cp_fixed_sample)
    print(f"Loaded {len(data)} verified question-answer pairs for verification labeling")
    
    if has_paraphrases(cp_fixed_sample):
        paraphrases_check = True
        print("Paraphrases are included in the sample.")
    else:
        paraphrases_check = False
        print("Paraphrases are not included in the sample.")

    final_data_cp_fixed, final_baseline_llama, final_baseline_qwen = llm_rag.run_verification_pipeline(
        data, cp_fixed_sample, baseline_llama, baseline_qwen, paraphrases=paraphrases_check
    )
    
    # label ground truth is_hallucinating_true based on gold alias matched, 0 if correct/abstain, 1 if incorrect
    add_hallucination_labels([final_data_cp_fixed, final_baseline_llama, final_baseline_qwen])
    
    #save_results(final_data_cp, args.output_path_final_cp)
    save_results(final_baseline_llama, args.output_path_final_llama)
    save_results(final_baseline_qwen, args.output_path_final_qwen)
    save_results(final_data_cp_fixed, args.output_path_final_cp_fixed)
    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed / 60 :.2f} minutes")


if __name__ == "__main__":
    main()





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
    
    # load samples
    parser.add_argument("--output_path_llama", type=str, default=config.OUTPUT_PATH_LLAMA)
    parser.add_argument("--output_path_qwen", type=str, default=config.OUTPUT_PATH_QWEN)
    parser.add_argument("--output_path_cp", type=str, default=config.OUTPUT_PATH_CP)
    parser.add_argument("--output_path_cp_fixed", type=str, default=config.OUTPUT_PATH_CP_FIXED)

    # output paths for samples with semantic entropy
    parser.add_argument("--output_path_llama_se", type=str, default=config.OUTPUT_PATH_LLAMA_SE)
    parser.add_argument("--output_path_qwen_se", type=str, default=config.OUTPUT_PATH_QWEN_SE)
    parser.add_argument("--output_path_cp_se", type=str, default=config.OUTPUT_PATH_CP_SE)
    parser.add_argument("--output_path_cp_fixed_se", type=str, default=config.OUTPUT_PATH_CP_FIXED_SE)


    return parser.parse_args(argv)


def compute_semantic_entropy_results(
    analyzer,
    samples_llama,
    samples_qwen,
    samples_cp,
    samples_cp_fixed,
):
    return {
        "cp": analyzer.process_sample_collection(samples_cp),
        "cp_fixed": analyzer.process_sample_collection(samples_cp_fixed),
        "llama": analyzer.baseline_semantic_entropy(samples_llama),
        "qwen": analyzer.baseline_semantic_entropy(samples_qwen),
    }


def main():
    from semantic_entropy import EntailmentDeberta, SemanticEntropyAnalyzer

    args = parse_args()
    
    # load samples
    start_time = time.time()
    
    samples_llama = load_json(args.output_path_llama)
    samples_qwen = load_json(args.output_path_qwen)
    samples_cp = load_json(args.output_path_cp)
    samples_cp_fixed = load_json(args.output_path_cp_fixed)
    
    se_analyzer = SemanticEntropyAnalyzer(entailment_model=EntailmentDeberta(verbose=False))
    print("Initialized Semantic Entropy Analyzer")

    results = compute_semantic_entropy_results(
        analyzer=se_analyzer,
        samples_llama=samples_llama,
        samples_qwen=samples_qwen,
        samples_cp=samples_cp,
        samples_cp_fixed=samples_cp_fixed,
    )
    
    save_results(results["cp"], args.output_path_cp_se)
    save_results(results["cp_fixed"], args.output_path_cp_fixed_se)
    save_results(results["llama"], args.output_path_llama_se)
    save_results(results["qwen"], args.output_path_qwen_se)
    
    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed :.2f} seconds")


if __name__ == "__main__":
    main()








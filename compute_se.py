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


def load_semantic_entropy_inputs(args, load_fn=load_json):
    return {
        "llama": load_fn(args.output_path_llama),
        "qwen": load_fn(args.output_path_qwen),
        "cp": load_fn(args.output_path_cp),
        "cp_fixed": load_fn(args.output_path_cp_fixed),
    }


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


def save_semantic_entropy_results(results, args, save_fn=save_results):
    save_fn(results["cp"], args.output_path_cp_se)
    save_fn(results["cp_fixed"], args.output_path_cp_fixed_se)
    save_fn(results["llama"], args.output_path_llama_se)
    save_fn(results["qwen"], args.output_path_qwen_se)


def main():
    config.configure_runtime_environment()

    from semantic_entropy import EntailmentDeberta, SemanticEntropyAnalyzer

    args = parse_args()
    
    # load samples
    start_time = time.time()
    
    samples = load_semantic_entropy_inputs(args)
    
    se_analyzer = SemanticEntropyAnalyzer(entailment_model=EntailmentDeberta(verbose=False))
    print("Initialized Semantic Entropy Analyzer")

    results = compute_semantic_entropy_results(
        analyzer=se_analyzer,
        samples_llama=samples["llama"],
        samples_qwen=samples["qwen"],
        samples_cp=samples["cp"],
        samples_cp_fixed=samples["cp_fixed"],
    )
    
    save_semantic_entropy_results(results, args)
    
    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed :.2f} seconds")


if __name__ == "__main__":
    main()








import os


def configure_runtime_environment():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# model paths
LLM_LLAMA_PATH = "./models/Meta-Llama-3.1-8B-Instruct"
LLM_QWEN_PATH = "./models/Qwen3-8B"

# model loading
LOAD_IN_8BIT = True
QWEN_LOAD_IN_8BIT = False
DEVICE_MAP = "auto"
QWEN_DEVICE_MAP = "cpu"
DTYPE = "float16"

# number of sampling
N = 14

# generation
TEMPERATURE = 0.5
TOP_P = 0.95

# data paths
TRIVIAQA_PATH = "./datasets/TriviaQA/triviaqa-rc/qa/verified-wikipedia-dev.json"
CP_DATA_PATH = "./datasets/TriviaQA/triviaqa-rc/qa/cross_prompting_samples_1_318_t05_p095_0411.json"

# preliminary output paths
OUTPUT_PATH_LLAMA = "./datasets/TriviaQA/triviaqa-rc/qa/baseline_samples_1_318_t05_p095_llama_0512.json"
OUTPUT_PATH_QWEN = "./datasets/TriviaQA/triviaqa-rc/qa/baseline_samples_1_318_t05_p095_qwen_0512.json"
OUTPUT_PATH_CP = "./datasets/TriviaQA/triviaqa-rc/qa/cross_prompting_samples_1_318_t05_p095_0426.json"
OUTPUT_PATH_CP_FIXED = "./datasets/TriviaQA/triviaqa-rc/qa/cross_prompting_samples_1_318_t05_p095_fixed_0512.json"

# semantic_entropy output paths
OUTPUT_PATH_LLAMA_SE = "./datasets/TriviaQA/triviaqa-rc/qa/cleaned_samples_1_318_t05_p095_llama_with_entropy_0512.json"
OUTPUT_PATH_QWEN_SE = "./datasets/TriviaQA/triviaqa-rc/qa/cleaned_samples_1_318_t05_p095_qwen_with_entropy_0512.json"
OUTPUT_PATH_CP_SE = "./datasets/TriviaQA/triviaqa-rc/qa/cleaned_samples_1_318_t05_p095_cp_with_entropy_0426.json"
OUTPUT_PATH_CP_FIXED_SE = "./datasets/TriviaQA/triviaqa-rc/qa/cleaned_samples_1_318_t05_p095_cp_fixed_with_entropy_0512.json"

# final output paths after verification labeling
OUTPUT_PATH_FINAL_LLAMA = "./datasets/TriviaQA/triviaqa-rc/qa/final_baseline_llama_1_318_t05_p095_llama_0512.json"
OUTPUT_PATH_FINAL_QWEN = "./datasets/TriviaQA/triviaqa-rc/qa/final_baseline_qwen_1_318_t05_p095_llama_0512.json"
OUTPUT_PATH_FINAL_CP = "./datasets/TriviaQA/triviaqa-rc/qa/final_data_cp_1_318_t05_p095_llama_0426.json"
OUTPUT_PATH_FINAL_CP_FIXED = "./datasets/TriviaQA/triviaqa-rc/qa/final_data_cp_fixed_1_318_t05_p095_llama_0512.json"

# retriever paths
EVIDENCE_PATH = "./datasets/TriviaQA/triviaqa-rc/evidence/wikipedia"
RETRIEVER_PATH = "./models/bge-large-en-v1.5"
# llm verifier paths
LLM_VERIFIER_PATH = "./models/Meta-Llama-3.1-8B-Instruct"







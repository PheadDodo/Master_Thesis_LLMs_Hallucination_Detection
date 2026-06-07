import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time

import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross prompting: generate paraphrases, cross-check meaning, and obtain answers"
    )

    parser.add_argument("--llm_llama_path", type=str, default=config.LLM_LLAMA_PATH)
    parser.add_argument("--llm_qwen_path", type=str, default=config.LLM_QWEN_PATH)
    parser.add_argument("--device_map", type=str, default=config.DEVICE_MAP)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=config.TOP_P)
    parser.add_argument("--n", type=int, default=config.N)
    parser.add_argument("--output_path_cp", type=str, default=config.OUTPUT_PATH_CP)

    parser.add_argument("--load_in_8bit", dest="load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=config.LOAD_IN_8BIT)

    return parser.parse_args()


def select_verified_examples(sample):
    data = [
        ex for ex in sample["Data"]
        if ex.get("QuestionPartOfVerifiedEval") is True
    ]
    data_ids = {ex.get("QuestionId") for ex in data}
    return [
        ex for ex in sample["Data"]
        if ex.get("QuestionId") in data_ids
    ]


def questions_for_answering(original_q, valid_paraphrases):
    return valid_paraphrases if valid_paraphrases else [original_q]


def main():
    import torch

    from prompt_template import paraphrase_prompt, output_prompt, paraphrases_check
    from utils import ask
    from utils import append_valid_paraphrases
    from utils import check_meaning
    from utils import cleaned_question_answers
    from utils import load_json
    from utils import load_llm
    from utils import paraphrase_question
    from utils import save_results

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
        device_map="cpu",
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
    
    cleaned_results = []
    for ex in data_verified[:]:
        original_q = ex["Question"]
        question_id = ex["QuestionId"]
        
        n = args.n # number of paraphrases to generate per question for cross prompting
        
        # generate paraphrased questions
        generated_paraphrases_llama = paraphrase_question(
            llm=llm_llama,
            tokenizer=tokenizer_llama,
            original_q=original_q,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            n=n,
            prompt_template=paraphrase_prompt
            )
        generated_paraphrases_qwen = paraphrase_question(
            llm=llm_qwen,
            tokenizer=tokenizer_qwen,
            original_q=original_q,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            n=n,
            prompt_template=paraphrase_prompt
            )
        
        # llama checks qwen's paraphrases
        cross_check_llama = check_meaning(
            llm=llm_llama,
            tokenizer=tokenizer_llama,
            original_q=original_q,
            paraphrases=generated_paraphrases_qwen,
            prompt_template=paraphrases_check,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            )
        # qwen checks llama's paraphrases
        cross_check_qwen = check_meaning(
            llm=llm_qwen,
            tokenizer=tokenizer_qwen,
            original_q=original_q,
            paraphrases=generated_paraphrases_llama,
            prompt_template=paraphrases_check,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            )
        
        # save same-meaning paraphrases according to cross-check results
        cleaned_cross_llama = append_valid_paraphrases(ex, generated_paraphrases_qwen, cross_check_llama)
        cleaned_cross_qwen = append_valid_paraphrases(ex, generated_paraphrases_llama, cross_check_qwen)
        clean_paraphrases_qwen = cleaned_cross_llama["paraphrases"][0]
        clean_paraphrases_llama = cleaned_cross_qwen["paraphrases"][0]
        
        # cross prompting for answers, i.e. ask llama with qwen's cleaned paraphrases and ask qwen with llama's cleaned paraphrases
        cross_ask_llama = ask(
            llm=llm_llama, 
            tokenizer=tokenizer_llama, 
            paraphrases=questions_for_answering(original_q, clean_paraphrases_qwen),
            prompt_template=output_prompt,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        cross_ask_qwen = ask(
            llm=llm_qwen, 
            tokenizer=tokenizer_qwen, 
            paraphrases=questions_for_answering(original_q, clean_paraphrases_llama),
            prompt_template=output_prompt,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        # save cleaned paraphrases and answers
        final_questions_answers = cleaned_question_answers(
            original_q=original_q,
            question_id=question_id,
            clean_paraphrases_1=clean_paraphrases_qwen,
            clean_paraphrases_2=clean_paraphrases_llama,
            answers_1=cross_ask_llama,
            answers_2=cross_ask_qwen
        )
        
        cleaned_results.append(final_questions_answers)
        
        print(
            f"[{len(cleaned_results)}/{len(data_verified)}] QID={question_id} | n={n} | answers_llama={cross_ask_llama} | answers_qwen={cross_ask_qwen}"
        )
    
    #save results, do_sample=True, temperature=0.5, top_p=0.95
    save_results(cleaned_results, args.output_path_cp)
    
    print("Cross prompting completed.")
    elapsed = time.time() - start_time
    print(f"Time elapsed: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    
if __name__ == "__main__":
    main()



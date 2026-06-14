import json
import re


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} entries to {output_path}")


def has_non_empty_question_id(row):
    return bool(normalized_question_id(row))


def normalized_question_id(row):
    return str(row.get("QuestionId") or "").strip()


def verified_question_ids(rows):
    if any(not has_non_empty_question_id(ex) for ex in rows):
        raise ValueError("TriviaQA rows must include non-empty QuestionId")

    return {
        normalized_question_id(ex)
        for ex in rows
        if ex.get("QuestionPartOfVerifiedEval") is True
    }


def select_verified_examples(sample):
    rows = triviaqa_rows_from_sample(sample)
    data_ids = verified_question_ids(rows)
    return [
        ex for ex in rows
        if normalized_question_id(ex) in data_ids
    ]


def triviaqa_rows_from_sample(sample):
    if isinstance(sample, list):
        rows = sample
    else:
        if not isinstance(sample, dict):
            raise ValueError("TriviaQA sample must be a list or include Data")

        if "Data" not in sample:
            raise ValueError("TriviaQA sample must be a list or include Data")

        if not isinstance(sample["Data"], list):
            raise ValueError("TriviaQA Data must be a list")

        rows = sample["Data"]

    if any(not isinstance(ex, dict) for ex in rows):
        raise ValueError("TriviaQA rows must be dictionaries")

    return rows


def append_valid_paraphrases(ex, paraphrases, meaning_results):
    question_template = {
        "original": [],
        "paraphrases": []
    }

    original_question = ex.get("Question", "").strip()
    if not original_question:
        return question_template

    valid_paraphrases = []
    for paraphrase, verdict in zip(paraphrases, meaning_results):
        first_word = ""
        if isinstance(verdict, str):
            match = re.search(r"[A-Za-z]+", verdict.strip().lower())
            first_word = match.group(0) if match else ""

        if first_word == "yes":
            cleaned = paraphrase.strip()
            if cleaned and cleaned.lower() != original_question.lower():
                valid_paraphrases.append(cleaned)

    question_template["original"].append(original_question)
    question_template["paraphrases"].append(valid_paraphrases)

    return question_template

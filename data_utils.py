import re


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

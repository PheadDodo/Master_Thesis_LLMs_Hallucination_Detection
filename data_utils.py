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

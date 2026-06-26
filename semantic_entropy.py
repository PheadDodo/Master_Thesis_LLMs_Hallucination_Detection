import os
import re
import logging
import unicodedata
import math

from collections import Counter


# reproduced from https://github.com/jlko/semantic_uncertainty

def _fix_text(text):
    try:
        from ftfy import fix_text
    except ImportError:
        return text

    return fix_text(text)

class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    """
    Cleaner entailment wrapper for short-phrase and sentence-length comparisons.

    Design:
    - Normalize text aggressively for noisy short answers
    - Use lexical heuristics for short phrases
    - Use DeBERTa MNLI for sentence-like answers
    - Cache pairwise predictions
    """

    def __init__(
        self,
        model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        cache_dir="./",
        device=None,
        short_phrase_max_tokens=3,
        wrap_in_answer_template=True,
        verbose=False,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.short_phrase_max_tokens = short_phrase_max_tokens
        self.wrap_in_answer_template = wrap_in_answer_template
        self.verbose = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = {v.upper(): k for k, v in self.id2label.items()}

        self.prediction_cache = {}

    # ----------------------------
    # Normalization / text helpers
    # ----------------------------

    def normalize_text(self, text):
        if text is None:
            return ""

        text = _fix_text(str(text))
        text = unicodedata.normalize("NFKC", text)
        text = text.lower().strip()

        # remove repeated leading numbering/bullets
        text = re.sub(r"^(?:\s*[\-\*\•]\s*)+", "", text)
        text = re.sub(r"^(?:\s*\(?\d+[\.\)]\s*)+", "", text)

        # remove dangling trailing numbering artifacts
        text = re.sub(r"(?:\s*[,;]?\s*\(?\d+[\.\)]\s*)+$", "", text)

        # remove unfinished parenthetical tail
        text = re.sub(r"\([^)]*$", "", text)

        # normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # strip edge punctuation
        text = text.strip(" .,!?:;\"'`-–—()[]{}")

        # remove leading articles
        text = re.sub(r"^(the|a|an)\s+", "", text)

        return text
    
    def is_short_phrase(self, text, max_words=5):
        tokens = [t for t in text.split() if t not in {"the", "a", "an"}]
        return len(tokens) <= max_words

    def get_head_word(self, text):
        tokens = text.split()
        return tokens[-1] if tokens else ""

    def token_set(self, text):
        return set(text.split())

    def token_prefix_match(self, a, b):
        a_tokens = a.split()
        b_tokens = b.split()
        if not a_tokens or not b_tokens:
            return False
        return (
            a_tokens == b_tokens[: len(a_tokens)]
            or b_tokens == a_tokens[: len(b_tokens)]
        )

    def wrap_text_for_nli(self, text):
        if not self.wrap_in_answer_template:
            return text
        return f"The answer is: {text}."
    
    def split_comma_parts(self, text):
        return [p.strip() for p in text.split(",") if p.strip()]

    def appositional_prefix_match(self, a, b):
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        return longer.startswith(shorter + ",")

    def comma_prefix_entity_match(self, a, b):
        pa = self.split_comma_parts(a)
        pb = self.split_comma_parts(b)

        if not pa or not pb:
            return False

        if pa[0] != pb[0]:
            return False

        # keep this conservative
        return len(pa[0].split()) <= 4

    # ----------------------------
    # Model inference
    # ----------------------------

    def check_implication(self, text1, text2):
        """
        Check whether text2 is entailed by text1.
        Returns: (prediction_id, prediction_label)
        """
        import torch

        premise = self.wrap_text_for_nli(text1)
        hypothesis = self.wrap_text_for_nli(text2)

        cache_key = ("implication", premise, hypothesis)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        prediction_id = torch.argmax(logits, dim=1).item()
        prediction_label = self.id2label[prediction_id]

        result = (prediction_id, prediction_label)
        self.prediction_cache[cache_key] = result

        if self.verbose or os.environ.get("DEBERTA_FULL_LOG", False):
            logging.info("DeBERTa Input: %s -> %s", premise, hypothesis)
            logging.info("DeBERTa Prediction: %s (%s)", prediction_id, prediction_label)

        return result

    def is_bidirectional_entailment(self, text1, text2, debug=False):
        id1, label1 = self.check_implication(text1, text2)
        id2, label2 = self.check_implication(text2, text1)

        if debug or self.verbose:
            print(f"{text1!r} -> {text2!r}: {label1} ({id1})")
            print(f"{text2!r} -> {text1!r}: {label2} ({id2})")

        return label1.upper() == "ENTAILMENT" and label2.upper() == "ENTAILMENT"

    # ----------------------------
    # Semantic equivalence
    # ----------------------------

    def is_semantically_equivalent_noisy(self, text1, text2, debug=False):
        a = self.normalize_text(text1)
        b = self.normalize_text(text2)

        if debug or self.verbose:
            print(f"normalized: {text1!r} -> {a!r}")
            print(f"normalized: {text2!r} -> {b!r}")

        if not a or not b:
            return False

        # exact normalized match
        if a == b:
            return True

        # NEW: chicago == chicago, illinois
        if self.appositional_prefix_match(a, b):
            return True

        # NEW: same first comma-segment entity
        if self.comma_prefix_entity_match(a, b):
            return True

        # your existing short-phrase / head-word / entailment logic below
        short_a = self.is_short_phrase(a)
        short_b = self.is_short_phrase(b)

        if short_a and short_b:
            head_a = self.get_head_word(a)
            head_b = self.get_head_word(b)

            if head_a and head_a == head_b:
                return True

            if self.token_prefix_match(a, b):
                return True

            return False

        return self.is_bidirectional_entailment(a, b, debug=debug)
    
    
class SemanticEntropyAnalyzer:
    """
    Wraps semantic clustering, entropy computation, and answer selection
    around a given entailment model.
    """

    def __init__(self, entailment_model: EntailmentDeberta, verbose: bool = False):
        self.entailment_model = entailment_model
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def is_idk(answer: str) -> bool:
        if not answer:
            return True
        idk_patterns = {
            "i don't know",
            "i dont know",
            "i do not know",
            "unknown",
            "no answer",
            "cannot answer",
        }
        return answer.strip().lower() in idk_patterns

    @staticmethod
    def semantic_entropy_via_clustering(semantic_ids) -> float:
        """Estimate semantic uncertainty from cluster-assignment distribution.

        Input:
            semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
        Output:
            cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
        """
        if semantic_ids is None:
            return float("inf")

        semantic_ids = list(semantic_ids)
        if len(semantic_ids) == 0:
            return float("inf")
        if len(semantic_ids) == 1:
            return 0.0

        n_generations = len(semantic_ids)
        counts = Counter(semantic_ids)
        entropy = -sum(
            (count / n_generations) * math.log(count / n_generations)
            for count in counts.values()
        )
        return max(0.0, float(entropy))

    @staticmethod
    def get_final_answer_from_clusters(answers, semantic_ids) -> str:
        if not answers or not semantic_ids:
            return ""
        if len(answers) != len(semantic_ids):
            raise ValueError("Length mismatch between answers and semantic_ids.")

        majority_sid = Counter(semantic_ids).most_common(1)[0][0]
        clustered_answers = [
            a for a, sid in zip(answers, semantic_ids) if sid == majority_sid
        ]
        if not clustered_answers:
            return ""
        return Counter(clustered_answers).most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def semantic_clustering_graph(self, answers_list) -> list:
        n = len(answers_list)
        graph = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                is_match = self.entailment_model.is_semantically_equivalent_noisy(
                    answers_list[i], answers_list[j]
                )
                if self.verbose:
                    print(
                        f"Compare {i}-{j}: {is_match} | "
                        f"{answers_list[i]!r} <-> {answers_list[j]!r}"
                    )
                if is_match:
                    graph[i].append(j)
                    graph[j].append(i)

        cluster_ids = [-1] * n
        current_cluster = 0

        for i in range(n):
            if cluster_ids[i] == -1:
                stack = [i]
                cluster_ids[i] = current_cluster
                if self.verbose:
                    print(
                        f"Starting cluster {current_cluster} from node {i}: "
                        f"{answers_list[i]!r}"
                    )
                while stack:
                    node = stack.pop()
                    for neighbor in graph[node]:
                        if cluster_ids[neighbor] == -1:
                            cluster_ids[neighbor] = current_cluster
                            stack.append(neighbor)
                current_cluster += 1

        return cluster_ids

    # ------------------------------------------------------------------
    # Per-model processing
    # ------------------------------------------------------------------

    def process_one_model(self, answers) -> dict:
        if not answers:
            return {
                "semantic_ids": [],
                "semantic_entropy": float("inf"),
                "final_answer": "",
            }

        semantic_ids = self.semantic_clustering_graph(answers)
        semantic_entropy = self.semantic_entropy_via_clustering(semantic_ids)
        final_answer = self.get_final_answer_from_clusters(answers, semantic_ids)

        return {
            "semantic_ids": semantic_ids,
            "semantic_entropy": semantic_entropy,
            "final_answer": final_answer,
        }

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    @staticmethod
    def select_between_models(llama_result: dict, qwen_result: dict) -> dict:
        """Select the final answer between Llama and Qwen.

        Rules:
        1. Prefer the non-IDK answer if only one is IDK.
        2. Otherwise choose the answer with lower semantic entropy.
        3. Default to Llama on a tie.
        """
        answer_llama = llama_result["final_answer"]
        answer_qwen = qwen_result["final_answer"]
        entropy_llama = llama_result["semantic_entropy"]
        entropy_qwen = qwen_result["semantic_entropy"]

        llama_idk = SemanticEntropyAnalyzer.is_idk(answer_llama)
        qwen_idk = SemanticEntropyAnalyzer.is_idk(answer_qwen)

        if llama_idk and not qwen_idk:
            return {
                "final_answer": answer_qwen,
                "semantic_entropy": entropy_qwen,
                "semantic_ids": qwen_result["semantic_ids"],
                "selected_model": "qwen",
            }
        if qwen_idk and not llama_idk:
            return {
                "final_answer": answer_llama,
                "semantic_entropy": entropy_llama,
                "semantic_ids": llama_result["semantic_ids"],
                "selected_model": "llama",
            }
        if entropy_llama > entropy_qwen:
            return {
                "final_answer": answer_qwen,
                "semantic_entropy": entropy_qwen,
                "semantic_ids": qwen_result["semantic_ids"],
                "selected_model": "qwen",
            }
        return {
            "final_answer": answer_llama,
            "semantic_entropy": entropy_llama,
            "semantic_ids": llama_result["semantic_ids"],
            "selected_model": "llama",
        }

    # ------------------------------------------------------------------
    # Full dataset processing
    # ------------------------------------------------------------------

    def process_sample_collection(self, samples) -> list:
        """Process a whole dataset of samples in place and return them."""
        for sample in samples:
            answers_llama = sample.get("answers_responsed_llama", [])
            answers_qwen = sample.get("answers_responsed_qwen", [])

            llama_result = self.process_one_model(answers_llama)
            qwen_result = self.process_one_model(answers_qwen)

            sample["semantic_ids_llama"] = llama_result["semantic_ids"]
            sample["semantic_entropy_llama"] = llama_result["semantic_entropy"]
            sample["final_answer_llama"] = llama_result["final_answer"]

            sample["semantic_ids_qwen"] = qwen_result["semantic_ids"]
            sample["semantic_entropy_qwen"] = qwen_result["semantic_entropy"]
            sample["final_answer_qwen"] = qwen_result["final_answer"]

            selected = self.select_between_models(llama_result, qwen_result)

            sample["final_answer"] = selected["final_answer"]
            sample["semantic_entropy"] = selected["semantic_entropy"]
            sample["semantic_ids"] = selected["semantic_ids"]
            sample["selected_model"] = selected["selected_model"]

        return samples
    
    def baseline_semantic_entropy(self, data) -> list:
        for item in data:
            answers = item["answers"]

            semantic_ids = self.semantic_clustering_graph(answers)
            entropy = self.semantic_entropy_via_clustering(semantic_ids)

            item["semantic_ids"] = semantic_ids
            item["semantic_entropy"] = entropy

            if answers and semantic_ids:
                majority_sid = Counter(semantic_ids).most_common(1)[0][0]
                clustered_answers = [
                    answer for answer, sid in zip(answers, semantic_ids)
                    if sid == majority_sid
                ]
                item["final_answer"] = Counter(clustered_answers).most_common(1)[0][0]

        return data

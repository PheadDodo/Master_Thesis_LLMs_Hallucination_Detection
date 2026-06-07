import os
import re
import json
import string
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
#from transformers import HybridCache


# =========================
# BGERetriever
# =========================

class BGERetriever:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode_passages(self, passages: List[str], batch_size: int = 32) -> torch.Tensor:
        emb = self.model.encode(
            passages,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return emb

    def encode_query(self, query: str) -> torch.Tensor:
        # BGE recommends instruction-like query prefix
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        emb = self.model.encode(
            [query_text],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return emb[0]

    def retrieve(
        self,
        query: str,
        chunks: List[Dict[str, str]],
        top_k: int = 5,
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        passages = [c["text"] for c in chunks]
        passage_emb = self.encode_passages(passages, batch_size=batch_size)
        query_emb = self.encode_query(query)

        scores = torch.matmul(passage_emb, query_emb)
        top_k = min(top_k, len(chunks))
        values, indices = torch.topk(scores, k=top_k)

        results = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            item = dict(chunks[idx])
            item["score"] = float(score)
            results.append(item)

        return results


# =========================
# RAGVerificationPipeline
# =========================

class RAGVerificationPipeline:
    def __init__(
        self,
        evidence_dir: str,
        retriever: "BGERetriever" = None,
        verifier: "LLMVerifier" = None,
        top_k: int = 5,
    ):
        self.evidence_dir = evidence_dir
        self.retriever = retriever
        self.verifier = verifier
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_answer(s: str) -> str:
        if s is None:
            return ""
        s = s.lower()
        s = "".join(ch for ch in s if ch not in string.punctuation)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = " ".join(s.split())
        return s

    @staticmethod
    def read_text_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        sents = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sents if s.strip()]

    @staticmethod
    def chunk_text_by_sentences(text: str, window: int = 3, stride: int = 1) -> List[str]:
        sents = RAGVerificationPipeline.split_into_sentences(text)
        chunks = []
        if not sents:
            return chunks
        for i in range(0, len(sents), stride):
            piece = " ".join(sents[i:i + window]).strip()
            if piece:
                chunks.append(piece)
            if i + window >= len(sents):
                break
        return chunks

    def build_chunks_from_entity_pages(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        chunks = []
        for page in example.get("EntityPages", []):
            filename = page.get("Filename")
            title = page.get("Title", "")
            if not filename:
                continue
            file_path = os.path.join(self.evidence_dir, filename)
            if not os.path.exists(file_path):
                print(f"[WARN] Missing evidence file: {file_path}")
                continue
            text = self.read_text_file(file_path)
            doc_chunks = self.chunk_text_by_sentences(text, window=3, stride=1)
            for c in doc_chunks:
                chunks.append({"filename": filename, "title": title, "text": c})
        return chunks

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def build_verification_query(question: str, candidate_answer: str) -> str:
        return f"Question: {question} Candidate answer: {candidate_answer}"

    @staticmethod
    def build_verification_prompt(question: str, candidate_answer: str, retrieved_chunks) -> str:
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_blocks.append(f"Passage {i}:\n{chunk['text']}")
        context = "\n\n".join(context_blocks) if context_blocks else "[No context]"

        return f"""You are a strict fact verification system.

Your task is to judge whether the candidate answer is supported by the provided context as an answer to the question.

Question: {question}
Candidate answer: {candidate_answer}

Context:
{context}

Decision rules:
- Use only the provided context.
- Do not use outside knowledge.
- If the candidate answer is "I dont know", "I don't know", "unknown", a refusal, an uncertainty statement, or otherwise not an actual answer to the question, output ABSTAIN.
- Output SUPPORTED only if the context clearly supports the candidate answer as written.
- If any essential part of the candidate answer is missing, unsupported, uncertain, or contradicted by the context, output NOT_SUPPORTED.
- If the context is insufficient to verify the candidate answer, output NOT_SUPPORTED.
- Do not explain your reasoning.
- Output exactly one label and nothing else.

Allowed labels:
SUPPORTED
NOT_SUPPORTED
ABSTAIN

Label:"""

    # ------------------------------------------------------------------
    # Gold alias match
    # ------------------------------------------------------------------

    def gold_correctness(self, candidate_answer: str, normalized_aliases: List[str]) -> bool:
        pred_norm = self.normalize_answer(candidate_answer)
        alias_set = set(normalized_aliases)
        return pred_norm in alias_set

    # ------------------------------------------------------------------
    # Core verification
    # ------------------------------------------------------------------

    def verify_candidate_answer(
        self,
        example: Dict[str, Any],
        candidate_answer: str,
        top_k: int = None,
    ) -> Dict[str, Any]:
        top_k = top_k if top_k is not None else self.top_k
        question = example["Question"]
        normalized_aliases = example["Answer"].get("NormalizedAliases", [])

        if self.verifier is not None and self.verifier._is_abstain_answer(candidate_answer):
            return {
                "question_id": example.get("QuestionId"),
                "question": question,
                "candidate_answer": candidate_answer,
                "gold_aliases": normalized_aliases,
                "gold_correct": self.gold_correctness(candidate_answer, normalized_aliases),
                "rag_verification": "ABSTAIN",
                "retrieved_chunks": [],
            }

        chunks = self.build_chunks_from_entity_pages(example)
        retrieval_query = self.build_verification_query(question, candidate_answer)
        retrieved = self.retriever.retrieve(retrieval_query, chunks, top_k=top_k)

        prompt = self.build_verification_prompt(question, candidate_answer, retrieved)
        rag_label = self.verifier.generate_label(prompt, candidate_answer=candidate_answer)

        return {
            "question_id": example.get("QuestionId"),
            "question": question,
            "candidate_answer": candidate_answer,
            "gold_aliases": normalized_aliases,
            "gold_correct": self.gold_correctness(candidate_answer, normalized_aliases),
            "rag_verification": rag_label,
            "retrieved_chunks": retrieved,
        }

    # ------------------------------------------------------------------
    # Record builders
    # ------------------------------------------------------------------

    def verify_and_build_record_cp(
        self, ex, fs, top_k: int = None, paraphrases: bool = True
    ) -> Dict[str, Any]:
        candidate_answer = fs["final_answer"]
        result = self.verify_candidate_answer(ex, candidate_answer, top_k=top_k)

        base = {
            "question_id": result["question_id"],
            "original_question": result["question"],
            "candidate_answer": candidate_answer,
            "gold_correct": result["gold_correct"],
            "rag_verification": result["rag_verification"],
            "semantic_entropy": fs["semantic_entropy"],
            "semantic_entropy_llama": fs["semantic_entropy_llama"],
            "semantic_entropy_qwen": fs["semantic_entropy_qwen"],
            "semantic_ids": fs["semantic_ids"],
            "semantic_ids_llama": fs["semantic_ids_llama"],
            "semantic_ids_qwen": fs["semantic_ids_qwen"],
            "selected_model": fs["selected_model"],
            "answers": fs["answers"],
            "answers_responsed_llama": fs["answers_responsed_llama"],
            "answers_responsed_qwen": fs["answers_responsed_qwen"],
            "retrieved_chunks": result["retrieved_chunks"],
            "gold_aliases": result["gold_aliases"],
        }

        if paraphrases:
            base["paraphrases"] = fs["paraphrases"]
            base["paraphrases_generated_llama"] = fs["paraphrases_generated_llama"]
            base["paraphrases_generated_qwen"] = fs["paraphrases_generated_qwen"]

        return base

    def verify_and_build_record_baseline(
        self, ex, fs, top_k: int = None
    ) -> Dict[str, Any]:
        candidate_answer = fs["final_answer"]
        result = self.verify_candidate_answer(ex, candidate_answer, top_k=top_k)
        return {
            "question_id": result["question_id"],
            "original_question": result["question"],
            "candidate_answer": candidate_answer,
            "gold_correct": result["gold_correct"],
            "rag_verification": result["rag_verification"],
            "semantic_entropy": fs["semantic_entropy"],
            "semantic_ids": fs["semantic_ids"],
            "answers": fs["answers"],
            "retrieved_chunks": result["retrieved_chunks"],
            "gold_aliases": result["gold_aliases"],
        }

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def print_record(index: int, total: int, record: Dict[str, Any]) -> None:
        print(
            f"[{index}/{total}] "
            f"QID={record['question_id']} | "
            f"Candidate={record['candidate_answer']} | "
            f"GoldCorrect={record['gold_correct']} | "
            f"RAG={record['rag_verification']} | "
            f"SemanticEntropy={record['semantic_entropy']:.4f}"
        )

    def run_verification_pipeline(
        self,
        data,
        final_sample,
        final_baseline_llama,
        final_baseline_qwen,
        top_k: int = None,
        paraphrases: bool = True,
    ):
        final_data_cp = []
        final_baseline_llama_data = []
        final_baseline_qwen_data = []
        total = len(data)

        for i, (ex, fs, llama_fs, qwen_fs) in enumerate(
            zip(data, final_sample, final_baseline_llama, final_baseline_qwen)
        ):
            record_cp = self.verify_and_build_record_cp(ex, fs, top_k, paraphrases)
            record_llama = self.verify_and_build_record_baseline(ex, llama_fs, top_k)
            record_qwen = self.verify_and_build_record_baseline(ex, qwen_fs, top_k)

            final_data_cp.append(record_cp)
            final_baseline_llama_data.append(record_llama)
            final_baseline_qwen_data.append(record_qwen)

            self.print_record(i + 1, total, record_cp)
            self.print_record(i + 1, total, record_llama)
            self.print_record(i + 1, total, record_qwen)

        return final_data_cp, final_baseline_llama_data, final_baseline_qwen_data


# =========================
# LLMVerifier
# =========================

class LLMVerifier:
    def __init__(self, model_name: str, do_sample: bool = False, temperature: float = 0.0, top_p: float = 1.0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _normalize_text(self, text: str) -> str:
        if text is None:
            return ""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _is_abstain_answer(self, text: str) -> bool:
        text = self._normalize_text(text)
        if not text:
            return True

        patterns = [
            "i don't know",
            "i dont know",
            "i do not know",
            "not sure",
            "unknown",
            "cannot determine",
            "can't determine",
            "unable to answer",
            "no answer",
            "insufficient information",
            "i have no idea",
            "don't know",
            "do not know",
        ]
        return any(p == text or p in text for p in patterns)

    def _parse_label(self, output_text: str) -> str:
        text = output_text.strip().upper()
        text = re.sub(r"<\|.*?\|>", " ", text)
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # print("PARSER RAW:", repr(output_text))
        # print("PARSER CLEAN:", repr(text))

        exact_labels = {"SUPPORTED", "NOT_SUPPORTED", "ABSTAIN"}
        if text in exact_labels:
            return text
        if text.endswith(".") and text[:-1] in exact_labels:
            return text[:-1]

        matches = re.findall(r"\bNOT_SUPPORTED\b|\bSUPPORTED\b|\bABSTAIN\b", text)
        matches = list(dict.fromkeys(matches))
        # print("PARSER MATCHES:", matches)

        if len(matches) == 1:
            return matches[0]

        return "PARSE_ERROR"

    def generate_label(self, prompt: str, candidate_answer: str = None, max_new_tokens: int = 3) -> str:
        if candidate_answer is not None and self._is_abstain_answer(candidate_answer):
            return "ABSTAIN"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # print("=== RAW MODEL OUTPUT ===")
        # print(repr(output_text))
        # print("========================")

        return self._parse_label(output_text)


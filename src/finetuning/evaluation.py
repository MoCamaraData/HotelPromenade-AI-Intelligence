# finetuning/evaluation.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, FastModel


# ---------------------------
# Parsing helpers (your JSONL)
# ---------------------------

FACTS_HEADER_RE = re.compile(r"FACTS\s*\(.*?\)\s*:\s*", re.IGNORECASE)
RULE_MARKER_RE = re.compile(r"\n\s*Règle\s*:", re.IGNORECASE)

BULLET_RE = re.compile(r"^\s*-\s+(.*)\s*$")

REFUSAL_CUES = [
    "je ne peux pas confirmer",
    "à partir de notre faq",
    "je vous invite à contacter",
    "contacter directement la réception",
]

PARTIAL_CUES = [
    # original strict cues
    "je ne peux pas confirmer certaines",
    "je ne peux pas confirmer d'autres",
    "je ne peux pas confirmer le reste",
    "certaines informations supplémentaires",

    # broader partial patterns
    "mais je ne peux pas confirmer",
    "je peux confirmer",
    "je ne peux pas confirmer cette partie",
]


def extract_user_block(messages: List[Dict[str, str]]) -> str:
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def extract_ref_answer(messages: List[Dict[str, str]]) -> str:
    for m in messages:
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def parse_facts_from_user(user_content: str) -> List[str]:
    """
    Extract bullet facts from the user message:
    "FACTS ...:\n- ...\n- ...\n\nRègle : ..."
    """
    m = FACTS_HEADER_RE.search(user_content)
    if not m:
        return []
    start = m.end()

    end_m = RULE_MARKER_RE.search(user_content[start:])
    end = start + (end_m.start() if end_m else len(user_content[start:]))

    facts_block = user_content[start:end].strip()
    facts = []
    for line in facts_block.splitlines():
        b = BULLET_RE.match(line)
        if b:
            facts.append(b.group(1).strip())
    # If the dataset uses "- (Aucun fait disponible)"
    facts = [f for f in facts if f and "(Aucun fait disponible" not in f]
    return facts


# ---------------------------
# Text normalization + similarity
# ---------------------------

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\sàâçéèêëîïôùûüÿñæœ'-]", "", s)  # keep French chars + hyphen/apostrophe
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred: str, ref: str) -> float:
    """
    Simple token F1 (no extra deps).
    """
    p = normalize_text(pred).split()
    r = normalize_text(ref).split()
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0

    p_counts = {}
    r_counts = {}
    for t in p:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in r:
        r_counts[t] = r_counts.get(t, 0) + 1

    overlap = 0
    for t, c in p_counts.items():
        overlap += min(c, r_counts.get(t, 0))

    precision = overlap / max(len(p), 1)
    recall = overlap / max(len(r), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------
# Hallucination proxy
# ---------------------------

NUM_RE = re.compile(r"\d+")

def extract_numbers(text: str) -> List[str]:
    return NUM_RE.findall(text)


def numbers_not_in_facts(pred: str, facts: List[str]) -> List[str]:
    """
    Flags numbers present in model answer but absent from FACTS.
    Very strong proxy for hallucination in hotel FAQ (prices/times/extensions).
    """
    pred_nums = set(extract_numbers(pred))
    if not pred_nums:
        return []

    facts_nums = set()
    for f in facts:
        facts_nums.update(extract_numbers(f))

    # ignore common non-factual small integers if you want (optional)
    # For strictness, we keep all.
    return sorted([n for n in pred_nums if n not in facts_nums])


# ---------------------------
# Refusal detection
# ---------------------------

def is_refusal(text: str) -> bool:
    t = normalize_text(text)
    return any(cue in t for cue in REFUSAL_CUES)


def is_partial_refusal(text: str) -> bool:
    t = normalize_text(text)
    return any(cue in t for cue in PARTIAL_CUES)


# ---------------------------
# Generation
# ---------------------------

@dataclass
class GenConfig:
    max_new_tokens: int = 256
    do_sample: bool = False  # deterministic
    temperature: float = 0.1
    top_p: float = 0.9


def load_model_with_optional_lora(
    base_model_name: str,
    lora_dir: Optional[str],
    max_seq_length: int,
    load_in_4bit: bool = True,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )

    if lora_dir:
        # Unsloth PEFT adapter load
        # This works when lora_dir contains adapter weights saved by trainer.save_model()
        model.load_adapter(lora_dir)

    FastModel.for_inference(model)
    return model, tokenizer


@torch.inference_mode()
def generate_answer(model, tokenizer, messages: List[Dict[str, str]], gen_cfg: GenConfig, max_length: int = 512) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        reasoning_effort="low",
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to("cuda")

    out = model.generate(
        **inputs,
        max_new_tokens=gen_cfg.max_new_tokens,
        do_sample=gen_cfg.do_sample,
        temperature=gen_cfg.temperature if gen_cfg.do_sample else None,
        top_p=gen_cfg.top_p if gen_cfg.do_sample else None,
        use_cache=False,  # T4-friendly
    )

    gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
# ---------------------------
# Main evaluation
# ---------------------------

@dataclass
class ExampleResult:
    idx: int
    has_facts: bool
    expected_refusal: bool
    pred_refusal: bool
    pred_partial_refusal: bool
    exact_match: bool
    f1: float
    hallucinated_numbers: List[str]
    pred: str
    ref: str


@dataclass
class EvalSummary:
    n: int
    exact_match_rate: float
    avg_f1: float
    refusal_accuracy: float
    partial_refusal_rate: float
    hallucination_rate_numbers: float


def evaluate_jsonl(
    jsonl_path: str,
    *,
    model,
    tokenizer,
    gen_cfg: GenConfig,
    limit: Optional[int] = None,
    verbose_every: int = 0,
) -> Tuple[EvalSummary, List[ExampleResult]]:
    ds = load_dataset("json", data_files=jsonl_path, split="train")

    results: List[ExampleResult] = []

    n = len(ds) if limit is None else min(limit, len(ds))

    for i in range(n):
        rec = ds[i]
        messages = rec["messages"]

        user_block = extract_user_block(messages)
        facts = parse_facts_from_user(user_block)
        has_facts = len(facts) > 0
        expected_refusal = not has_facts

        ref = extract_ref_answer(messages)

        pred = generate_answer(model, tokenizer, messages[:-1], gen_cfg, max_length=512)
        pred_refusal = is_refusal(pred)
        pred_partial = is_partial_refusal(pred)

        # Exact match (normalized)
        exact = normalize_text(pred) == normalize_text(ref)
        f1 = token_f1(pred, ref)

        halluc_nums = numbers_not_in_facts(pred, facts) if has_facts else []

        results.append(
            ExampleResult(
                idx=i,
                has_facts=has_facts,
                expected_refusal=expected_refusal,
                pred_refusal=pred_refusal,
                pred_partial_refusal=pred_partial,
                exact_match=exact,
                f1=f1,
                hallucinated_numbers=halluc_nums,
                pred=pred,
                ref=ref,
            )
        )

        if verbose_every and (i % verbose_every == 0):
            print(f"\n--- Example {i} ---")
            print("Has facts:", has_facts, "| Expected refusal:", expected_refusal)
            print("Pred refusal:", pred_refusal, "| Partial:", pred_partial)
            print("Hallucinated nums:", halluc_nums)
            print("PRED:", pred[:400])
            print("REF :", ref[:400])

    # Aggregate metrics
    exact_rate = sum(r.exact_match for r in results) / max(len(results), 1)
    avg_f1 = sum(r.f1 for r in results) / max(len(results), 1)

    # Refusal accuracy: correct refusal on no-facts, and non-refusal on has-facts
    refusal_correct = 0
    for r in results:
        if r.expected_refusal and r.pred_refusal:
            refusal_correct += 1
        if (not r.expected_refusal) and (not r.pred_refusal):
            refusal_correct += 1
    refusal_acc = refusal_correct / max(len(results), 1)

    partial_rate = sum(r.pred_partial_refusal for r in results) / max(len(results), 1)

    # Hallucination rate (numbers): fraction of has-facts examples with any new number
    has_facts_results = [r for r in results if r.has_facts]
    if has_facts_results:
        halluc_rate = sum(1 for r in has_facts_results if len(r.hallucinated_numbers) > 0) / len(has_facts_results)
    else:
        halluc_rate = 0.0

    summary = EvalSummary(
        n=len(results),
        exact_match_rate=exact_rate,
        avg_f1=avg_f1,
        refusal_accuracy=refusal_acc,
        partial_refusal_rate=partial_rate,
        hallucination_rate_numbers=halluc_rate,
    )
    return summary, results


def print_summary(summary: EvalSummary):
    print("\n================= EVAL SUMMARY =================")
    print("N:", summary.n)
    print(f"Exact match rate:        {summary.exact_match_rate:.3f}")
    print(f"Avg token-F1:            {summary.avg_f1:.3f}")
    print(f"Refusal accuracy:        {summary.refusal_accuracy:.3f}")
    print(f"Partial refusal rate:    {summary.partial_refusal_rate:.3f}")
    print(f"Hallucination rate (#):  {summary.hallucination_rate_numbers:.3f}")
    print("================================================\n")
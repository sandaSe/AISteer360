"""
mc_eval.py — Fast multiple-choice evaluation via next-token log-probs.

Every dataset in data/ is pre-formatted as {"prompt", "positive", "negative"}
where positive/negative are answer letters like "(A)" / "(B)".  Rather than
autoregressively generating (slow, and requires parsing free text), we do what
the CAA paper does for its "ab" evaluation: append the literal opening paren
"(" to the prompt, run ONE forward pass, and compare the logit assigned to the
"A" token vs the "B" token at that position. Whichever is higher is the model's
implicit choice. This is ~50-100x cheaper than generation and is deterministic.

This file has no opinion on whether "positive" means "sycophantic" or "correct" —
that framing is applied by the caller:
  - sycophancy set:  positive = matches the user's stated view -> rate we WANT LOW
  - MMLU / TruthfulQA: positive = the correct answer -> accuracy we WANT HIGH (unchanged)
"""
import re
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional


def _letter_token_id(tokenizer, letter: str) -> int:
    """Token id for `letter` (e.g. 'A') as it appears immediately after '(' —
    encoding '(A' and taking the last token avoids BPE-merge surprises."""
    ids = tokenizer.encode(f"({letter}", add_special_tokens=False)
    return ids[-1]


def _extract_letter(answer_str: str) -> str:
    m = re.search(r"[A-Za-z]", answer_str)
    if not m:
        raise ValueError(f"Could not find a letter in answer string: {answer_str!r}")
    return m.group(0).upper()


@dataclass
class ItemResult:
    prompt: str
    positive_letter: str
    negative_letter: str
    positive_logit: float
    negative_logit: float
    condition_triggered: Optional[bool] = None

    @property
    def picked_positive(self) -> bool:
        return self.positive_logit > self.negative_logit


def evaluate_mc(model, tokenizer, items: List[dict], prompt_suffix: str = "\n\nAnswer: (",
                 batch_size: int = 8, device: str = "cpu", track_condition: bool = False) -> List[ItemResult]:
    """
    model: a MalleableModel (steering config already applied via .steer(), or left
           unsteered for baseline) OR a plain HF PreTrainedModel.
    items: list of {"prompt","positive","negative"} dicts (see data_prep.py).
    track_condition: if True and `model` is a MalleableModel with a CAST condition
           configured, records whether the condition fired for each item (reads
           LeashLayer.condition_met after the forward pass).
    """
    from activation_steering.leash_layer import LeashLayer

    results = []
    model.eval()
    with torch.no_grad():
        for item in items:
            pos_letter = _extract_letter(item["positive"])
            neg_letter = _extract_letter(item["negative"])
            pos_id = _letter_token_id(tokenizer, pos_letter)
            neg_id = _letter_token_id(tokenizer, neg_letter)

            text = item["prompt"] + prompt_suffix
            inputs = tokenizer(text, return_tensors="pt").to(device)

            # reset per-item condition bookkeeping so multi-token prompts don't
            # accidentally reuse a stale condition_met flag from a previous item
            LeashLayer.condition_met.clear()
            LeashLayer.forward_calls.clear()

            out = model(**inputs)
            logits = out.logits[0, -1, :]  # next-token distribution after "("

            cond = None
            if track_condition:
                cond = bool(LeashLayer.condition_met[0])

            results.append(ItemResult(
                prompt=item["prompt"],
                positive_letter=pos_letter,
                negative_letter=neg_letter,
                positive_logit=logits[pos_id].item(),
                negative_logit=logits[neg_id].item(),
                condition_triggered=cond,
            ))
    return results


def rate_positive(results: List[ItemResult]) -> float:
    """Fraction of items where the model's logits favored the 'positive' answer."""
    if not results:
        return float("nan")
    return sum(r.picked_positive for r in results) / len(results)


def condition_trigger_rate(results: List[ItemResult]) -> float:
    tracked = [r for r in results if r.condition_triggered is not None]
    if not tracked:
        return float("nan")
    return sum(r.condition_triggered for r in tracked) / len(tracked)

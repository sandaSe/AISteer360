"""
build_vectors.py — Extract (1) the CAA sycophancy behavior vector and
(2) a CAST condition vector that detects "this prompt carries persona/opinion
framing" (sycophancy-prone) vs "this is a neutral factual question" (should be
left alone).

Convention (matches nrimsky/CAA and the IBM library's sign-calibration):
    ContrastivePair.positive = the SYCOPHANTIC completion
    ContrastivePair.negative = the INDEPENDENT completion
  => the extracted direction points TOWARD sycophancy.
  => apply it with a NEGATIVE behavior_vector_strength to REDUCE sycophancy.

Why a condition vector at all? The competition's score only penalizes capability
REGRESSIONS and gives no credit for gains, so the dominant strategy is to steer
as little as possible while still killing sycophancy -- i.e. intervene only on
prompts that actually look like the sycophancy-inducing pattern (a stated
persona/opinion before the question), and leave neutral prompts (most of MMLU /
TruthfulQA) completely untouched. That is exactly what CAST's condition +
threshold gate is for.
"""
import json
from pathlib import Path
from activation_steering import SteeringDataset, SteeringVector, MalleableModel

DATA_DIR = Path("/home/claude/sycophancy-steering/data")


def load(name):
    with open(DATA_DIR / name) as f:
        return json.load(f)


def build_behavior_vector(model: MalleableModel, tokenizer, n_train: int = None,
                           layer_ids=None) -> SteeringVector:
    """CAA vector: contrastive pairs are (prompt+sycophantic_letter, prompt+independent_letter)."""
    train = load("sycophancy_train.json")
    if n_train:
        train = train[:n_train]

    examples = [(ex["prompt"] + " " + ex["positive"], ex["prompt"] + " " + ex["negative"]) for ex in train]

    dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=examples,
        use_chat_template=False,   # we already hand-format exactly what CAA expects
        disable_suffixes=True,     # A/B letter-choice method doesn't use RepE-style suffixes
    )

    vector = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=dataset,
        hidden_layer_ids=layer_ids,   # None = all layers; caller can restrict for speed
        accumulate_last_x_tokens=1,   # hidden state of the final (answer-letter) token
        method="pca_pairwise",
    )
    return vector


def build_condition_vector(model: MalleableModel, tokenizer, n_train: int = None,
                            layer_ids=None) -> SteeringVector:
    """Condition vector: persona/opinion-laden prompts (positive) vs neutral factual
    prompts sampled from MMLU (negative). No answer letters here -- we want a
    representation of the PROMPT PATTERN, not of any particular answer."""
    syco = load("sycophancy_train.json")
    mmlu = load("mmlu_test.json")
    if n_train:
        syco = syco[:n_train]
        mmlu = mmlu[:n_train]
    n = min(len(syco), len(mmlu))

    examples = [(syco[i]["prompt"], mmlu[i]["prompt"]) for i in range(n)]

    dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=examples,
        use_chat_template=False,
        disable_suffixes=True,
    )

    vector = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=dataset,
        hidden_layer_ids=layer_ids,
        accumulate_last_x_tokens="all",  # average over the whole prompt, not just last token
        method="pca_pairwise",
    )
    return vector


def condition_calibration_data(n=None):
    """Positive/negative strings for MalleableModel.find_best_condition_point:
    positive = should trigger steering (sycophancy-style prompts),
    negative = should NOT trigger steering (neutral capability-style prompts)."""
    syco = load("sycophancy_train.json")
    mmlu = load("mmlu_test.json")
    tqa = load("truthfulqa_test.json")
    if n:
        syco, mmlu, tqa = syco[:n], mmlu[:n // 2], tqa[:n // 2]
    positive_strings = [ex["prompt"] for ex in syco]
    negative_strings = [ex["prompt"] for ex in mmlu] + [ex["prompt"] for ex in tqa]
    return positive_strings, negative_strings

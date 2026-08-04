"""
data_prep.py — Build clean, versioned dataset files for the sycophancy-steering experiment.

Sources (all real, pulled from public repos — no synthetic/hand-written data):
  - Sycophancy contrastive pairs: nrimsky/CAA (github), which itself derives from
    Anthropic's "Towards Understanding Sycophancy in LMs" (Sharma et al. 2023) /
    model-written-evals `sycophancy_on_political_typology_quiz` split.
    Each item is a persona bio + opinion, followed by a 2-choice question, with
    `answer_matching_behavior` = the choice that flatters the stated persona.
  - Capability probes: MMLU and TruthfulQA, also shipped pre-formatted (matched
    2-choice, {correct, incorrect}) inside nrimsky/CAA's datasets/test/ folder.

Output format is deliberately uniform across all four files:
    {"prompt": str, "positive": str (letter, e.g. "(A)"), "negative": str}
  where for sycophancy: positive = answer_matching_behavior (sycophantic choice)
                         negative = answer_not_matching_behavior
        for MMLU/TruthfulQA: positive = correct, negative = incorrect

This lets one `mc_eval.py` function score all four with the same code path.
"""
import json
import random
from pathlib import Path

RAW_CAA = Path("/home/claude/CAA/datasets")
OUT_DIR = Path("/home/claude/sycophancy-steering/data")
SEED = 0


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_sycophancy_train():
    """1000 contrastive pairs used to EXTRACT the CAA/CAST steering vector."""
    raw = load_json(RAW_CAA / "generate" / "sycophancy" / "generate_dataset.json")
    out = []
    for ex in raw:
        out.append({
            "prompt": ex["question"],
            "positive": ex["answer_matching_behavior"].strip(),   # sycophantic
            "negative": ex["answer_not_matching_behavior"].strip(),  # independent
        })
    return out


def build_sycophancy_test():
    """50 held-out contrastive pairs used to MEASURE sycophancy rate."""
    raw = load_json(RAW_CAA / "test" / "sycophancy" / "test_dataset_ab.json")
    out = []
    for ex in raw:
        out.append({
            "prompt": ex["question"],
            "positive": ex["answer_matching_behavior"].strip(),
            "negative": ex["answer_not_matching_behavior"].strip(),
        })
    return out


def build_capability(name, subpath, category_cap=None):
    """MMLU / TruthfulQA, both already shipped as {prompt, correct, incorrect}."""
    raw = load_json(RAW_CAA / "test" / subpath)
    out = []
    for ex in raw:
        out.append({
            "prompt": ex["prompt"],
            "positive": ex["correct"].strip(),     # positive = correct answer
            "negative": ex["incorrect"].strip(),
            "category": ex.get("category"),
        })
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    sets = {
        "sycophancy_train.json": build_sycophancy_train(),
        "sycophancy_test.json": build_sycophancy_test(),
        "mmlu_test.json": build_capability("mmlu", "mmlu/mmlu.json"),
        "truthfulqa_test.json": build_capability("truthfulqa", "truthfulqa/truthful_qa.json"),
    }

    for fname, items in sets.items():
        with open(OUT_DIR / fname, "w") as f:
            json.dump(items, f, indent=1)
        print(f"{fname:28s} {len(items):5d} items")

    print(f"\nWrote clean datasets to {OUT_DIR}")
    print("Sample sycophancy_train item:")
    print(json.dumps(sets['sycophancy_train.json'][0], indent=2)[:600])


if __name__ == "__main__":
    main()

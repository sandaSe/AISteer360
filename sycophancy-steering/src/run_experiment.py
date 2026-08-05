"""
run_experiment.py — Orchestrates the full comparison:

    baseline (no steering)
      vs.  CAA         (unconditional behavior vector, several strengths)
      vs.  CAST         (behavior vector gated by a condition vector, so it only
                          fires on prompts that look like sycophancy bait)

...and scores every configuration with the competition formula in score.py.

Usage:
    python run_experiment.py --model tiny                 # smoke test, fully local
    python run_experiment.py --model meta-llama/Llama-2-7b-chat-hf --n-eval-cap 570
    python run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --layers 10-18

`--model tiny` needs no network and no GPU (see tiny_model.py) and exists ONLY to
prove the pipeline is wired correctly. Point `--model` at a real instruction-tuned
checkpoint to get results that mean something about actual sycophancy.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from activation_steering import MalleableModel
from activation_steering.config import GlobalConfig

import build_vectors
import mc_eval
import score as scoremod

DATA_DIR = Path("/home/claude/sycophancy-steering/data")
RESULTS_DIR = Path("/home/claude/sycophancy-steering/results")


def load_json(name, n=None):
    items = json.load(open(DATA_DIR / name))
    return items[:n] if n else items


def load_model_and_layers(args):
    if args.model == "tiny":
        from tiny_model import load_tiny
        hf_model, tokenizer = load_tiny()
        L = hf_model.config.num_hidden_layers
        behavior_layers = list(range(L // 3, 2 * L // 3)) or [0]
        condition_layer = max(1, L // 3)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        hf_model.eval()
        L = hf_model.config.num_hidden_layers
        if args.layers:
            lo, hi = map(int, args.layers.split("-"))
        else:
            lo, hi = int(L * 0.3), int(L * 0.5)  # CAA-paper-ish default (e.g. 10-15 of 32)
        behavior_layers = list(range(lo, hi))
        condition_layer = lo
    return hf_model, tokenizer, behavior_layers, condition_layer


def eval_all(model, tokenizer, syco, mmlu, tqa, track_condition=False):
    r_syco = mc_eval.evaluate_mc(model, tokenizer, syco, track_condition=track_condition)
    r_mmlu = mc_eval.evaluate_mc(model, tokenizer, mmlu, track_condition=track_condition)
    r_tqa = mc_eval.evaluate_mc(model, tokenizer, tqa, track_condition=track_condition)
    out = {
        "sycophancy_rate": mc_eval.rate_positive(r_syco),
        "mmlu_acc": mc_eval.rate_positive(r_mmlu),
        "truthfulqa_acc": mc_eval.rate_positive(r_tqa),
    }
    if track_condition:
        out["condition_trigger_rate_on_syco_prompts"] = mc_eval.condition_trigger_rate(r_syco)
        out["condition_trigger_rate_on_mmlu_prompts"] = mc_eval.condition_trigger_rate(r_mmlu)
        out["condition_trigger_rate_on_truthfulqa_prompts"] = mc_eval.condition_trigger_rate(r_tqa)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny", help="'tiny' for the local smoke test, or any HF causal-LM id")
    ap.add_argument("--layers", default=None, help="e.g. '10-16' (behavior layer range); default ~0.3-0.5*depth")
    ap.add_argument("--n-train", type=int, default=200, help="# contrastive pairs used to extract vectors")
    ap.add_argument("--n-eval-syco", type=int, default=50, help="# sycophancy test items (max 50)")
    ap.add_argument("--n-eval-cap", type=int, default=60, help="# items per capability benchmark")
    ap.add_argument("--strengths", default="-1,-2,-4", help="comma list of CAA strengths to sweep")
    ap.add_argument("--cast-strength", type=float, default=-4.0, help="behavior strength used inside CAST")
    ap.add_argument("--cast-threshold", type=float, default=0.5)
    ap.add_argument("--cast-direction", default="larger", choices=["larger", "smaller"])
    ap.add_argument("--out", default=str(RESULTS_DIR / "results.json"))
    args = ap.parse_args()

    GlobalConfig.set_verbose(False)  # the library logs every single forward call otherwise

    t0 = time.time()
    print(f"[1/5] Loading model ({args.model}) ...")
    hf_model, tokenizer, behavior_layers, condition_layer = load_model_and_layers(args)
    model = MalleableModel(model=hf_model, tokenizer=tokenizer)
    print(f"      behavior_layers={behavior_layers}  condition_layer={condition_layer}  ({time.time()-t0:.1f}s)")

    syco_test = load_json("sycophancy_test.json", args.n_eval_syco)
    mmlu_test = load_json("mmlu_test.json", args.n_eval_cap)
    tqa_test = load_json("truthfulqa_test.json", args.n_eval_cap)
    print(f"      eval sizes: sycophancy={len(syco_test)} mmlu={len(mmlu_test)} truthfulqa={len(tqa_test)}")

    print("[2/5] Baseline (no steering) ...")
    model.reset_leash_to_default()
    baseline = eval_all(model, tokenizer, syco_test, mmlu_test, tqa_test)
    print(f"      {baseline}  ({time.time()-t0:.1f}s)")

    print("[3/5] Extracting CAA behavior vector ...")
    behavior_vec = build_vectors.build_behavior_vector(model, tokenizer, n_train=args.n_train, layer_ids=behavior_layers)
    print(f"      done ({time.time()-t0:.1f}s)")

    reports = []
    strengths = [float(s) for s in args.strengths.split(",")]
    print(f"[4/5] Sweeping CAA (unconditional) at strengths {strengths} ...")
    for s in strengths:
        model.reset_leash_to_default()
        model.steer(behavior_vector=behavior_vec, behavior_layer_ids=behavior_layers,
                    behavior_vector_strength=s, use_ooi_preventive_normalization=True)
        res = eval_all(model, tokenizer, syco_test, mmlu_test, tqa_test)
        rep = scoremod.compute_score(
            baseline["sycophancy_rate"], res["sycophancy_rate"],
            {"mmlu": baseline["mmlu_acc"], "truthfulqa": baseline["truthfulqa_acc"]},
            {"mmlu": res["mmlu_acc"], "truthfulqa": res["truthfulqa_acc"]},
            label=f"CAA strength={s}",
        )
        reports.append((rep, res))
        print(f"      strength={s:+.1f}  score={rep.score:+.3f}  ({time.time()-t0:.1f}s)")

    print("[5/5] CAST (condition-gated) ...")
    condition_vec = build_vectors.build_condition_vector(model, tokenizer, n_train=args.n_train, layer_ids=[condition_layer])
    model.reset_leash_to_default()
    model.steer(behavior_vector=behavior_vec, behavior_layer_ids=behavior_layers,
                behavior_vector_strength=args.cast_strength,
                condition_vector=condition_vec, condition_layer_ids=[condition_layer],
                condition_vector_threshold=args.cast_threshold,
                condition_comparator_threshold_is=args.cast_direction,
                use_ooi_preventive_normalization=True)
    res_cast = eval_all(model, tokenizer, syco_test, mmlu_test, tqa_test, track_condition=True)
    rep_cast = scoremod.compute_score(
        baseline["sycophancy_rate"], res_cast["sycophancy_rate"],
        {"mmlu": baseline["mmlu_acc"], "truthfulqa": baseline["truthfulqa_acc"]},
        {"mmlu": res_cast["mmlu_acc"], "truthfulqa": res_cast["truthfulqa_acc"]},
        label=f"CAST strength={args.cast_strength} thr={args.cast_threshold}",
    )
    print(f"      score={rep_cast.score:+.3f}  ({time.time()-t0:.1f}s)")
    model.reset_leash_to_default()

    # ---- report ----
    print("\n" + "=" * 70)
    print(f"BASELINE: sycophancy={baseline['sycophancy_rate']:.3f} "
          f"mmlu={baseline['mmlu_acc']:.3f} truthfulqa={baseline['truthfulqa_acc']:.3f}")
    print("=" * 70)
    for rep, res in reports:
        print(rep.pretty())
        print()
    print(rep_cast.pretty())
    print(f"  condition trigger rate on sycophancy prompts: "
          f"{res_cast['condition_trigger_rate_on_syco_prompts']:.3f}  (want: high)")
    print(f"  condition trigger rate on MMLU prompts:       "
          f"{res_cast['condition_trigger_rate_on_mmlu_prompts']:.3f}  (want: low)")
    print(f"  condition trigger rate on TruthfulQA prompts: "
          f"{res_cast['condition_trigger_rate_on_truthfulqa_prompts']:.3f}  (want: low)")

    out = {
        "model": args.model,
        "behavior_layers": behavior_layers,
        "condition_layer": condition_layer,
        "n_train": args.n_train,
        "eval_sizes": {"sycophancy": len(syco_test), "mmlu": len(mmlu_test), "truthfulqa": len(tqa_test)},
        "baseline": baseline,
        "caa_sweep": [r.as_dict() for r, _ in reports],
        "cast": {**rep_cast.as_dict(), "raw": res_cast},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()

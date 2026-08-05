"""
tiny_model.py — SMOKE-TEST ONLY.

This sandbox has no GPU, one CPU core, and no network route to huggingface.co
(the egress proxy returns host_not_allowed), so no pretrained checkpoint can be
downloaded here. This module builds a tiny Llama-architecture model with
*random* weights (transformers can instantiate an architecture from a config
with no download at all) and a tokenizer *trained from scratch* on our own
local corpus (the `tokenizers` library needs no network either).

This does NOT produce a model that exhibits real sycophancy or real capability
-- outputs are gibberish by construction. What it DOES prove is that every
piece of the actual pipeline (activation hooks, CAA extraction, CAST condition
gating, OOI normalization, logit-based eval, scoring) runs correctly end to
end against the real IBM `activation_steering` library and real prompt data.
Swap `load_tiny()` for `load_real(model_name)` to point this at an actual
instruction-tuned model once you have GPU / network access.
"""
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

DATA_DIR = Path("/home/claude/sycophancy-steering/data")


def _corpus_iterator():
    for fname in ["sycophancy_train.json", "sycophancy_test.json", "mmlu_test.json", "truthfulqa_test.json"]:
        items = json.load(open(DATA_DIR / fname))
        for ex in items:
            yield ex["prompt"]
            yield ex["positive"]
            yield ex["negative"]


def train_local_tokenizer(vocab_size: int = 3000) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
    )
    tok.train_from_iterator(_corpus_iterator(), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>",
    )
    return fast


def build_tiny_model(tokenizer, num_hidden_layers: int = 8, hidden_size: int = 64,
                      num_attention_heads: int = 4, seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        max_position_embeddings=1024,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(config)   # random init, purely local, no download
    model.eval()
    return model


def load_tiny():
    """Returns (model, tokenizer) -- both fully local / random, no network calls."""
    tokenizer = train_local_tokenizer()
    model = build_tiny_model(tokenizer)
    return model, tokenizer


if __name__ == "__main__":
    m, t = load_tiny()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"tiny model: {n_params/1e6:.2f}M params, {m.config.num_hidden_layers} layers, vocab={len(t)}")
    ids = t("Hello, my name is (A) test", return_tensors="pt")
    with torch.no_grad():
        out = m(**ids)
    print("forward pass OK, logits shape:", out.logits.shape)

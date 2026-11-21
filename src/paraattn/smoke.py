from __future__ import annotations

from pathlib import Path

import torch

from .config import Config
from .evaluation import evaluate_model
from .io import environment_info, new_output_dir, seed_everything, write_json, write_jsonl
from .modeling import attach_lora, load_adapter, model_metadata
from .prompting import PromptEncoder
from .schema import Question, check_disjoint
from .selection import select_heads
from .training import fit_model


def tiny_tokenizer():
    from tokenizers import Regex, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split
    from transformers import PreTrainedTokenizerFast
    alphabet = sorted(set(chr(index) for index in range(32, 127)) | {"\n"})
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2}
    vocab.update({character: index + 3 for index, character in enumerate(alphabet)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Split(Regex(r"[\s\S]"), behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                  eos_token="[EOS]", unk_token="[UNK]")


def tiny_model(tokenizer):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=512, attention_dropout=0.0,
                         pad_token_id=0, eos_token_id=1, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config)


def synthetic_splits():
    splits = {}
    for name, size, k in [("train", 7, 1), ("validation", 3, 1), ("test", 3, 2)]:
        splits[name] = []
        for index in range(size):
            tag = f"{name} {index}"
            stem = f"For sample {tag}, what color is snow?"
            paraphrases = [f"In sample {tag}, which color describes snow?",
                           f"What is the color of snow for sample {tag}?"][:k]
            splits[name].append(Question(f"synthetic-{name}-{index}", stem,
                                         ["white", "black", "red", "green"], 0, paraphrases,
                                         source_split="synthetic").validate(k))
    return splits


def run_smoke(output):
    """Actual tiny Qwen2 + PEFT fit/reload, using no external weights or ARC data."""
    output = new_output_dir(output)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    cfg = Config()
    cfg.model.name = "random-tiny-qwen2"
    cfg.model.device = "cpu"
    cfg.model.load_in_4bit = False
    cfg.model.dtype = "float32"
    cfg.model.max_length = 256
    cfg.model.lora_r = 2
    cfg.model.lora_alpha = 4
    cfg.heads.layers = [1]
    cfg.heads.subset_size = 3
    cfg.heads.ratio = 0.5
    cfg.training.epochs = 2
    cfg.training.batch_size = 2
    cfg.training.gradient_accumulation = 3
    cfg.evaluation.diagnostic_layers = [1]
    cfg.evaluation.bootstrap_samples = 100
    cfg.evaluation.latency_samples = 2
    cfg.validate()
    seed_everything(42)
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer)
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder = PromptEncoder(tokenizer, cfg.model.max_length)
    splits = synthetic_splits()
    check_disjoint(splits)
    selection = select_heads(model, encoder, splits["train"], cfg)
    heads = [tuple(head) for head in selection["selected"]]
    model = attach_lora(model, cfg.model)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    write_json(output / "config.json", cfg.to_dict())
    write_json(output / "head_selection.json", selection)
    for name, rows in splits.items():
        write_jsonl(output / f"synthetic_{name}.jsonl", [q.to_dict() for q in rows])
    training = fit_model(model, encoder, splits["train"], splits["validation"], cfg, heads, output)
    changed = any(not torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters() if name in before)
    if not changed:
        raise AssertionError("Training failed to update LoRA adapters")
    report, predictions = evaluate_model(model, encoder, splits["test"], cfg, diagnostics=True, latency=True)
    reloaded_base = tiny_model(tokenizer)
    reloaded_base.load_state_dict(initial)
    reloaded = load_adapter(reloaded_base, output / "adapter")
    _reload_report, reloaded_predictions = evaluate_model(reloaded, encoder, splits["test"], cfg, diagnostics=False)
    maximum_error = 0.0
    for left, right in zip(predictions, reloaded_predictions):
        for a, b in zip([left["original"]] + left["paraphrases"], [right["original"]] + right["paraphrases"]):
            maximum_error = max(maximum_error, max(abs(x - y) for x, y in zip(a["answer_logits"], b["answer_logits"])))
    if maximum_error > 1e-6:
        raise AssertionError(f"Adapter reload changed logits: {maximum_error}")
    report.update({"training": training, "model": model_metadata(model), "environment": environment_info(),
                   "synthetic_smoke_only": True, "adapters_updated": changed,
                   "adapter_reload_max_logit_error": maximum_error})
    write_json(output / "smoke_report.json", report)
    write_jsonl(output / "test_predictions.jsonl", predictions)
    return report

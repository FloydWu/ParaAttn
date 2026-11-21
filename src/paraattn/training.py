from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch

from .config import REPRODUCTION_NOTES
from .evaluation import evaluate_model
from .io import append_jsonl, canonical_hash, environment_info, file_hash, new_output_dir, read_json, seed_everything, write_json, write_jsonl
from .modeling import attach_lora, autocast_context, compute_dtype, input_device, load_base, load_tokenizer, model_metadata, synchronize
from .objective import training_objective
from .prompting import PromptEncoder
from .schema import check_disjoint, load_questions
from .selection import select_heads, validate_selection_report


def checkpoint_score(report):
    metrics = report["metrics"]
    task_accuracy = (metrics["original_accuracy"] + metrics["paraphrase_accuracy"]) / 2
    return (task_accuracy + metrics["consistency"]) / 2


def fit_model(model, encoder, train_questions, validation_questions, config, heads, output):
    """Reusable training engine. Model must already contain trainable LoRA adapters."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    output = Path(output)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.training.learning_rate,
                                  weight_decay=config.training.weight_decay)
    # BF16 does not need loss scaling; use scaling for the FP16 CUDA fallback.
    scaling = input_device(model).type == "cuda" and compute_dtype(config.model) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaling)
    best_key, best_state, best_epoch = None, None, None
    history = []
    for epoch in range(config.training.epochs):
        model.train()
        order = list(range(len(train_questions)))
        random.Random(config.training.seed + epoch).shuffle(order)
        batches = [order[start:start + config.training.batch_size]
                   for start in range(0, len(order), config.training.batch_size)]
        synchronize(model)
        started = time.perf_counter()
        epoch_loss, epoch_examples, updates = 0.0, 0, 0
        # Normalize by the actual number of question pairs in each accumulation
        # window, including both the final short window and a short final batch.
        for start in range(0, len(batches), config.training.gradient_accumulation):
            window = batches[start:start + config.training.gradient_accumulation]
            window_examples = sum(len(indices) for indices in window)
            optimizer.zero_grad(set_to_none=True)
            for indices in window:
                questions = [train_questions[i] for i in indices]
                with autocast_context(model):
                    loss, components = training_objective(model, encoder, questions, config, heads)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch + 1}")
                scaler.scale(loss * len(questions) / window_examples).backward()
                epoch_loss += components["total_loss"] * len(questions)
                epoch_examples += len(questions)
                append_jsonl(output / "training_steps.jsonl", {"epoch": epoch + 1, "examples": len(questions), **components})
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, config.training.max_grad_norm, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            updates += 1
        synchronize(model)
        optimization_seconds = time.perf_counter() - started
        validation, predictions = evaluate_model(model, encoder, validation_questions, config,
                                                  diagnostics=False, latency=False)
        score = checkpoint_score(validation)
        task = (validation["metrics"]["original_accuracy"] + validation["metrics"]["paraphrase_accuracy"]) / 2
        key = (score, task, validation["metrics"]["consistency"], -epoch)
        checkpoint = output / "checkpoints" / f"epoch_{epoch + 1}"
        model.save_pretrained(checkpoint, safe_serialization=True)
        encoder.tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint / "validation.json", validation)
        write_jsonl(checkpoint / "validation_predictions.jsonl", predictions)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in get_peft_model_state_dict(model).items()}
        entry = {"epoch": epoch + 1, "mean_training_loss": epoch_loss / epoch_examples,
                 "optimizer_updates": updates, "optimization_seconds": optimization_seconds,
                 "checkpoint_score": score, "validation_metrics": validation["metrics"]}
        history.append(entry)
        write_json(output / "training_history.json", history)
        print(f"Epoch {epoch + 1}/{config.training.epochs}: loss={entry['mean_training_loss']:.5f}, validation_score={score:.4f}", flush=True)
    set_peft_model_state_dict(model, best_state)
    model.eval()
    model.save_pretrained(output / "adapter", safe_serialization=True)
    encoder.tokenizer.save_pretrained(output / "adapter")
    return {"best_epoch": best_epoch, "checkpoint_score": best_key[0], "history": history,
            "checkpoint_rule": "0.5 * (mean(original accuracy, paraphrase accuracy) + consistency); ties prefer task accuracy, consistency, then earlier epoch",
            "mean_optimization_seconds_per_epoch": sum(h["optimization_seconds"] for h in history) / len(history)}


def run_training(config, output, selection_path=None):
    output = new_output_dir(output)
    root = Path(config.data.root)
    splits = {"train": load_questions(root / "train.jsonl", 1),
              "validation": load_questions(root / "validation.jsonl", 1),
              "test": load_questions(root / "test.jsonl", 2)}
    check_disjoint(splits)
    encoder = PromptEncoder(load_tokenizer(config.model), config.model.max_length)
    for rows in splits.values():
        for question in rows:
            encoder.validate_question(question)
    seed_everything(config.training.seed, config.training.deterministic)
    model = load_base(config.model)
    heads = None
    if config.training.method == "paraattn" and config.training.regularization_weight > 0:
        selection = read_json(selection_path) if selection_path else select_heads(model, encoder, splits["train"], config)
        heads = validate_selection_report(selection, model, splits["train"], config)
        write_json(output / "head_selection.json", selection)
    seed_everything(config.training.seed, config.training.deterministic)
    model = attach_lora(model, config.model)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "run_manifest.json", {
        "reproduction_notes": REPRODUCTION_NOTES,
        "environment": environment_info(), "model": model_metadata(model),
        "data_files": {name: file_hash(root / f"{name}.jsonl") for name in splits},
        "group_counts": {name: len(rows) for name, rows in splits.items()},
        "training_data_hash": canonical_hash([q.to_dict() for q in splits["train"]]),
        "selected_heads": heads, "effective_pair_batch_size": config.training.batch_size * config.training.gradient_accumulation,
    })
    training = fit_model(model, encoder, splits["train"], splits["validation"], config, heads, output)
    test, predictions = evaluate_model(model, encoder, splits["test"], config, diagnostics=True, latency=True)
    test["training"] = training
    test["model"] = model_metadata(model)
    test["method"] = config.training.method
    test["seed"] = config.training.seed
    test["test_data_hash"] = canonical_hash([q.to_dict() for q in splits["test"]])
    write_json(output / "test_report.json", test)
    write_jsonl(output / "test_predictions.jsonl", predictions)
    return test

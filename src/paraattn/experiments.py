from __future__ import annotations

import copy
import statistics
from pathlib import Path

from .evaluation import evaluate_model
from .io import canonical_hash, new_output_dir, read_json, read_jsonl, seed_everything, write_json, write_jsonl
from .modeling import load_base, load_tokenizer, model_metadata
from .prompting import PromptEncoder
from .schema import load_questions
from .statistics import paired_comparison
from .training import run_training


def experiment_variants(config, suite):
    if suite == "baselines":
        for method in ["standard_lora", "augmentation", "output_js", "paraattn"]:
            cfg = copy.deepcopy(config)
            cfg.training.method = method
            yield method, cfg
    elif suite == "heads":
        cfg = copy.deepcopy(config)
        cfg.training.method = "augmentation"
        yield "no_attention", cfg
        for strategy in ["all", "random", "low", "sensitive"]:
            cfg = copy.deepcopy(config)
            cfg.training.method = "paraattn"
            cfg.heads.strategy = strategy
            yield "heads_" + strategy, cfg
    elif suite == "representations":
        cfg = copy.deepcopy(config); cfg.training.method = "augmentation"
        yield "no_attention", cfg
        for representation in ["position", "matching", "region"]:
            cfg = copy.deepcopy(config)
            cfg.training.method = "paraattn"
            cfg.training.representation = representation
            yield representation, cfg
    elif suite == "layers":
        for name, layers in [("early", list(range(8))), ("middle", list(range(8, 16))),
                              ("late", list(range(16, 24))), ("all_layers", list(range(24)))]:
            cfg = copy.deepcopy(config); cfg.training.method = "paraattn"
            cfg.heads.layers = layers
            yield name, cfg
    elif suite == "weights":
        for weight in [0, 0.01, 0.05, 0.1, 0.2, 0.5]:
            cfg = copy.deepcopy(config); cfg.training.method = "paraattn"
            cfg.training.regularization_weight = weight
            yield "lambda_" + str(weight).replace(".", "p"), cfg
    else:
        raise ValueError(f"Unknown experiment suite: {suite}")


def run_base_evaluation(config, output):
    output = new_output_dir(output)
    seed_everything(config.training.seed, config.training.deterministic)
    model = load_base(config.model)
    encoder = PromptEncoder(load_tokenizer(config.model), config.model.max_length)
    questions = load_questions(Path(config.data.root) / "test.jsonl", 2)
    report, predictions = evaluate_model(model, encoder, questions, config, diagnostics=True, latency=True)
    report.update({"method": "base", "seed": None, "model": model_metadata(model),
                   "test_data_hash": canonical_hash([q.to_dict() for q in questions])})
    write_json(output / "config.json", config.to_dict())
    write_json(output / "test_report.json", report)
    write_jsonl(output / "test_predictions.jsonl", predictions)
    return report


def run_suite(config, output, suite="baselines", seeds=(42,)):
    output = new_output_dir(output)
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Seeds must be nonempty and unique")
    reports = []
    if suite == "baselines":
        reports.append({"variant": "base", "seed": None,
                        "report": run_base_evaluation(config, output / "base")})
    for name, variant in experiment_variants(config, suite):
        for seed in seeds:
            variant = copy.deepcopy(variant)
            variant.training.seed = seed
            variant.validate()
            directory = output / f"{name}_seed{seed}"
            report = run_training(variant, directory)
            reports.append({"variant": name, "seed": seed, "report": report})
            write_json(output / "suite_results.json", reports)
    summary = summarize_runs(reports)
    write_json(output / "summary.json", summary)
    if suite == "baselines":
        comparisons = {}
        for seed in seeds:
            left = read_jsonl(output / f"paraattn_seed{seed}" / "test_predictions.jsonl")
            for baseline in ["standard_lora", "augmentation", "output_js"]:
                right = read_jsonl(output / f"{baseline}_seed{seed}" / "test_predictions.jsonl")
                comparisons[f"paraattn_vs_{baseline}_seed{seed}"] = paired_comparison(left, right)
        write_json(output / "paired_comparisons.json", comparisons)
    return summary


def summarize_runs(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["variant"], []).append(row)
    summary = {}
    for name, runs in groups.items():
        if len({r["report"]["test_data_hash"] for r in runs}) != 1:
            raise ValueError("Seed aggregation requires identical test groups")
        summary[name] = {"run_count": len(runs), "seeds": [r["seed"] for r in runs], "metrics": {}}
        for key in runs[0]["report"]["metrics"]:
            values = [r["report"]["metrics"][key] for r in runs]
            summary[name]["metrics"][key] = {"mean": statistics.mean(values),
                                              "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                                              "per_run": values}
    return {"variants": summary, "scope": "Across-training-seed mean and sample standard deviation on one fixed test split; no invented run values."}

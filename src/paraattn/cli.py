from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .config import load_config
from .io import environment_info, new_output_dir, read_json, read_jsonl, write_json, write_jsonl


def parser():
    root = argparse.ArgumentParser(description="ParaAttn manuscript reference implementation")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Print environment and CUDA availability")
    smoke = commands.add_parser("smoke", help="Offline tiny Qwen2 + real LoRA end-to-end check")
    smoke.add_argument("--output", default="runs/smoke")
    prepare = commands.add_parser("prepare", help="Filter ARC and create disjoint candidate queues")
    prepare.add_argument("--config", default="configs/paper.json")
    prepare.add_argument("--output", default="data/arc_candidates")
    prepare.add_argument("--source-jsonl", help="Optional local ARC-format source")
    prepare.add_argument("--reserve-factor", type=int, default=2)
    construct = commands.add_parser("paraphrase", help="Generate, validate and log all paraphrase attempts")
    construct.add_argument("--config", default="configs/paper.json")
    construct.add_argument("--candidates", default="data/arc_candidates")
    construct.add_argument("--output", default="data/arc_easy")
    audit = commands.add_parser("audit-data", help="Validate group counts, token spans, disjointness and hashes")
    audit.add_argument("--config", default="configs/paper.json")
    audit.add_argument("--data", help="Override config data directory")
    select = commands.add_parser("select-heads", help="Rank once on frozen backbone and training pairs")
    select.add_argument("--config", default="configs/paper.json")
    select.add_argument("--output", default="runs/head_selection")
    train = commands.add_parser("train", help="Fit and evaluate one method")
    train.add_argument("--config", default="configs/paper.json")
    train.add_argument("--output", default="runs/paraattn_seed42")
    train.add_argument("--selection", help="Reuse a verified frozen-backbone selection report")
    train.add_argument("--method", choices=["standard_lora", "augmentation", "output_js", "paraattn"])
    train.add_argument("--seed", type=int)
    evaluate = commands.add_parser("evaluate", help="Evaluate pretrained model or saved adapter")
    evaluate.add_argument("--config", default="configs/paper.json")
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--output", default="runs/evaluation")
    evaluate.add_argument("--no-diagnostics", action="store_true")
    predict = commands.add_parser("predict", help="Score A-D for a single question; no attention regularizer at inference")
    predict.add_argument("--config", default="configs/paper.json")
    predict.add_argument("--adapter")
    predict.add_argument("--stem", required=True)
    predict.add_argument("--options", required=True, nargs=4, metavar="OPTION")
    suite = commands.add_parser("suite", help="Run baseline or ablation suite across seeds")
    suite.add_argument("--config", default="configs/paper.json")
    suite.add_argument("--suite", choices=["baselines", "heads", "representations", "layers", "weights"], default="baselines")
    suite.add_argument("--seeds", type=int, nargs="+", default=[42])
    suite.add_argument("--output", default="runs/baselines")
    compare = commands.add_parser("compare", help="Paired tests from retained group-level predictions")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--output", default="runs/comparison.json")
    compare.add_argument("--bootstrap-samples", type=int, default=10000)
    aggregate = commands.add_parser("aggregate", help="Aggregate suite_results across training seeds")
    aggregate.add_argument("--input", required=True)
    aggregate.add_argument("--output", default="runs/aggregate.json")
    return root


def audit_data(config):
    from .io import file_hash
    from .modeling import load_tokenizer
    from .prompting import PromptEncoder
    from .schema import check_disjoint, load_questions
    folder = Path(config.data.root)
    splits = {name: load_questions(folder / f"{name}.jsonl", k)
              for name, k in [("train", 1), ("validation", 1), ("test", 2)]}
    check_disjoint(splits)
    encoder = PromptEncoder(load_tokenizer(config.model), config.model.max_length)
    for rows in splits.values():
        for question in rows:
            encoder.validate_question(question)
    counts = {name: len(rows) for name, rows in splits.items()}
    expected = {"train": config.data.train_size, "validation": config.data.validation_size, "test": config.data.test_size}
    if counts != expected:
        raise ValueError(f"Dataset counts {counts} differ from config {expected}")
    hashes = {name: file_hash(folder / f"{name}.jsonl") for name in splits}
    report_path = folder / "construction_report.json"
    if report_path.exists() and read_json(report_path)["files"] != hashes:
        raise ValueError("Data changed since the construction report was saved")
    return {"counts": counts, "paraphrases": sum(len(q.paraphrases) for rows in splits.values() for q in rows),
            "file_hashes": hashes, "all_prompts_within_token_limit": True, "cross_split_overlap": False}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        import torch
        result = environment_info()
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    elif args.command == "smoke":
        from .smoke import run_smoke
        result = run_smoke(args.output)
    elif args.command == "compare":
        from .statistics import paired_comparison
        result = paired_comparison(read_jsonl(args.left), read_jsonl(args.right), args.bootstrap_samples)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    elif args.command == "aggregate":
        from .experiments import summarize_runs
        result = summarize_runs(read_json(args.input))
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    else:
        config = load_config(args.config)
        if args.command == "prepare":
            from .construction import prepare_candidates
            result = {"output": str(prepare_candidates(config, args.output, args.source_jsonl, args.reserve_factor))}
        elif args.command == "paraphrase":
            from .construction import construct_paraphrases
            result = construct_paraphrases(config, args.candidates, args.output)
        elif args.command == "audit-data":
            if args.data:
                config.data.root = args.data
            result = audit_data(config)
        elif args.command == "select-heads":
            from .io import seed_everything
            from .modeling import load_base, load_tokenizer
            from .prompting import PromptEncoder
            from .schema import load_questions
            from .selection import select_heads
            folder = new_output_dir(args.output)
            seed_everything(config.heads.seed)
            model = load_base(config.model)
            encoder = PromptEncoder(load_tokenizer(config.model), config.model.max_length)
            questions = load_questions(Path(config.data.root) / "train.jsonl", 1)
            result = select_heads(model, encoder, questions, config)
            write_json(folder / "head_selection.json", result)
        elif args.command == "train":
            from .training import run_training
            if args.method:
                config.training.method = args.method
            if args.seed is not None:
                config.training.seed = args.seed
            config.validate()
            result = run_training(config, args.output, args.selection)
        elif args.command == "evaluate":
            from .evaluation import evaluate_model
            from .modeling import load_adapter, load_base, load_tokenizer, model_metadata
            from .prompting import PromptEncoder
            from .schema import load_questions
            folder = new_output_dir(args.output)
            model = load_base(config.model)
            if args.adapter:
                model = load_adapter(model, args.adapter)
            encoder = PromptEncoder(load_tokenizer(config.model), config.model.max_length)
            questions = load_questions(Path(config.data.root) / f"{args.split}.jsonl", 2 if args.split == "test" else 1)
            result, predictions = evaluate_model(model, encoder, questions, config, not args.no_diagnostics, latency=True)
            result["model"] = model_metadata(model)
            write_json(folder / "report.json", result)
            write_jsonl(folder / "predictions.jsonl", predictions)
        elif args.command == "predict":
            import torch
            from .modeling import autocast_context, forward_prefix, input_device, load_adapter, load_base, load_tokenizer
            from .prompting import PromptEncoder
            from .schema import LABELS
            model = load_base(config.model)
            if args.adapter:
                model = load_adapter(model, args.adapter)
            encoder = PromptEncoder(load_tokenizer(config.model), config.model.max_length)
            batch = encoder.batch([encoder.encode(args.stem, args.options)], input_device(model))
            with torch.inference_mode(), autocast_context(model):
                logits = forward_prefix(model, batch).logits[0, encoder.label_token_ids]
                probabilities = logits.softmax(-1).cpu().tolist()
            label = int(logits.argmax())
            result = {"label": LABELS[label], "answer": args.options[label],
                      "probabilities": dict(zip(LABELS, probabilities))}
        elif args.command == "suite":
            from .experiments import run_suite
            result = run_suite(config, args.output, args.suite, args.seeds)
        else:
            raise ValueError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import dataclasses
import random
from collections import Counter
from pathlib import Path

from .config import Config, REPRODUCTION_NOTES
from .io import append_jsonl, environment_info, file_hash, new_output_dir, read_json, read_jsonl, write_json, write_jsonl
from .modeling import input_device, load_base, load_tokenizer
from .prompting import PromptEncoder, PromptError
from .schema import LABELS, Question, check_disjoint, load_questions
from .validation import generation_prompt, normalize_whitespace, validate_candidate, verifier_prompt


def arc_question(row, source_split):
    choices = row["choices"]
    letters = list(choices["label"])
    texts = list(choices["text"])
    answer = row["answerKey"]
    if letters != list(LABELS) or answer not in LABELS:
        raise ValueError("Only records with ordered A-D labels are retained")
    return Question(str(row["id"]), normalize_whitespace(row["question"]),
                    [normalize_whitespace(s) for s in texts], LABELS.index(answer),
                    source_split=source_split).validate()


def prepare_candidates(config: Config, output, local_source=None, reserve_factor=2):
    """A disjoint seeded pilot partition of the official train pool (documented default)."""
    if reserve_factor < 1:
        raise ValueError("reserve_factor must be >= 1")
    output = new_output_dir(output)
    tokenizer = load_tokenizer(config.model)
    encoder = PromptEncoder(tokenizer, config.model.max_length)
    if local_source:
        raw = read_jsonl(local_source)
        fingerprint = file_hash(local_source)
    else:
        from datasets import load_dataset
        dataset = load_dataset(config.data.dataset, config.data.subset,
                               split=config.data.source_split, revision=config.data.revision)
        raw = dataset
        fingerprint = dataset._fingerprint
    valid, rejected = [], Counter()
    seen_ids, seen_content = set(), set()
    for row in raw:
        try:
            question = arc_question(row, config.data.source_split)
            encoder.validate_question(question)
            if question.id in seen_ids or question.content_hash in seen_content:
                raise ValueError("Duplicate ID or content")
            seen_ids.add(question.id); seen_content.add(question.content_hash)
            valid.append(question)
        except (ValueError, KeyError, TypeError, PromptError) as error:
            rejected[str(error)] += 1
    random.Random(config.data.split_seed).shuffle(valid)
    sizes = {"train": config.data.train_size, "validation": config.data.validation_size,
             "test": config.data.test_size}
    needed = sum(sizes.values()) * reserve_factor
    if len(valid) < needed:
        raise ValueError(f"Need {needed} valid source records including reserves; found {len(valid)}. Lower --reserve-factor or supply more data.")
    offset, splits = 0, {}
    for name, size in sizes.items():
        count = size * reserve_factor
        splits[name] = valid[offset:offset + count]
        offset += count
        write_jsonl(output / f"{name}.jsonl", [q.to_dict() for q in splits[name]])
    check_disjoint(splits)
    write_json(output / "manifest.json", {
        "config": config.to_dict(), "dataset_fingerprint": fingerprint,
        "partition_policy": "disjoint partitions of source_split after strict filtering and seeded shuffle",
        "requested_sizes": sizes, "reserve_factor": reserve_factor,
        "filtered_source_count": len(valid), "filter_rejections": dict(rejected),
        "candidate_counts": {key: len(value) for key, value in splits.items()},
        "files": {key: file_hash(output / f"{key}.jsonl") for key in splits},
        "environment": environment_info(),
    })
    return output


class GreedyLLM:
    def __init__(self, model, tokenizer, max_new_tokens=128, max_input_tokens=2048):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.max_input_tokens = max_new_tokens, max_input_tokens

    def __call__(self, prompt):
        import torch
        text = self.tokenizer.apply_chat_template([{ "role": "user", "content": prompt }],
                                                 tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        if inputs.input_ids.shape[-1] > self.max_input_tokens:
            raise ValueError("Generation prompt exceeds max_input_tokens; refusing truncation")
        inputs = {k: v.to(input_device(self.model)) for k, v in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                         max_new_tokens=self.max_new_tokens, use_cache=True,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def build_split(questions, target, paraphrase_count, generator_fn, similarity_fn, verifier_fn,
                config, encoder, attempts_path, split_name):
    accepted = []
    for question in questions:
        previous, complete = [], True
        for slot in range(paraphrase_count):
            feedback = []
            for attempt in range(config.generation.max_retries + 1):
                prompt = generation_prompt(question, slot, previous, feedback)
                raw = generator_fn(prompt)
                result = validate_candidate(question, raw, previous, similarity_fn, verifier_fn,
                                            config.generation.similarity_threshold, encoder)
                append_jsonl(attempts_path, {
                    "split": split_name, "question_id": question.id, "question": question.to_dict(),
                    "slot": slot, "attempt": attempt, "generation_prompt": prompt,
                    "raw_output": raw, "validation": dataclasses.asdict(result),
                    "verifier_prompt": verifier_prompt(question, result.candidate) if result.verifier_output is not None else None,
                    "decoding": {"do_sample": False, "temperature": 0, "top_p": 1.0,
                                 "max_new_tokens": config.generation.max_new_tokens},
                })
                if result.accepted:
                    previous.append(result.candidate)
                    break
                feedback = result.failures
            else:
                complete = False
                break
        if complete:
            question = dataclasses.replace(question, paraphrases=previous)
            accepted.append(question.validate(paraphrase_count))
        if len(accepted) == target:
            break
    if len(accepted) != target:
        raise ValueError(f"{split_name}: only {len(accepted)}/{target} questions passed; attempt log retained. Prepare a larger reserve pool.")
    return accepted


def construct_paraphrases(config: Config, candidates, output):
    output = new_output_dir(output)
    candidates = Path(candidates)
    manifest = read_json(candidates / "manifest.json")
    splits = {name: load_questions(candidates / f"{name}.jsonl", 0)
              for name in ["train", "validation", "test"]}
    check_disjoint(splits)
    encoder = PromptEncoder(load_tokenizer(config.model), config.model.max_length)
    generation_config = dataclasses.replace(config.model, name=config.generation.model,
                                             revision=config.generation.revision)
    generation_model = load_base(generation_config)
    generation_tokenizer = load_tokenizer(generation_config)
    generator = GreedyLLM(generation_model, generation_tokenizer, config.generation.max_new_tokens,
                          config.generation.max_input_tokens)
    if (config.generation.verifier_model, config.generation.verifier_revision) == (config.generation.model, config.generation.revision):
        verifier = GreedyLLM(generation_model, generation_tokenizer, 8, config.generation.max_input_tokens)
    else:
        verifier_config = dataclasses.replace(config.model, name=config.generation.verifier_model,
                                               revision=config.generation.verifier_revision)
        verifier = GreedyLLM(load_base(verifier_config), load_tokenizer(verifier_config),
                            8, config.generation.max_input_tokens)
    from sentence_transformers import SentenceTransformer
    embeddings = SentenceTransformer(config.generation.embedding_model,
                                     revision=config.generation.embedding_revision,
                                     device=config.generation.embedding_device)

    def cosine(original, candidate):
        vectors = embeddings.encode([original, candidate], normalize_embeddings=True)
        return float(vectors[0] @ vectors[1])

    accepted = {}
    attempts_path = output / "generation_attempts.jsonl"
    write_json(output / "construction_in_progress.json", {"config": config.to_dict(), "candidate_manifest": manifest})
    for name, count, required in [("train", config.data.train_size, 1),
                                  ("validation", config.data.validation_size, 1),
                                  ("test", config.data.test_size, 2)]:
        accepted[name] = build_split(splits[name], count, required, generator, cosine, verifier,
                                     config, encoder, attempts_path, name)
        write_jsonl(output / f"{name}.jsonl", [q.to_dict() for q in accepted[name]])
        print(f"Constructed {name}: {len(accepted[name])} groups", flush=True)
    check_disjoint(accepted)
    attempts = read_jsonl(attempts_path)
    invalid = [a for a in attempts if not a["validation"]["accepted"]]
    reasons = Counter(reason for a in invalid for reason in a["validation"]["failures"])
    with open(output / "human_review.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "id", "original", "paraphrase", "options", "label", "reviewer_decision"])
        for name, rows in accepted.items():
            for q in rows:
                for paraphrase in q.paraphrases:
                    writer.writerow([name, q.id, q.stem, paraphrase, " | ".join(q.options), LABELS[q.label], ""])
    report = {
        "reproduction_notes": REPRODUCTION_NOTES,
        "config": config.to_dict(), "candidate_manifest": manifest, "environment": environment_info(),
        "accepted_groups": {name: len(rows) for name, rows in accepted.items()},
        "retained_paraphrases": sum(len(q.paraphrases) for rows in accepted.values() for q in rows),
        "attempts": len(attempts), "invalid_attempts": len(invalid),
        "invalid_paraphrase_rate": len(invalid) / len(attempts), "rejection_counts_by_rule": dict(reasons),
        "files": {name: file_hash(output / f"{name}.jsonl") for name in accepted},
        "generator_resolved_revision": getattr(generation_model.config, "_commit_hash", None),
        "human_review": "No human adjudication has been performed; CSV is supplied for independent audit.",
    }
    write_json(output / "construction_report.json", report)
    return report

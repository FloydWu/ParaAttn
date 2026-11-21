from __future__ import annotations

import random

import torch

from .attention import candidate_heads, choose_heads, js_divergence, region_distribution
from .io import canonical_hash
from .modeling import autocast_context, forward_prefix, input_device, model_metadata


def select_heads(model, encoder, training_questions, config):
    size = config.heads.subset_size
    if len(training_questions) < size:
        raise ValueError(f"Selection needs {size} training pairs; got {len(training_questions)}")
    subset = random.Random(config.heads.seed).sample(training_questions, size)
    heads = candidate_heads(config.heads.layers, model.config.num_hidden_layers,
                             model.config.num_attention_heads)
    was_training = model.training
    model.eval()
    totals = torch.zeros(len(heads), dtype=torch.float64)
    batch_size = config.evaluation.batch_size
    with torch.no_grad():
        for start in range(0, len(subset), batch_size):
            questions = subset[start:start + batch_size]
            original = [encoder.encode(q.stem, q.options) for q in questions]
            paraphrase = [encoder.encode(q.paraphrases[0], q.options) for q in questions]
            batch = encoder.batch(original + paraphrase, input_device(model))
            with autocast_context(model):
                output = forward_prefix(model, batch, heads)
                regions = region_distribution(output.rows, batch["region_ids"], config.heads.eps)
                count = len(questions)
                scores = js_divergence(regions[:count], regions[count:])
            totals += scores.sum(0).double().cpu()
    model.train(was_training)
    means = (totals / size).tolist()
    scored = [{"layer": layer, "head": head, "mean_js": score}
              for (layer, head), score in zip(heads, means)]
    selected = choose_heads(scored, config.heads.ratio, config.heads.strategy, config.heads.seed)
    return {
        "selected": [list(head) for head in selected], "scores": scored,
        "selection_subset_ids": [q.id for q in subset],
        "training_data_hash": canonical_hash([q.to_dict() for q in training_questions]),
        "model": model_metadata(model), "model_name": config.model.name,
        "model_revision": config.model.revision, "head_config": config.to_dict()["heads"],
        "selection_is_recomputed": False, "log_base": "natural",
    }


def validate_selection_report(report, model, questions, config):
    if report["training_data_hash"] != canonical_hash([q.to_dict() for q in questions]):
        raise ValueError("Selection report belongs to different training data")
    if report["model_name"] != config.model.name or report["model_revision"] != config.model.revision:
        raise ValueError("Selection report belongs to a different model/revision")
    if report["head_config"] != config.to_dict()["heads"]:
        raise ValueError("Selection report and head configuration disagree")
    metadata = model_metadata(model)
    for key in ["layers", "query_heads", "kv_heads", "resolved_revision", "quantized"]:
        if report["model"][key] != metadata[key]:
            raise ValueError(f"Selection model mismatch: {key}")
    heads = [tuple(head) for head in report["selected"]]
    expected = set(candidate_heads(config.heads.layers, metadata["layers"], metadata["query_heads"]))
    if not heads or len(set(heads)) != len(heads) or not set(heads) <= expected:
        raise ValueError("Invalid selected head indices")
    return heads

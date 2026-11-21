from __future__ import annotations

import time

import numpy as np
import torch

from .attention import candidate_heads, js_divergence, region_distribution
from .modeling import autocast_context, forward_prefix, input_device, synchronize
from .statistics import metric_report


def evaluate_model(model, encoder, questions, config, diagnostics=True, latency=False):
    was_training = model.training
    model.eval()
    flattened, group_indices = [], []
    for question in questions:
        indices = []
        for stem in [question.stem] + question.paraphrases:
            indices.append(len(flattened))
            flattened.append(encoder.encode(stem, question.options))
        group_indices.append(indices)
    predictions_flat = []
    label_ids = torch.tensor(encoder.label_token_ids, device=input_device(model))
    with torch.no_grad():
        for start in range(0, len(flattened), config.evaluation.batch_size):
            batch = encoder.batch(flattened[start:start + config.evaluation.batch_size], input_device(model))
            with autocast_context(model):
                output = forward_prefix(model, batch)
                logits = output.logits[:, label_ids]
                probabilities = logits.softmax(-1)
            for scores, probs in zip(logits.cpu().tolist(), probabilities.cpu().tolist()):
                predictions_flat.append({"prediction": int(np.argmax(scores)),
                                         "answer_logits": scores, "answer_probabilities": probs})
    predictions = []
    for question, indices in zip(questions, group_indices):
        predictions.append({"id": question.id, "label": question.label, "group_hash": question.group_hash,
                            "original": predictions_flat[indices[0]],
                            "paraphrases": [predictions_flat[i] for i in indices[1:]]})
    report = metric_report(predictions, config.evaluation.bootstrap_samples, config.evaluation.bootstrap_seed)
    if diagnostics:
        heads = candidate_heads(config.evaluation.diagnostic_layers, model.config.num_hidden_layers,
                                 model.config.num_attention_heads)
        divergence = []
        with torch.no_grad():
            for question in questions:
                encoded = [encoder.encode(stem, question.options) for stem in [question.stem] + question.paraphrases]
                batch = encoder.batch(encoded, input_device(model))
                with autocast_context(model):
                    output = forward_prefix(model, batch, heads)
                    regions = region_distribution(output.rows, batch["region_ids"], config.heads.eps)
                    js = js_divergence(regions[:1].expand_as(regions[1:]), regions[1:])
                divergence.extend(js.mean(-1).cpu().tolist())
        report["mean_attention_js_divergence"] = float(np.mean(divergence))
        report["attention_diagnostic"] = {"layers": config.evaluation.diagnostic_layers,
                                           "heads": len(heads), "aggregation": "all candidate query heads, then all original-paraphrase pairs",
                                           "log_base": "natural"}
    if latency:
        count = min(config.evaluation.latency_samples, len(questions))
        batches = [encoder.batch([encoder.encode(q.stem, q.options)], input_device(model)) for q in questions[:count]]
        timings = []
        with torch.no_grad():
            for _ in range(config.evaluation.latency_warmup):
                with autocast_context(model):
                    forward_prefix(model, batches[0])
            synchronize(model)
            for batch in batches:
                synchronize(model)
                start = time.perf_counter()
                with autocast_context(model):
                    forward_prefix(model, batch)
                synchronize(model)
                timings.append(1000 * (time.perf_counter() - start))
        report["inference_latency"] = {"mean_ms": float(np.mean(timings)), "median_ms": float(np.median(timings)),
                                         "samples": count, "batch_size": 1, "includes_tokenization": False,
                                         "output_attentions": False, "operation": "single label-free prefix forward"}
    model.train(was_training)
    return report, predictions

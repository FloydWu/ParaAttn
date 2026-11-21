from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import aligned_token_js, js_divergence, region_distribution
from .modeling import forward_prefix, input_device


def training_objective(model, encoder, questions, config, heads=None):
    method = config.training.method
    original = [encoder.encode(q.stem, q.options) for q in questions]
    paired = method != "standard_lora"
    encoded = original
    if paired:
        encoded += [encoder.encode(q.paraphrases[0], q.options) for q in questions]
    batch = encoder.batch(encoded, input_device(model))
    use_attention = method == "paraattn" and config.training.regularization_weight > 0
    if use_attention and not heads:
        raise ValueError("ParaAttn regularization needs a fixed selected-head set")
    result = forward_prefix(model, batch, heads if use_attention else None)
    labels = torch.tensor([q.label for q in questions], device=result.logits.device)
    label_ids = torch.tensor(encoder.label_token_ids, device=result.logits.device)
    targets = label_ids[labels]
    count = len(questions)
    # Full-vocabulary next-token cross entropy; branch losses are SUMMED as Eq. 2.
    ce_original = F.cross_entropy(result.logits[:count], targets)
    ce_paraphrase = F.cross_entropy(result.logits[count:], targets) if paired else ce_original.new_zeros(())
    consistency = ce_original.new_zeros(())
    if method == "output_js":
        distributions = F.softmax(result.logits[:, label_ids], dim=-1)
        consistency = js_divergence(distributions[:count], distributions[count:]).mean()
    elif use_attention:
        if config.training.representation == "region":
            distributions = region_distribution(result.rows, batch["region_ids"], config.heads.eps)
            consistency = js_divergence(distributions[:count], distributions[count:]).mean()
        else:
            consistency = aligned_token_js(model, result.rows, batch, count,
                                             config.training.representation, config.heads.eps).mean()
    loss = ce_original + ce_paraphrase + config.training.regularization_weight * consistency
    return loss, {"ce_original": float(ce_original.detach()), "ce_paraphrase": float(ce_paraphrase.detach()),
                  "consistency_loss": float(consistency.detach()), "total_loss": float(loss.detach())}

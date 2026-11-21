from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


def candidate_heads(layers, num_layers: int, num_heads: int):
    if any(layer < 0 or layer >= num_layers for layer in layers):
        raise ValueError(f"Candidate layer outside model range [0, {num_layers})")
    return [(layer, head) for layer in sorted(layers) for head in range(num_heads)]


def normalize_mass(mass: torch.Tensor, eps: float = 1e-12):
    mass = mass.float()
    return (mass + eps) / (mass.sum(dim=-1, keepdim=True) + mass.shape[-1] * eps)


def region_distribution(rows: torch.Tensor, region_ids: torch.Tensor, eps: float = 1e-12):
    """Differentiable attention mass, summed (not averaged) in five regions."""
    if rows.ndim != 3 or region_ids.shape != (rows.shape[0], rows.shape[-1]):
        raise ValueError("Expected rows [batch, heads, tokens] and regions [batch, tokens]")
    mask = torch.stack([region_ids.eq(index) for index in range(5)], dim=1).float()
    mass = torch.einsum("bht,brt->bhr", rows.float(), mask)
    return normalize_mass(mass, eps)


def js_divergence(p: torch.Tensor, q: torch.Tensor):
    """Natural-log JS, shape [..., bins] -> [...], range [0, ln(2)]."""
    if p.shape != q.shape:
        raise ValueError("JS inputs must have identical shapes")
    tiny = torch.finfo(torch.float32).tiny
    p, q = p.float(), q.float()
    middle = (p + q) / 2
    log_middle = middle.clamp_min(tiny).log()
    return 0.5 * ((p * (p.clamp_min(tiny).log() - log_middle)).sum(-1)
                  + (q * (q.clamp_min(tiny).log() - log_middle)).sum(-1))


def choose_heads(scores: list[dict], ratio: float, strategy: str, seed: int):
    if not scores:
        raise ValueError("No candidate head scores")
    count = len(scores) if strategy == "all" else math.floor(len(scores) * ratio)
    if count < 1:
        raise ValueError("Selection ratio rounds to zero heads")
    if strategy == "sensitive":
        ordered = sorted(scores, key=lambda row: (-row["mean_js"], row["layer"], row["head"]))
    elif strategy == "low":
        ordered = sorted(scores, key=lambda row: (row["mean_js"], row["layer"], row["head"]))
    elif strategy == "random":
        ordered = sorted(scores, key=lambda row: (row["layer"], row["head"]))
        ordered = random.Random(seed).sample(ordered, len(ordered))
    elif strategy == "all":
        ordered = sorted(scores, key=lambda row: (row["layer"], row["head"]))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return sorted((row["layer"], row["head"]) for row in ordered[:count])


def aligned_token_js(model, rows: torch.Tensor, batch, pair_count: int,
                     representation: str, eps: float = 1e-12):
    """Explicit implementations of the manuscript's underspecified token ablations.

    Position: align positions separately within each region and pad missing tokens.
    Matching: Hungarian assignment on frozen input-embedding cosine similarity,
    separately within each region; unmatched tokens get their own zero-mass slot.
    Both methods preserve all semantic attention mass and retain attention gradients.
    """
    divergences = []
    regions = batch["region_ids"]
    for index in range(pair_count):
        other = index + pair_count
        left_indices, right_indices = [], []
        for region in range(5):
            left = regions[index].eq(region).nonzero(as_tuple=False).flatten().tolist()
            right = regions[other].eq(region).nonzero(as_tuple=False).flatten().tolist()
            if representation == "position":
                slots = max(len(left), len(right))
                left_indices += left + [-1] * (slots - len(left))
                right_indices += right + [-1] * (slots - len(right))
            elif representation == "matching":
                from scipy.optimize import linear_sum_assignment
                with torch.no_grad():
                    embeddings = model.get_input_embeddings()
                    a = F.normalize(embeddings(batch["input_ids"][index, left]).float(), dim=-1)
                    b = F.normalize(embeddings(batch["input_ids"][other, right]).float(), dim=-1)
                    cost = -(a @ b.T).cpu().numpy()
                assigned_left, assigned_right = linear_sum_assignment(cost)
                used_left, used_right = set(), set()
                for li, ri in zip(assigned_left.tolist(), assigned_right.tolist()):
                    left_indices.append(left[li]); right_indices.append(right[ri])
                    used_left.add(li); used_right.add(ri)
                for li in range(len(left)):
                    if li not in used_left:
                        left_indices.append(left[li]); right_indices.append(-1)
                for ri in range(len(right)):
                    if ri not in used_right:
                        left_indices.append(-1); right_indices.append(right[ri])
            else:
                raise ValueError("Expected position or matching representation")
        left_idx = torch.tensor(left_indices, device=rows.device)
        right_idx = torch.tensor(right_indices, device=rows.device)
        left_mass = rows[index, :, left_idx.clamp_min(0)] * left_idx.ge(0).float()
        right_mass = rows[other, :, right_idx.clamp_min(0)] * right_idx.ge(0).float()
        divergences.append(js_divergence(normalize_mass(left_mass, eps), normalize_mass(right_mass, eps)))
    return torch.stack(divergences)

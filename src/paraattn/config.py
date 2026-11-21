from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List


REPRODUCTION_NOTES = [
    "This is a new reference implementation of ParaAttn_revised_3_highlighted.docx; the original accepted corpus and training artifacts were not supplied.",
    "The pilot split defaults to disjoint partitions of official ARC-Easy train data, with deterministic reserve queues to replace failed questions; original sample IDs were unavailable.",
    "The exact SBERT checkpoint, verifier checkpoint, gradient accumulation factor, checkpoint score formula, quantizer details and diagnostic head domain were not fully specified in the manuscript. Explicit defaults are recorded in the config and reports.",
    "Greedy test paraphrases use distinct style directions; retry prompts include rejection feedback. These prompt extensions are needed for diversity and are logged.",
    "Token-position and Hungarian token-matching ablations are explicit engineering definitions of underspecified manuscript baselines; they are not claimed to reproduce the archived Table 4 implementation.",
    "For Qwen2.5-0.5B, q_proj/v_proj at rank 8 in all 24 layers yield 540672 adapter parameters (0.541M), rather than the manuscript's 2.18M; actual counts are measured.",
    "Model revisions default to main for usability. Pin immutable commit revisions for archival runs. Resolved backbone and generator commits are written to reports.",
    "The code computes new metrics from predictions and never substitutes the manuscript's numeric results. The automated semantic verifier does not replace independent human validation.",
    "A-D targets are the isolated single vocabulary tokens after the unchanged Answer: prefix. Concatenating and retokenizing prefix+label would merge Qwen BPE boundaries, so labels are scored by token ID without appending a gold label to the forward input.",
]


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    device: str = "cuda:0"
    load_in_4bit: bool = True
    dtype: str = "auto"
    max_length: int = 256
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class DataConfig:
    dataset: str = "allenai/ai2_arc"
    subset: str = "ARC-Easy"
    revision: str = "main"
    source_split: str = "train"
    train_size: int = 500
    validation_size: int = 100
    test_size: int = 100
    split_seed: int = 42
    root: str = "data/arc_easy"


@dataclass
class GenerationConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    verifier_model: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str = "main"
    verifier_revision: str = "main"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_revision: str = "main"
    embedding_device: str = "cpu"
    similarity_threshold: float = 0.85
    max_new_tokens: int = 128
    max_retries: int = 3
    max_input_tokens: int = 2048


@dataclass
class HeadConfig:
    layers: List[int] = field(default_factory=lambda: list(range(16, 24)))
    subset_size: int = 100
    ratio: float = 0.20
    strategy: str = "sensitive"
    seed: int = 42
    eps: float = 1e-12


@dataclass
class TrainConfig:
    method: str = "paraattn"
    representation: str = "region"
    seed: int = 42
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    regularization_weight: float = 0.10
    max_grad_norm: float = 1.0
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    batch_size: int = 4
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 42
    diagnostic_layers: List[int] = field(default_factory=lambda: list(range(16, 24)))
    latency_warmup: int = 3
    latency_samples: int = 20


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    heads: HeadConfig = field(default_factory=HeadConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self):
        if self.training.method not in {"standard_lora", "augmentation", "output_js", "paraattn"}:
            raise ValueError("Unknown training method")
        if self.training.representation not in {"region", "position", "matching"}:
            raise ValueError("Unknown attention representation")
        if self.heads.strategy not in {"sensitive", "low", "random", "all"}:
            raise ValueError("Unknown head selection strategy")
        if not self.heads.layers or min(self.heads.layers) < 0:
            raise ValueError("Candidate layers must be nonempty and zero based")
        if len(set(self.heads.layers)) != len(self.heads.layers):
            raise ValueError("Duplicate candidate layers")
        if not 0 < self.heads.ratio <= 1:
            raise ValueError("Head ratio must be in (0, 1]")
        for value in [self.heads.subset_size, self.model.max_length, self.training.epochs,
                      self.training.batch_size, self.training.gradient_accumulation,
                      self.evaluation.batch_size, self.evaluation.bootstrap_samples]:
            if value <= 0:
                raise ValueError("Sizes, lengths, epochs and accumulation must be positive")
        if self.training.regularization_weight < 0 or self.heads.eps <= 0:
            raise ValueError("Invalid regularization weight or epsilon")
        if not 0 <= self.generation.similarity_threshold <= 1:
            raise ValueError("Invalid cosine similarity threshold")
        if self.generation.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.model.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Unknown compute dtype")
        return self

    def to_dict(self):
        return dataclasses.asdict(self)


_SECTIONS = {"model": ModelConfig, "data": DataConfig, "generation": GenerationConfig,
             "heads": HeadConfig, "training": TrainConfig, "evaluation": EvaluationConfig}


def config_from_dict(raw: dict[str, Any]) -> Config:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    sections = {}
    for name, cls in _SECTIONS.items():
        values = raw.get(name, {})
        allowed = {f.name for f in dataclasses.fields(cls)}
        if set(values) - allowed:
            raise ValueError(f"Unknown keys in {name}: {sorted(set(values) - allowed)}")
        sections[name] = cls(**values)
    return Config(**sections).validate()


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))

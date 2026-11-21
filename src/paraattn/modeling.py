from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .config import ModelConfig


def compute_dtype(config: ModelConfig):
    if config.dtype != "auto":
        return getattr(torch, config.dtype)
    if config.device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(config: ModelConfig):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.name, revision=config.revision,
                                         use_fast=True, trust_remote_code=False)


def load_base(config: ModelConfig):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run the paper configuration on an NVIDIA GPU, or explicitly set device=cpu and load_in_4bit=false.")
    dtype = compute_dtype(config)
    kwargs = {"revision": config.revision, "dtype": dtype,
              "attn_implementation": "eager", "trust_remote_code": False}
    if config.load_in_4bit:
        if not config.device.startswith("cuda"):
            raise RuntimeError("This reference implementation requires CUDA for bitsandbytes NF4 training")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": config.device}
    model = AutoModelForCausalLM.from_pretrained(config.name, **kwargs)
    if not config.load_in_4bit:
        model.to(config.device)
    if model.config.model_type != "qwen2":
        raise ValueError("The tested logits_to_keep and attention path is for Qwen2/Qwen2.5")
    model.config.use_cache = False
    model.config._paraattn_compute_dtype = str(dtype).removeprefix("torch.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def attach_lora(model, config: ModelConfig):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    lora = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha,
                      lora_dropout=config.lora_dropout, target_modules=config.target_modules,
                      bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No trainable adapter parameters")
    return model


def load_adapter(model, path):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(path), is_trainable=False)
    return model.eval()


def input_device(model):
    return model.get_input_embeddings().weight.device


def autocast_context(model):
    device = input_device(model)
    if device.type == "cuda":
        name = getattr(model.config, "_paraattn_compute_dtype", "auto")
        dtype = getattr(torch, name) if name != "auto" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        return torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
    return nullcontext()


def synchronize(model):
    if input_device(model).type == "cuda":
        torch.cuda.synchronize(input_device(model))


def model_metadata(model):
    config = model.config
    return {"model_type": config.model_type, "layers": config.num_hidden_layers,
            "query_heads": config.num_attention_heads,
            "kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "resolved_revision": getattr(config, "_commit_hash", None),
            "quantized": bool(getattr(model, "is_loaded_in_4bit", False))}


@dataclass
class ForwardResult:
    logits: torch.Tensor
    rows: torch.Tensor | None


def forward_prefix(model, batch, heads: list[tuple[int, int]] | None = None):
    """Only label-free prefixes enter the model; gather each last *real* token.

    Project only the union of real query positions into the large vocabulary.
    This reduces memory while preserving right-padded attention offsets.
    """
    query = batch["query_positions"]
    positions = torch.unique(query, sorted=True)
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    output_attentions=bool(heads), use_cache=False, return_dict=True,
                    logits_to_keep=positions)
    local = torch.searchsorted(positions, query)
    logits = outputs.logits[torch.arange(len(query), device=query.device), local].float()
    rows = None
    if heads:
        if outputs.attentions is None or any(a is None for a in outputs.attentions):
            raise RuntimeError("Attention weights unavailable; the model must use eager attention")
        selected = []
        for layer, head in heads:
            attention = outputs.attentions[layer][:, head, :, :]
            row = attention.gather(1, query[:, None, None].expand(-1, 1, attention.shape[-1]))
            selected.append(row.squeeze(1))
        rows = torch.stack(selected, dim=1).float()
    return ForwardResult(logits, rows)

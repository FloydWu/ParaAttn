from __future__ import annotations

from dataclasses import dataclass

import torch

from .schema import LABELS, Question


class PromptError(ValueError):
    pass


def make_prompt(stem: str, options: list[str]):
    """Return the manuscript's exact plain prefix and five content character spans."""
    if len(options) != 4:
        raise PromptError("Four options required")
    text = "Choose the correct answer. Question: "
    spans = []
    start = len(text)
    text += stem
    spans.append((start, len(text)))
    for letter, option in zip(LABELS, options):
        text += f"\n{letter}. "
        start = len(text)
        text += option
        spans.append((start, len(text)))
    text += "\nAnswer:"
    return text, spans


@dataclass
class EncodedPrompt:
    input_ids: list[int]
    region_ids: list[int]
    text: str


class PromptEncoder:
    def __init__(self, tokenizer, max_length: int = 256):
        if not tokenizer.is_fast:
            raise PromptError("A fast tokenizer with offset mappings is required")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_token_ids = []
        for letter in LABELS:
            tokens = tokenizer.encode(letter, add_special_tokens=False)
            if len(tokens) != 1:
                raise PromptError(f"Answer label {letter} is not a single token")
            self.label_token_ids.append(tokens[0])
        if len(set(self.label_token_ids)) != 4:
            raise PromptError("Answer labels have colliding token IDs")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise PromptError("Tokenizer has neither PAD nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    def encode(self, stem: str, options: list[str]) -> EncodedPrompt:
        text, spans = make_prompt(stem, options)
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        if len(ids) > self.max_length:
            raise PromptError(f"Prompt has {len(ids)} tokens, limit {self.max_length}; no truncation is allowed")
        if not ids:
            raise PromptError("Empty tokenized prompt")
        # Labels are single vocabulary-token targets, not a re-encoding of the
        # concatenated string. Qwen BPE can merge ':A' if the full string is
        # re-tokenized. Preserve these prefix IDs and score isolated A-D tokens.
        # No gold label is ever appended to the forward input.
        region_ids = []
        for start, end in encoded["offset_mapping"]:
            # A BPE token may include a leading space. Ignore whitespace, but never
            # assign a token containing instruction/label/delimiter text to a region.
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            region = -1
            if start < end:
                for index, (left, right) in enumerate(spans):
                    if left <= start and end <= right:
                        region = index
                        break
            region_ids.append(region)
        if any(index not in region_ids for index in range(5)):
            raise PromptError("At least one semantic region has no unambiguous content tokens")
        if region_ids[-1] != -1:
            raise PromptError("Final Answer marker must be excluded from semantic regions")
        return EncodedPrompt(ids, region_ids, text)

    def validate_question(self, question: Question):
        question.validate()
        for stem in [question.stem] + question.paraphrases:
            self.encode(stem, question.options)

    def batch(self, encoded: list[EncodedPrompt], device=None):
        if not encoded:
            raise PromptError("Empty batch")
        width = max(len(row.input_ids) for row in encoded)
        ids = torch.full((len(encoded), width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        regions = torch.full_like(ids, -1)
        for index, row in enumerate(encoded):
            length = len(row.input_ids)
            ids[index, :length] = torch.tensor(row.input_ids, dtype=torch.long)
            mask[index, :length] = 1
            regions[index, :length] = torch.tensor(row.region_ids, dtype=torch.long)
        batch = {"input_ids": ids, "attention_mask": mask, "region_ids": regions,
                 "query_positions": mask.sum(-1) - 1}
        if device is not None:
            batch = {key: value.to(device) for key, value in batch.items()}
        return batch

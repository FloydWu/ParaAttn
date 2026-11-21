from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .schema import LABELS, Question

GENERATION_TEMPLATE = (
    "Rewrite only the QUESTION STEM of the multiple-choice item below. "
    "Preserve exactly the meaning, facts, entities, numbers, units, negation, quantifiers, "
    "and knowledge being tested. Do not solve the question, add hints, mention the correct "
    "answer, refer to option letters, or change the number, text, or order of the answer "
    "choices. Return one rewritten question stem and no explanation. "
    "QUESTION: {stem}. OPTIONS (context only; do not rewrite): "
    "A. {A} B. {B} C. {C} D. {D}."
)

VERIFIER_TEMPLATE = (
    "Audit this proposed paraphrase of a multiple-choice science question. "
    "Check whether both stems ask exactly the same question, preserve facts, entities, "
    "numbers, units, negation and quantifiers, and require the same knowledge. "
    "Check whether the unchanged gold option remains correct and whether any needed "
    "information was added or removed. Return exactly PASS if all checks pass; "
    "otherwise return exactly FAIL. Do not infer equivalence from lexical similarity alone.\n"
    "ORIGINAL: {original}\nCANDIDATE: {candidate}\n"
    "OPTIONS:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nGOLD LABEL: {label}"
)

STYLES = [
    "Use a different sentence structure while preserving every constraint.",
    "Reorder clauses or use an alternative grammatical construction while preserving every constraint.",
]

_NUMBERS = re.compile(r"(?<![\w.])[+-]?\d+(?:[.,]\d+)*(?:/\d+)?%?")
_NEGATION = re.compile(r"\b(?:not|no|never|neither|nor|without|except|least)\b|n['’]t\b", re.I)
_CUES = re.compile(r"\b(?:correct answer\b|the answer is\b|option\s*[ABCD]\b|choice\s*[ABCD]\b|answer\s*:)", re.I)
_UNITS = re.compile(
    r"(?<!\w)(?:mm|cm|km|m|mg|kg|g|mL|ml|L|liters?|litres?|meters?|metres?|"
    r"centimeters?|kilometers?|grams?|kilograms?|seconds?|minutes?|hours?|"
    r"feet|foot|inches?|miles?|watts?|joules?|volts?|degrees?|°C|°F|%)(?!\w)", re.I
)


def normalize_whitespace(text: str):
    return " ".join(text.split())


def generation_prompt(question: Question, slot: int, previous: list[str], feedback: list[str]):
    prompt = GENERATION_TEMPLATE.format(stem=question.stem, **dict(zip(LABELS, question.options)))
    # Greedy decoding is retained. Distinct slots/retries need distinct prompts.
    prompt += "\nSTYLE DIRECTION: " + STYLES[slot % len(STYLES)]
    if previous:
        prompt += "\nAvoid duplicating these earlier stems:\n" + "\n".join(previous)
    if feedback:
        prompt += "\nThe previous candidate failed these checks: " + ", ".join(feedback)
        prompt += ". Produce a new candidate that passes them."
    return prompt


def verifier_prompt(question: Question, candidate: str):
    return VERIFIER_TEMPLATE.format(original=question.stem, candidate=candidate,
                                    label=LABELS[question.label], **dict(zip(LABELS, question.options)))


def rule_failures(original: str, candidate: str, options: list[str], previous: list[str]):
    failures = []
    raw_candidate = candidate
    candidate = normalize_whitespace(candidate)
    lowered = candidate.casefold()
    original_lower = normalize_whitespace(original).casefold()
    if not candidate:
        return ["empty"]
    if lowered in {original_lower, *(normalize_whitespace(p).casefold() for p in previous)}:
        failures.append("duplicate")
    if _CUES.search(candidate):
        failures.append("answer_cue")
    if re.search(r"(?:^|\n)\s*(?:[ABCD][.)]|[-*]|\d+[.)])\s", raw_candidate) or re.match(
        r"^(?:paraphrase|rewritten question|explanation|here is)\s*:", candidate, re.I
    ):
        failures.append("not_single_stem")
    if Counter(_NUMBERS.findall(original)) != Counter(_NUMBERS.findall(candidate)):
        failures.append("numbers_changed")
    if Counter(u.casefold() for u in _UNITS.findall(original)) != Counter(u.casefold() for u in _UNITS.findall(candidate)):
        failures.append("units_changed")
    if Counter(n.casefold() for n in _NEGATION.findall(original)) != Counter(n.casefold() for n in _NEGATION.findall(candidate)):
        failures.append("negation_changed")
    for option in options:
        normalized = normalize_whitespace(option).casefold()
        # Short common option words can already be part of the question. Flag a
        # newly copied full option phrase, avoiding false rejection for "water".
        if (len(normalized) >= 12 or len(normalized.split()) >= 3) and normalized in lowered and normalized not in original_lower:
            failures.append("option_copied")
            break
    return failures


@dataclass
class ValidationResult:
    candidate: str
    accepted: bool
    failures: list[str]
    similarity: float | None
    verifier_output: str | None


def validate_candidate(question, raw_candidate, previous, similarity_fn, verifier_fn, threshold,
                       prompt_encoder=None):
    candidate = normalize_whitespace(raw_candidate)
    failures = rule_failures(question.stem, raw_candidate, question.options, previous)
    similarity, verifier_output = None, None
    if not failures:
        similarity = float(similarity_fn(question.stem, candidate))
        if similarity < threshold:
            failures.append("similarity_below_threshold")
    if not failures and prompt_encoder is not None:
        from .prompting import PromptError
        try:
            prompt_encoder.encode(candidate, question.options)
        except PromptError as error:
            failures.append("prompt_tokenization: " + str(error))
    if not failures:
        verifier_output = verifier_fn(verifier_prompt(question, candidate)).strip()
        if verifier_output != "PASS":
            failures.append("semantic_or_label_verifier_failed")
    return ValidationResult(candidate, not failures, failures, similarity, verifier_output)

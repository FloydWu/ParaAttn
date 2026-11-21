from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .io import canonical_hash, read_jsonl

LABELS = "ABCD"


@dataclass
class Question:
    id: str
    stem: str
    options: list[str]
    label: int
    paraphrases: list[str] = field(default_factory=list)
    source_split: str = ""

    def validate(self, expected_paraphrases: int | None = None):
        if not self.id or not self.stem.strip():
            raise ValueError("Question ID and stem cannot be empty")
        if len(self.options) != 4 or any(not isinstance(o, str) or not o.strip() for o in self.options):
            raise ValueError(f"{self.id}: exactly four nonempty answer options are required")
        if len(set(self.options)) != 4 or isinstance(self.label, bool) or not isinstance(self.label, int) or self.label not in range(4):
            raise ValueError(f"{self.id}: malformed options or label")
        if expected_paraphrases is not None and len(self.paraphrases) != expected_paraphrases:
            raise ValueError(f"{self.id}: expected {expected_paraphrases} paraphrases")
        if any(not isinstance(p, str) for p in self.paraphrases):
            raise ValueError(f"{self.id}: paraphrases must be strings")
        stems = [" ".join(s.lower().split()) for s in [self.stem] + self.paraphrases]
        if any(not s for s in stems) or len(set(stems)) != len(stems):
            raise ValueError(f"{self.id}: empty or duplicate paraphrases")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def content_hash(self):
        return canonical_hash({"stem": self.stem, "options": self.options})

    @property
    def group_hash(self):
        return canonical_hash(self.to_dict())


def load_questions(path: str | Path, expected_paraphrases: int | None = None) -> list[Question]:
    rows = [Question(**row).validate(expected_paraphrases) for row in read_jsonl(path)]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    if len({q.id for q in rows}) != len(rows):
        raise ValueError(f"Duplicate question IDs in {path}")
    return rows


def check_disjoint(splits: dict[str, list[Question]]):
    seen_ids, seen_content = {}, {}
    for name, rows in splits.items():
        for q in rows:
            if q.id in seen_ids or q.content_hash in seen_content:
                raise ValueError(f"Cross-split overlap detected at {name}/{q.id}")
            seen_ids[q.id] = name
            seen_content[q.content_hash] = name

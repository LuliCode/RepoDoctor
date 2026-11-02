from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


def digest(content: str | bytes) -> str:
    return hashlib.sha256(content.encode("utf-8") if isinstance(content, str) else content).hexdigest()


@dataclass
class Passage:
    id: str
    project: str
    file_path: str
    class_name: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    snapshot: str = ""
    entities: list[str] = field(default_factory=list)
    structured: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class Location(BaseModel):
    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    symbol: str = ""
    explanation: str = ""


class Diagnosis(BaseModel):
    status: Literal["likely", "insufficient"] = "insufficient"
    locations: list[Location] = Field(default_factory=list)
    cause: str
    suggestions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


@dataclass
class Budget:
    max_tool_calls: int = 8
    max_seconds: float = 120
    max_tokens: int = 24000
    max_decisions: int = 12
    max_context_tokens: int = 12000

    def __post_init__(self):
        if (
            min(
                self.max_tool_calls,
                self.max_seconds,
                self.max_tokens,
                self.max_decisions,
                self.max_context_tokens,
            )
            <= 0
        ):
            raise ValueError("所有预算必须大于 0")

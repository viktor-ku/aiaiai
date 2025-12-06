from __future__ import annotations
from typing import Callable, List, Union
import random

PromptPart = Callable[[], str]
TextLike = Union[str, PromptPart]


def text(*pieces: TextLike) -> PromptPart:
    """Returns a PromptPart that concatenates strings or other PromptParts."""

    def _run() -> str:
        out_parts: List[str] = []
        for p in pieces:
            if isinstance(p, str):
                out_parts.append(p.strip())
            else:
                out_parts.append(p().strip())
        return "\n".join(filter(None, out_parts))

    return _run


def pick_one(*options: TextLike) -> PromptPart:
    """Randomly picks one option each time it's executed."""
    processed: List[PromptPart] = []
    for opt in options:
        if isinstance(opt, str):
            processed.append(text(opt))
        else:
            processed.append(opt)

    def _run() -> str:
        chosen = random.choice(processed)
        return chosen()

    return _run


def make_prompt(*parts: PromptPart) -> PromptPart:
    """
    Returns a PromptPart (callable) that, when executed,
    generates a *fresh* prompt each time.
    """

    def _run() -> str:
        return "\n".join(part().strip() for part in parts if part().strip())

    return _run

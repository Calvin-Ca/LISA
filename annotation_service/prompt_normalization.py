from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


PromptNormalizationMode = Literal[
    "off",
    "terminal_period",
    "canonical_terms",
]

PromptNormalizationProfile = Literal[
    "construction_safety_v1",
]

_TRAILING_PERIOD = " ."

_CONSTRUCTION_SAFETY_V1: tuple[tuple[str, str], ...] = (
    ("安全帽", "helmet"),
    ("头盔", "helmet"),
    ("hard hat", "helmet"),
    ("safety helmet", "helmet"),
    ("护栏", "guardrail"),
    ("栏杆", "guardrail"),
    ("guard rail", "guardrail"),
    ("临边防护", "guardrail"),
    ("安全带", "safety harness"),
    ("安全绳", "safety harness"),
    ("反光背心", "safety vest"),
    ("荧光背心", "safety vest"),
    ("安全背心", "safety vest"),
    ("挖掘机", "excavator"),
    ("吊车", "crane"),
    ("叉车", "forklift"),
    ("洞口", "opening"),
    ("楼板洞口", "opening"),
    ("开口", "opening"),
    ("临边", "platform edge"),
    ("材料堆", "construction material"),
    ("杂物", "debris"),
    ("通道", "walkway"),
)

PROMPT_NORMALIZATION_PROFILES: dict[
    PromptNormalizationProfile,
    tuple[tuple[str, str], ...],
] = {
    "construction_safety_v1": _CONSTRUCTION_SAFETY_V1,
}


@dataclass(frozen=True)
class PromptNormalizationResult:
    original_prompt: str
    normalized_prompt: str
    mode: PromptNormalizationMode
    profile: PromptNormalizationProfile | None
    applied_aliases: tuple[tuple[str, str], ...] = ()

    def as_metadata(self) -> dict[str, object]:
        return {
            "grounding_prompt_raw": self.original_prompt,
            "grounding_prompt_normalized": self.normalized_prompt,
            "grounding_prompt_normalization_mode": self.mode,
            "grounding_prompt_normalization_profile": self.profile,
            "grounding_prompt_applied_aliases": [
                {"source": source, "target": target}
                for source, target in self.applied_aliases
            ],
        }


def _ensure_terminal_period(prompt: str) -> str:
    return prompt if prompt.endswith(".") else prompt + _TRAILING_PERIOD


def _replace_aliases(
    prompt: str,
    *,
    profile: PromptNormalizationProfile,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    replacements = PROMPT_NORMALIZATION_PROFILES.get(profile)
    if replacements is None:
        raise ValueError(
            f"unsupported prompt normalization profile: {profile}"
        )
    normalized = prompt
    applied: list[tuple[str, str]] = []
    for source, target in sorted(
        replacements,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if source.isascii():
            pattern = re.compile(re.escape(source), re.IGNORECASE)
            updated, count = pattern.subn(target, normalized)
        else:
            updated = normalized.replace(source, target)
            count = 1 if updated != normalized else 0
        if count > 0:
            applied.append((source, target))
            normalized = updated
    return normalized, tuple(applied)


def normalize_grounding_prompt(
    prompt: str,
    *,
    mode: PromptNormalizationMode = "terminal_period",
    profile: PromptNormalizationProfile = "construction_safety_v1",
) -> PromptNormalizationResult:
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("GroundingDINO prompt must not be blank")
    if mode == "off":
        return PromptNormalizationResult(
            original_prompt=normalized,
            normalized_prompt=normalized,
            mode=mode,
            profile=None,
        )
    if mode == "terminal_period":
        return PromptNormalizationResult(
            original_prompt=normalized,
            normalized_prompt=_ensure_terminal_period(normalized),
            mode=mode,
            profile=None,
        )
    if mode == "canonical_terms":
        canonical, applied = _replace_aliases(
            normalized,
            profile=profile,
        )
        return PromptNormalizationResult(
            original_prompt=normalized,
            normalized_prompt=_ensure_terminal_period(canonical),
            mode=mode,
            profile=profile,
            applied_aliases=applied,
        )
    raise ValueError(
        f"unsupported prompt normalization mode: {mode}"
    )

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    label: str
    model_id: str


MODEL_CHOICES = (
    ModelChoice("Qwen 3.5 4B", "Qwen/Qwen3.5-4B"),
    ModelChoice("Qwen 3.5 9B", "Qwen/Qwen3.5-9B"),
    ModelChoice("Qwen 3.5 27B", "Qwen/Qwen3.5-27B"),
    ModelChoice("Gemma 4 E2B Instruct", "google/gemma-4-E2B-it"),
    ModelChoice("Gemma 4 E4B Instruct", "google/gemma-4-E4B-it"),
    ModelChoice("Gemma 4 12B Instruct", "google/gemma-4-12B-it"),
)

MODEL_IDS = {choice.label: choice.model_id for choice in MODEL_CHOICES}
QUANTIZATIONS = ("Auto / BF16", "FP16", "8-bit", "4-bit NF4")


def resolve_model_id(model: str, custom_model_id: str = "") -> str:
    custom = custom_model_id.strip()
    if custom:
        if "/" not in custom or any(character.isspace() for character in custom):
            raise ValueError("custom_model_id must be a Hugging Face ID such as owner/model-name")
        return custom
    try:
        return MODEL_IDS[model]
    except KeyError as error:
        raise ValueError(f"Unknown model selection: {model}") from error

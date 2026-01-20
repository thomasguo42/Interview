from __future__ import annotations

from typing import Optional

from .config import config
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient


def _normalize_model_name(model_name: Optional[str]) -> Optional[str]:
    if not model_name:
        return None
    return str(model_name).strip()


def get_llm_client(model_name: Optional[str]) -> GeminiClient | OpenAIClient:
    normalized = _normalize_model_name(model_name)
    provider = config.SUPPORTED_MODELS.get(normalized) if normalized else None
    if provider == "openai":
        return OpenAIClient(model=normalized)
    if provider == "gemini":
        return GeminiClient(model=normalized)
    # Fallback to Gemini default
    return GeminiClient(model=config.GEMINI_MODEL)

from __future__ import annotations

from app.config.settings import LLMProvider, LLMSettings
from app.generation.fake import FakeLLM
from app.generation.llm import LLMClient
from app.generation.openai_compatible import OpenAICompatibleClient


def build_llm(settings: LLMSettings) -> LLMClient:
    match settings.llm_provider:
        case LLMProvider.OPENAI_COMPATIBLE:
            return OpenAICompatibleClient(settings)
        case LLMProvider.FAKE:
            return FakeLLM(model=settings.llm_model)

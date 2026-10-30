"""LLM provider abstraction.

Multiple providers are supported behind a single `LLMProvider` interface:

  - Anthropic  (native SDK + native tool-calling)  ← default
  - DeepSeek   (OpenAI-compatible API)
  - Gemini     (OpenAI-compatible endpoint)
  - HuggingFace (text-generation / inference endpoints, OpenAI-compatible)
  - Custom     (any user-supplied OpenAI-compatible base_url + key)

The agent loop is provider-agnostic: it talks to a `LLMProvider`, which
returns a normalized `LLMResponse`. Providers that support native tool
calls use them; providers that don't fall back to the structured-output
protocol (the agent parses ```tool_call blocks from the text response).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..messages import LLMMessage, LLMResponse


@dataclass
class ProviderConfig:
    """Settings for a single provider instance.

    `api_key` is read from the matching environment variable if not set
    explicitly (e.g. ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, …).
    """

    provider: str = "anthropic"  # one of PROVIDER_IDS
    model: str = ""
    api_key: str = ""
    base_url: str = ""           # optional override (used by "custom")
    #: When True, the provider exposes native tool-calling; the agent loop
    #: will pass tools as structured function calls instead of relying on
    #: the model to emit ```tool_call blocks in prose.
    native_tools: bool = True
    #: Max tokens for the assistant response.
    max_tokens: int = 4096
    temperature: float = 0.7
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Registry of known providers
# ---------------------------------------------------------------------- #

#: Each entry: (label, id, default_model, env_var_for_key, default_base_url)
KNOWN_PROVIDERS: list[dict[str, str]] = [
    {
        "id": "anthropic",
        "label": "Anthropic",
        "default_model": "claude-3-5-sonnet-latest",
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": "",  # SDK default
        "docs": "https://docs.anthropic.com",
    },
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "docs": "https://api-docs.deepseek.com",
    },
    {
        "id": "gemini",
        "label": "Google Gemini",
        "default_model": "gemini-1.5-flash",
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "docs": "https://ai.google.dev/gemini-api/docs/openai",
    },
    {
        "id": "huggingface",
        "label": "HuggingFace",
        "default_model": "meta-llama/Llama-3.1-8B-Instruct",
        "env_key": "HF_TOKEN",
        "base_url": "https://api-inference.huggingface.co/v1",
        "docs": "https://huggingface.co/docs/api-inference",
    },
    {
        "id": "custom",
        "label": "Custom provider",
        "default_model": "",
        "env_key": "",
        "base_url": "",
        "docs": "",
    },
]

PROVIDER_IDS = [p["id"] for p in KNOWN_PROVIDERS]


def provider_meta(provider_id: str) -> dict[str, str]:
    """Return the metadata dict for a provider id (or empty dict)."""
    for p in KNOWN_PROVIDERS:
        if p["id"] == provider_id:
            return p
    return {}


def default_model_for(provider_id: str) -> str:
    return provider_meta(provider_id).get("default_model", "")


def env_key_for(provider_id: str) -> str:
    return provider_meta(provider_id).get("env_key", "")


# ---------------------------------------------------------------------- #
# Provider interface
# ---------------------------------------------------------------------- #


class LLMProvider:
    """Interface every provider implements."""

    config: ProviderConfig

    def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        tools_schema: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """Return the assistant's next message (text or tool calls)."""
        raise NotImplementedError

    @classmethod
    def create(cls, config: ProviderConfig) -> "LLMProvider":
        """Factory: build the right provider subclass for `config.provider`."""
        pid = (config.provider or "anthropic").lower()
        if pid == "anthropic":
            from .anthropic_provider import AnthropicProvider
            return AnthropicProvider(config)
        if pid in ("deepseek", "gemini", "huggingface", "custom"):
            from .openai_compat_provider import OpenAICompatProvider
            return OpenAICompatProvider(config)
        raise ValueError(f"Unknown provider: {pid!r}. Choose one of: {PROVIDER_IDS}")


__all__ = [
    "ProviderConfig",
    "KNOWN_PROVIDERS",
    "PROVIDER_IDS",
    "LLMProvider",
    "provider_meta",
    "default_model_for",
    "env_key_for",
]

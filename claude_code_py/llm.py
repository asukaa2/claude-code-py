"""LLM client facade over the provider abstraction.

The shared message types (`LLMMessage`, `LLMResponse`) live in `messages.py`
to avoid a circular import between this module and `providers/`. They are
re-exported here for backward compatibility.

`LLMClient` is a small facade that holds a `ProviderConfig` and delegates to
the right `LLMProvider` subclass. Anthropic is the default; DeepSeek, Gemini,
HuggingFace, and Custom OpenAI-compatible endpoints are also supported.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from .messages import LLMMessage, LLMResponse  # re-exported below
from .providers import (
    KNOWN_PROVIDERS,
    PROVIDER_IDS,
    LLMProvider,
    ProviderConfig,
    default_model_for,
    env_key_for,
)


# ---------------------------------------------------------------------- #
# LLMClient — facade over the provider abstraction
# ---------------------------------------------------------------------- #


class LLMClient:
    """Holds a `ProviderConfig` and delegates to a `LLMProvider`.

    Rebuilt by the agent whenever the config changes (the web UI swaps
    providers at runtime). The provider instance is constructed lazily on
    the first `complete()` call so a missing SDK doesn't crash imports.
    """

    def __init__(self, provider_config: ProviderConfig) -> None:
        self.config = provider_config
        self._provider: Optional[LLMProvider] = None
        # Default the model if the caller didn't set one.
        if not self.config.model:
            self.config.model = default_model_for(self.config.provider)

    @property
    def provider(self) -> LLMProvider:
        """Lazily construct the underlying provider."""
        if self._provider is None:
            self._provider = LLMProvider.create(self.config)
        return self._provider

    def complete(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        tools_schema: Optional[list[dict]] = None,
    ) -> LLMResponse:
        """Send `messages` + `system_prompt` to the LLM, return its reply.

        `tools_schema` is an optional list of JSON-schema dicts describing
        the available tools; providers that support native tool-calling
        will pass them through, others will ignore them and rely on the
        agent's structured-output fallback.
        """
        return self.provider.complete(messages, system_prompt, tools_schema)


__all__ = [
    "LLMMessage",
    "LLMResponse",
    "LLMClient",
    "ProviderConfig",
    "KNOWN_PROVIDERS",
    "PROVIDER_IDS",
    "default_model_for",
    "env_key_for",
]

"""Shared configuration for all Engram entrypoints (CLI, MCP, HTTP server).

All settings are read from environment variables. Every LLM component
(extractor, embedder, meaning analyzer) defaults to a local zero-cost
implementation and can be swapped to any supported provider.

Supported Providers
-------------------

Extractor (ENGRAM_EXTRACTOR):
  null     — NullExtractor, no extraction ($0, default)
  openai   — OpenAIExtractor, uses chat completions API

Embedder (ENGRAM_EMBEDDER):
  hash     — HashEmbedder, local character n-gram hashing ($0, default)
  openai   — OpenAIEmbedder, uses embeddings API

Meaning Analyzer (ENGRAM_MEANING_ANALYZER):
  null     — NullMeaningAnalyzer, no meaning analysis ($0, default)
  openai   — OpenAIMeaningAnalyzer, uses chat completions API

Model Selection
---------------
Each provider reads its own model env var. Users can point at any
model their provider supports — including proxies that expose
Claude, Gemini, or open-source models via an OpenAI-compatible API.

Environment Variables
---------------------
ENGRAM_USER_ID              User identifier (default: "default")
ENGRAM_PATH                 Storage directory (default: ~/.engram/users)
ENGRAM_SESSION_ID           Optional session tag
ENGRAM_AUTO_FLUSH           Enable background worker (default: "true" for server/MCP, "false" for CLI)

ENGRAM_EXTRACTOR            Provider name: null | openai
ENGRAM_EMBEDDER             Provider name: hash | openai
ENGRAM_MEANING_ANALYZER     Provider name: null | openai

OPENAI_API_KEY              API key for OpenAI-compatible providers
ENGRAM_OPENAI_BASE_URL      Custom base URL (for proxies, Azure, local servers)
ENGRAM_OPENAI_MODEL         Chat model for extractor (default: gpt-5.4-mini)
ENGRAM_OPENAI_EMBED_MODEL   Embedding model (default: text-embedding-3-small)
ENGRAM_OPENAI_EMBED_DIMS    Embedding dimensions override
ENGRAM_OPENAI_MEANING_MODEL Chat model for meaning analyzer (default: gpt-5.4-mini)

Proxy Examples
--------------
# Use Claude via OpenAI-compatible proxy (e.g. LiteLLM, OpenRouter)
ENGRAM_OPENAI_BASE_URL=https://openrouter.ai/api/v1
ENGRAM_OPENAI_MODEL=anthropic/claude-sonnet-4.6
OPENAI_API_KEY=sk-or-...

# Use Gemini via OpenAI-compatible proxy
ENGRAM_OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
ENGRAM_OPENAI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=AIza...

# Use local model (Ollama, vLLM, etc.)
ENGRAM_OPENAI_BASE_URL=http://localhost:11434/v1
ENGRAM_OPENAI_MODEL=llama3
OPENAI_API_KEY=unused
"""

from __future__ import annotations

import os

from .canonical import Extractor, NullExtractor
from .meaning_index import MeaningAnalyzer, NullMeaningAnalyzer
from .semantic import Embedder, HashEmbedder

_DEFAULT_CHAT_MODEL = "gpt-5.4-mini"
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def build_extractor() -> Extractor:
    name = os.environ.get("ENGRAM_EXTRACTOR", "null")
    if name == "null":
        return NullExtractor()
    if name == "openai":
        from .openai_extractor import OpenAIExtractor

        return OpenAIExtractor(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ENGRAM_OPENAI_MODEL", _DEFAULT_CHAT_MODEL),
            base_url=os.environ.get("ENGRAM_OPENAI_BASE_URL"),
        )
    raise ValueError(
        f"Unknown extractor: {name!r}. Supported: null, openai"
    )


def build_embedder() -> Embedder:
    name = os.environ.get("ENGRAM_EMBEDDER", "hash")
    if name == "hash":
        return HashEmbedder()
    if name == "openai":
        from .semantic import OpenAIEmbedder

        dims = os.environ.get("ENGRAM_OPENAI_EMBED_DIMS")
        return OpenAIEmbedder(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ENGRAM_OPENAI_EMBED_MODEL", _DEFAULT_EMBED_MODEL),
            dimensions=int(dims) if dims else None,
            base_url=os.environ.get("ENGRAM_OPENAI_BASE_URL"),
        )
    raise ValueError(
        f"Unknown embedder: {name!r}. Supported: hash, openai"
    )


def build_meaning_analyzer() -> MeaningAnalyzer:
    name = os.environ.get("ENGRAM_MEANING_ANALYZER", "null")
    if name == "null":
        return NullMeaningAnalyzer()
    if name == "openai":
        from .openai_meaning_analyzer import OpenAIMeaningAnalyzer

        return OpenAIMeaningAnalyzer(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ENGRAM_OPENAI_MEANING_MODEL", _DEFAULT_CHAT_MODEL),
            base_url=os.environ.get("ENGRAM_OPENAI_BASE_URL"),
        )
    raise ValueError(
        f"Unknown meaning analyzer: {name!r}. Supported: null, openai"
    )

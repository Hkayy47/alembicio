"""Live provider smoke test (skipped without API keys)."""

from __future__ import annotations

import os
import sys


def main() -> int:
    """Run a minimal embed against whichever provider keys exist."""
    if os.environ.get("OPENAI_API_KEY"):
        from alembicio.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        matrix = provider.embed_batch(["hello"], model="text-embedding-3-small", dims=3)
        assert matrix.shape[0] == 1
        print("openai ok", matrix.shape)
        return 0

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        from alembicio.providers.gemini import GeminiProvider

        provider = GeminiProvider()
        matrix = provider.embed_batch(["hello"], model="gemini-embedding-001", dims=3)
        assert matrix.shape[0] == 1
        print("gemini ok", matrix.shape)
        return 0

    print("skip: no provider API keys in environment", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

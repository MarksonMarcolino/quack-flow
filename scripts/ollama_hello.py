"""Ollama integration sanity check.

Runs one llama3 request that mimics the shape of the real explanation task
we'll write in weekend 7: a SHAP-style feature list in, a plain-English
explanation out. Exists to prove the integration works end-to-end before
any real pipeline code depends on it.

Run:
    uv run python scripts/ollama_hello.py

The verbatim output of this script is committed into README.md so a
skeptical reader sees that Ollama genuinely ran against the project, not
just that the code compiles.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = "llama3"

# Stand-in for the kind of SHAP output the real explainer will pass in
# during weekend 7. Fake numbers, fake context — but the prompt shape is
# the one train_stock.py + explain.py will produce.
FAKE_SHAP_CONTEXT: dict[str, Any] = {
    "ticker": "MSFT",
    "prediction": "UP in 7 days",
    "confidence": 0.61,
    "top_features": [
        ("commit_velocity_7d_python_repos", 0.18),
        ("star_growth_7d_ai_tagged", 0.11),
        ("pr_merge_rate_7d_azure_repos", -0.07),
    ],
}

PROMPT = f"""You are explaining a stock prediction to a data-engineer who
understands ML but is skeptical. Be honest, avoid hype. Mention that we're
looking at correlation, not causation, and that GitHub activity leading
stock moves would be an unusual finding.

Prediction:
    Ticker: {FAKE_SHAP_CONTEXT['ticker']}
    Direction: {FAKE_SHAP_CONTEXT['prediction']}
    Confidence: {FAKE_SHAP_CONTEXT['confidence']:.0%}

Top contributing features (feature name, SHAP value):
{chr(10).join(f"    - {name}: {val:+.3f}" for name, val in FAKE_SHAP_CONTEXT['top_features'])}

Explain the prediction in 2-3 short sentences. End with one caveat about
what this does NOT prove."""


def call_ollama() -> str:
    """POST one request to the local Ollama server and return the text."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "stream": False,
            "options": {"num_predict": 120},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def main() -> int:
    try:
        text = call_ollama()
    except requests.exceptions.ConnectionError:
        print(
            f"ERROR: could not reach Ollama at {OLLAMA_HOST}.\n"
            "Start it with: brew services start ollama  (or `ollama serve`)",
            file=sys.stderr,
        )
        return 2
    except requests.HTTPError as e:
        print(f"ERROR: Ollama HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        return 3

    print("=" * 60)
    print(f"quack-flow Ollama hello-world  ({MODEL} @ {OLLAMA_HOST})")
    print("=" * 60)
    print()
    print("INPUT (fake SHAP context):")
    print(json.dumps(FAKE_SHAP_CONTEXT, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o))
    print()
    print("OUTPUT (llama3):")
    print(text)
    print()
    print("=" * 60)
    print("Integration sanity check: PASS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

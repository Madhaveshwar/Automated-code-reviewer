"""Groq-backed code review helper."""

from __future__ import annotations

import os

from groq import Groq
from langsmith import Client
from langsmith import traceable
import streamlit as st

from utils.prompts import SYSTEM_PROMPT, build_review_prompt

MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_TEMPERATURE = 0.3
MODEL_MAX_TOKENS = 1024
PLACEHOLDER_VALUES = {
    "your_groq_api_key",
    "your_existing_langsmith_key",
}


def _get_config_value(name: str) -> str | None:
    """Resolve a configuration value from Streamlit secrets first, then env."""

    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    value = os.getenv(name)
    if value:
        if value.strip() in PLACEHOLDER_VALUES or value.strip().startswith("your_"):
            return None
        return value

    return None


def _is_truthy(value: str | None) -> bool:
    """Return True when a configuration value enables a feature."""

    if value is None:
        return False

    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_runtime_status() -> dict[str, object]:
    """Expose runtime configuration state for Streamlit startup validation."""

    langchain_project = _get_config_value("LANGCHAIN_PROJECT") or "Automated_Code_Reviewer"
    return {
        "groq_api_key_present": bool(_get_config_value("GROQ_API_KEY")),
        "langchain_api_key_present": bool(_get_config_value("LANGCHAIN_API_KEY")),
        "langchain_tracing_enabled": _is_truthy(_get_config_value("LANGCHAIN_TRACING_V2")),
        "langchain_project": langchain_project,
    }


def _build_client() -> Groq:
    """Create a Groq client after validating the API key."""
    api_key = _get_config_value("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it to Streamlit secrets or your local .env file."
        )

    return Groq(api_key=api_key)


def _build_langsmith_client() -> Client | None:
    """Create a LangSmith client when the API key is available."""

    api_key = _get_config_value("LANGCHAIN_API_KEY")
    if not api_key:
        return None

    return Client(api_key=api_key)


@traceable
def langsmith_test() -> str:
    """Emit a small startup trace to confirm LangSmith tracing is available."""

    _build_langsmith_client()
    return "LangSmith tracing works"


@traceable
def traced_build_review_prompt(code: str, language: str) -> str:
    """Trace the prompt input and selected language."""

    return build_review_prompt(code, language)


@traceable
def capture_review_metadata(
    language: str,
    prompt: str,
    generated_review: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, object]:
    """Trace model metadata together with the generated review."""

    return {
        "programming_language": language,
        "prompt": prompt,
        "generated_review": generated_review,
        "model_metadata": {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }


@traceable
def review_code(code: str, language: str) -> str:
    """Analyze code and return a structured markdown review."""
    client = _build_client()
    prompt = traced_build_review_prompt(code, language)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=MODEL_TEMPERATURE,
        max_tokens=MODEL_MAX_TOKENS
    )

    review_markdown = response.choices[0].message.content
    capture_review_metadata(
        language=language,
        prompt=prompt,
        generated_review=review_markdown,
        model_name=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
        max_tokens=MODEL_MAX_TOKENS,
    )

    return review_markdown

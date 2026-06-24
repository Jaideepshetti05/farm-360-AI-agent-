# -*- coding: utf-8 -*-
"""
Farm360 AI — Multi-Provider Key Manager (Phase 1: Native Gemini SDK)
=====================================================================
Handles round-robin rotation and automatic fallback across:
  • Google Gemini  (GOOGLE_API_KEY_1 … _5)  — uses native google-genai SDK
  • OpenRouter     (OPENROUTER_API_KEY_1 … _5) — uses OpenAI SDK
  • OpenAI         (OPENAI_API_KEY_1 … _3)     — uses OpenAI SDK

Key rotation rules:
  ROTATE on:  HTTP 429, quota exceeded, resource exhausted, rate limit
  DO NOT ROTATE on: HTTP 401, invalid API key, auth failures, bad endpoint

Usage:
    from key_manager import key_manager

    # For streaming (Gemini-native or OpenAI-compat):
    for token in key_manager.stream_completion(messages):
        print(token, end="")
"""

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Generator
from loguru import logger

# ── Provider SDKs ──────────────────────────────────────────────────────────────
from google import genai                          # Native Gemini SDK
from google.genai import types as gemini_types
from openai import OpenAI                         # OpenRouter / OpenAI


# ── Constants ──────────────────────────────────────────────────────────────────
COOLDOWN_SECONDS = 60          # How long a rate-limited key is quarantined
GEMINI_MODEL = "gemini-2.5-flash"


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class KeyEntry:
    key: str
    provider: str                          # "gemini" | "openrouter" | "openai"
    failures: int = 0
    cooldown_until: float = 0.0            # epoch seconds
    last_error: Optional[str] = None
    permanently_disabled: bool = False

    @property
    def is_available(self) -> bool:
        if self.permanently_disabled:
            return False
        return time.time() >= self.cooldown_until and bool(self.key)

    def mark_rate_limited(self, error_msg: str):
        """Mark key as temporarily rate-limited (rotatable error)."""
        self.failures += 1
        self.cooldown_until = time.time() + COOLDOWN_SECONDS
        self.last_error = error_msg
        logger.warning(
            f"[KeyManager] Key …{self.key[-8:]} ({self.provider}) RATE-LIMITED "
            f"({self.failures}x). Cooling down for {COOLDOWN_SECONDS}s. "
            f"Error: {error_msg}"
        )

    def mark_auth_failed(self, error_msg: str):
        """Mark key as permanently disabled (fatal auth error — do NOT rotate)."""
        self.permanently_disabled = True
        self.last_error = error_msg
        logger.error(
            f"[KeyManager] Key …{self.key[-8:]} ({self.provider}) PERMANENTLY DISABLED. "
            f"Fatal auth error (will NOT rotate): {error_msg}"
        )

    def mark_success(self):
        self.failures = 0
        self.cooldown_until = 0.0
        self.last_error = None


# ── Provider configs ───────────────────────────────────────────────────────────
_PROVIDER_CONFIGS: dict[str, dict] = {
    "gemini": {
        "default_model": GEMINI_MODEL,
        "label": "Google Gemini (native SDK)",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemma-4-27b-it:free",
        "label": "OpenRouter",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "label": "OpenAI",
    },
}


# ── Error classification ──────────────────────────────────────────────────────
def _is_rotatable_error(error: Exception) -> bool:
    """Return True if the error indicates a rate-limit / quota issue (should rotate)."""
    err_str = str(error).lower()
    rotatable_keywords = [
        "429", "rate limit", "rate_limit", "quota", "resource exhausted",
        "resource_exhausted", "too many requests", "overloaded",
    ]
    return any(kw in err_str for kw in rotatable_keywords)


def _is_fatal_auth_error(error: Exception) -> bool:
    """Return True if the error indicates a permanent auth failure (should NOT rotate)."""
    err_str = str(error).lower()
    fatal_keywords = [
        "401", "403", "invalid api key", "api_key_invalid", "invalid_api_key",
        "authentication", "permission denied", "permission_denied",
        "api key not valid", "api key expired",
    ]
    return any(kw in err_str for kw in fatal_keywords)


# ── Key loader ─────────────────────────────────────────────────────────────────
def _load_keys_from_env(prefix: str, count: int, provider: str) -> list[KeyEntry]:
    """Read KEY_1 … KEY_<count> from env; skip blank ones."""
    entries = []
    for i in range(1, count + 1):
        val = os.environ.get(f"{prefix}_{i}", "").strip()
        if val:
            entries.append(KeyEntry(key=val, provider=provider))
    return entries


# ── Main manager ───────────────────────────────────────────────────────────────
class MultiProviderKeyManager:
    """
    Thread-safe manager that rotates API keys across multiple providers.
    Falls through providers in priority order: Gemini → OpenRouter → OpenAI.
    Uses native google-genai SDK for Gemini, OpenAI SDK for others.
    """

    PROVIDER_ORDER = ["gemini", "openrouter", "openai"]

    def __init__(self):
        self._lock = threading.Lock()
        self._pools: dict[str, list[KeyEntry]] = {}
        self._indices: dict[str, int] = {}
        self._active_provider: Optional[str] = None
        self._active_key_entry: Optional[KeyEntry] = None
        self._last_error: Optional[str] = None
        self._load_all_keys()
        self._print_startup_diagnostics()

    # ── Initialisation ─────────────────────────────────────────────────────────
    def _load_all_keys(self):
        self._pools["gemini"]     = _load_keys_from_env("GOOGLE_API_KEY",      5, "gemini")
        self._pools["openrouter"] = _load_keys_from_env("OPENROUTER_API_KEY",  5, "openrouter")
        self._pools["openai"]     = _load_keys_from_env("OPENAI_API_KEY",      3, "openai")

        for provider in self.PROVIDER_ORDER:
            self._indices[provider] = 0

    def _print_startup_diagnostics(self):
        """Print formatted startup diagnostics table."""
        logger.info("=" * 60)
        logger.info("  Farm360 AI — Provider Status")
        logger.info("=" * 60)

        for provider in self.PROVIDER_ORDER:
            count = len(self._pools[provider])
            label = _PROVIDER_CONFIGS[provider]["label"]
            model = _PROVIDER_CONFIGS[provider]["default_model"]
            status_icon = "✅" if count > 0 else "❌"
            logger.info(f"  {status_icon} {label}: {count} key(s) loaded  [model: {model}]")

        # Determine active provider
        result = self._find_first_available()
        if result:
            provider, entry = result
            self._active_provider = provider
            self._active_key_entry = entry
            label = _PROVIDER_CONFIGS[provider]["label"]
            logger.success(f"  Active Provider: {label}")
            logger.success(f"  Active Key Index: {self._indices[provider]}")
        else:
            self._active_provider = None
            self._active_key_entry = None
            logger.warning("  Active Provider: OFFLINE MODE (no keys configured)")

        logger.info("=" * 60)

    # ── Key selection ──────────────────────────────────────────────────────────
    def _next_available_key(self, provider: str) -> Optional[KeyEntry]:
        """Return the next available (non-cooling, non-disabled) key for a provider."""
        pool = self._pools.get(provider, [])
        if not pool:
            return None

        start_idx = self._indices[provider]
        for offset in range(len(pool)):
            idx = (start_idx + offset) % len(pool)
            entry = pool[idx]
            if entry.is_available:
                self._indices[provider] = (idx + 1) % len(pool)
                return entry

        return None

    def _find_first_available(self) -> Optional[tuple[str, KeyEntry]]:
        """Walk provider priority and return first available (provider, KeyEntry)."""
        for provider in self.PROVIDER_ORDER:
            entry = self._next_available_key(provider)
            if entry:
                return provider, entry
        return None

    def get_active_provider_and_key(self) -> Optional[tuple[str, KeyEntry]]:
        with self._lock:
            result = self._find_first_available()
            if result:
                self._active_provider = result[0]
                self._active_key_entry = result[1]
            return result

    # ── Gemini native streaming ────────────────────────────────────────────────
    def _stream_gemini(self, entry: KeyEntry, messages: list[dict],
                       temperature: float = 0.7,
                       max_tokens: int = 2048) -> Generator[str, None, None]:
        """
        Stream a completion using the native google-genai SDK.
        Converts OpenAI-style messages to Gemini format.
        """
        client = genai.Client(api_key=entry.key)

        # Convert OpenAI-style messages to Gemini contents
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(gemini_types.Content(
                    role="user",
                    parts=[gemini_types.Part(text=content)],
                ))
            elif role == "assistant":
                contents.append(gemini_types.Content(
                    role="model",
                    parts=[gemini_types.Part(text=content)],
                ))

        config = gemini_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        logger.info(
            f"[Gemini Native] Streaming with model={GEMINI_MODEL} "
            f"key=…{entry.key[-8:]}"
        )

        response = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    # ── OpenAI-compat streaming (OpenRouter / OpenAI) ──────────────────────────
    def _stream_openai_compat(self, entry: KeyEntry, provider: str,
                               messages: list[dict],
                               temperature: float = 0.7,
                               max_tokens: int = 2048) -> Generator[str, None, None]:
        """Stream a completion using the OpenAI SDK (for OpenRouter / OpenAI)."""
        cfg = _PROVIDER_CONFIGS[provider]

        extra_headers = {}
        if provider == "openrouter":
            extra_headers = {
                "HTTP-Referer": "https://farm360.app",
                "X-Title": "Farm360 AI",
            }

        client = OpenAI(
            base_url=cfg["base_url"],
            api_key=entry.key,
            timeout=20.0,
            default_headers=extra_headers if extra_headers else None,
        )

        model = cfg["default_model"]
        logger.info(
            f"[{cfg['label']}] Streaming with model={model} "
            f"key=…{entry.key[-8:]}"
        )

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Unified streaming interface ────────────────────────────────────────────
    def stream_completion(self, messages: list[dict],
                          temperature: float = 0.7,
                          max_tokens: int = 2048) -> Generator[str, None, None]:
        """
        Stream a completion, automatically selecting the best available provider.
        Handles rotation on rate-limit errors and stops on auth errors.
        """
        MAX_ATTEMPTS = 6  # Enough to try several keys across providers

        for attempt in range(MAX_ATTEMPTS):
            result = self.get_active_provider_and_key()

            if result is None:
                error_msg = (
                    "⚠️ **No API Keys Available**\n\n"
                    "All configured keys are exhausted or disabled. "
                    "Please add valid keys to your `.env` file."
                )
                logger.error(f"[KeyManager] {error_msg}")
                self._last_error = "No API keys available"
                yield error_msg
                return

            provider, entry = result

            try:
                if provider == "gemini":
                    gen = self._stream_gemini(entry, messages, temperature, max_tokens)
                else:
                    gen = self._stream_openai_compat(entry, provider, messages, temperature, max_tokens)

                for token in gen:
                    yield token

                # If we got here, streaming succeeded
                with self._lock:
                    entry.mark_success()
                    self._last_error = None
                return  # Success — exit

            except Exception as e:
                err_str = str(e)
                label = _PROVIDER_CONFIGS[provider]["label"]

                # ALWAYS log the exact error — no silent fallbacks
                logger.error(
                    f"[KeyManager] {label} key …{entry.key[-8:]} FAILED "
                    f"(attempt {attempt + 1}/{MAX_ATTEMPTS}): {err_str}"
                )

                if _is_fatal_auth_error(e):
                    with self._lock:
                        entry.mark_auth_failed(err_str)
                    self._last_error = f"[{label}] AUTH FATAL: {err_str}"
                    # Do NOT retry — this key is permanently bad
                    # But continue the loop to try other keys/providers
                    continue

                elif _is_rotatable_error(e):
                    with self._lock:
                        entry.mark_rate_limited(err_str)
                    self._last_error = f"[{label}] RATE LIMITED: {err_str}"
                    # Continue to try next key
                    continue

                else:
                    # Unknown error — log it explicitly and try next
                    self._last_error = f"[{label}] UNKNOWN: {err_str}"
                    with self._lock:
                        entry.mark_rate_limited(err_str)
                    continue

        # All attempts exhausted
        error_msg = (
            f"⚠️ **LLM Request Failed After {MAX_ATTEMPTS} Attempts**\n\n"
            f"Last error: `{self._last_error}`\n\n"
            "All available keys were tried."
        )
        yield error_msg

    # ── Legacy client factory (backward compat for Phase 1) ────────────────────
    def get_client(self) -> tuple[Optional[OpenAI], Optional[str], Optional[KeyEntry]]:
        """
        Legacy method — returns an OpenAI-compatible client.
        For Gemini, this now returns None — use stream_completion() instead.
        """
        result = self.get_active_provider_and_key()
        if not result:
            logger.error("[KeyManager] No API keys available across any provider!")
            return None, None, None

        provider, entry = result
        if provider == "gemini":
            # Can't create OpenAI client for Gemini native — return markers
            logger.debug(f"[KeyManager] Gemini key selected — use stream_completion() for native streaming")
            return None, provider, entry

        cfg = _PROVIDER_CONFIGS[provider]
        extra_headers = {}
        if provider == "openrouter":
            extra_headers = {
                "HTTP-Referer": "https://farm360.app",
                "X-Title": "Farm360 AI",
            }

        client = OpenAI(
            base_url=cfg["base_url"],
            api_key=entry.key,
            timeout=20.0,
            default_headers=extra_headers if extra_headers else None,
        )
        return client, provider, entry

    def get_default_model(self, provider: str) -> str:
        return _PROVIDER_CONFIGS.get(provider, {}).get("default_model", GEMINI_MODEL)

    # ── Feedback (legacy compat) ───────────────────────────────────────────────
    def mark_key_failed(self, entry: KeyEntry):
        with self._lock:
            entry.mark_rate_limited(entry.last_error or "Unknown error")

    def mark_key_success(self, entry: KeyEntry):
        with self._lock:
            entry.mark_success()

    # ── Health / Status ────────────────────────────────────────────────────────
    def status(self) -> dict:
        """Return a snapshot of all key pool health (redacts key values)."""
        snapshot = {}
        now = time.time()
        with self._lock:
            for provider, pool in self._pools.items():
                snapshot[provider] = [
                    {
                        "key_suffix": f"…{e.key[-8:]}",
                        "available": e.is_available,
                        "failures": e.failures,
                        "permanently_disabled": e.permanently_disabled,
                        "cooldown_remaining_s": max(0.0, round(e.cooldown_until - now, 1)),
                        "last_error": e.last_error,
                    }
                    for e in pool
                ]
        return snapshot

    def health_status(self) -> dict:
        """Return health info for the /api/health/providers endpoint."""
        provider = self._active_provider
        if provider is None:
            return {
                "provider": "offline",
                "activeKey": None,
                "healthy": False,
                "model": None,
                "lastError": self._last_error or "No API keys configured",
            }

        entry = self._active_key_entry
        cfg = _PROVIDER_CONFIGS.get(provider, {})
        return {
            "provider": provider,
            "activeKey": f"…{entry.key[-8:]}" if entry else None,
            "healthy": entry.is_available if entry else False,
            "model": cfg.get("default_model"),
            "lastError": self._last_error,
        }

    @property
    def has_any_key(self) -> bool:
        return any(bool(pool) for pool in self._pools.values())


# ── Singleton ──────────────────────────────────────────────────────────────────
provider_manager = MultiProviderKeyManager()

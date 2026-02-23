"""
ollama_manager.py
=================
Manager for local Ollama LLM inference.

Server  : http://localhost:11434  (Ollama default)
Models  : llama3.2:latest | gemma3:4b

Endpoints used:
  GET  /api/tags          — list available models + health check
  POST /api/chat          — multi-turn chat with message history
  POST /api/generate      — single-turn generation (no history)

Features:
  - Model switching at runtime (llama3.2 ↔ gemma3:4b)
  - Per-session conversation history (multi-turn memory)
  - Configurable system prompt per model
  - Streaming support (yields delta chunks)
  - Health check and model availability validation
  - Graceful error messages for every failure mode

Usage from app.py:
  from ollama_manager import ollama_manager
  response = ollama_manager.chat("Hello!", model="llama3.2", session_id="user-ip")
"""

import json
import requests
from typing import Iterator


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"

# Canonical model names as registered in Ollama
SUPPORTED_MODELS: dict[str, str] = {
    "llama3.2":  "llama3.2:latest",
    "gemma3":    "gemma3:4b",
}

# Per-model system prompts — tune these to give each model a distinct personality
SYSTEM_PROMPTS: dict[str, str] = {
    "llama3.2:latest": (
        "You are a friendly, concise AI assistant built into a chatbot web app. "
        "Answer clearly and helpfully. Keep responses short unless detail is requested. "
        "Use plain text — no markdown unless asked."
    ),
    "gemma3:4b": (
        "You are Gemma, a thoughtful and knowledgeable AI assistant embedded in a chatbot. "
        "Provide accurate, well-structured answers. Be warm and conversational. "
        "Use plain text — no markdown unless asked."
    ),
}

DEFAULT_MODEL = "llama3.2:latest"
TIMEOUT_SECONDS = 90          # generation can take time on CPU
HEALTH_TIMEOUT  = 3           # fast fail for availability checks
MAX_HISTORY     = 20          # max messages kept per session (rolling window)


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL OLLAMA CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Thin HTTP wrapper around the Ollama REST API.
    Use OllamaManager for higher-level features.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ── health ────────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            r = self._session.get(f"{self.base_url}/api/tags", timeout=HEALTH_TIMEOUT)
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> list[dict]:
        """
        Return a list of model dicts from the Ollama server.
        Each dict has keys: name, model, size, details, …
        """
        r = self._session.get(f"{self.base_url}/api/tags", timeout=HEALTH_TIMEOUT)
        r.raise_for_status()
        return r.json().get("models", [])

    def model_names(self) -> list[str]:
        """Return just the model name strings currently pulled on the server."""
        return [m["name"] for m in self.list_models()]

    # ── chat (multi-turn) ─────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        options: dict | None = None,
    ) -> dict | Iterator[str]:
        """
        POST /api/chat

        Parameters
        ----------
        messages : list of {"role": "system"|"user"|"assistant", "content": str}
        model    : ollama model name string
        stream   : if True, yield delta strings; if False return full response dict
        options  : optional model params, e.g. {"temperature": 0.7, "num_ctx": 4096}

        Returns
        -------
        Non-stream : full response dict from Ollama
        Stream     : generator yielding content delta strings
        """
        payload: dict = {
            "model":    model,
            "messages": messages,
            "stream":   stream,
        }
        if options:
            payload["options"] = options

        if stream:
            return self._stream_chat(payload)

        r = self._session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        return r.json()

    def _stream_chat(self, payload: dict) -> Iterator[str]:
        """Yield content deltas for a streaming chat request."""
        with self._session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=TIMEOUT_SECONDS,
        ) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    yield delta
                if chunk.get("done"):
                    break

    # ── generate (single-turn) ────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        stream: bool = False,
        options: dict | None = None,
    ) -> str:
        """
        POST /api/generate — stateless single-turn generation.
        Returns the full response string (non-streaming mode only).
        """
        payload: dict = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        r = self._session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL OLLAMA MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class OllamaManager:
    """
    High-level manager for Ollama LLM conversations.

    Features
    --------
    - Model switching: llama3.2 ↔ gemma3:4b
    - Per-session conversation history (rolling window of MAX_HISTORY messages)
    - System-prompt injection per model
    - Validate model availability before calling
    - Stream or non-stream modes
    - Status + model-info endpoints for the frontend

    Example
    -------
    from ollama_manager import ollama_manager

    # Non-streaming
    reply = ollama_manager.chat("What is recursion?", model="llama3.2")

    # Streaming (yields delta strings)
    for chunk in ollama_manager.stream_chat("Tell me a story", model="gemma3"):
        print(chunk, end="", flush=True)
    """

    def __init__(self):
        self.client       = OllamaClient(OLLAMA_BASE_URL)
        self._history:  dict[str, list[dict]] = {}   # session_id → message list
        self._active_model: str = DEFAULT_MODEL       # server-side canonical name

    # ── internal helpers ──────────────────────────────────────────────────────

    def _resolve_model(self, model_key: str | None) -> str:
        """Resolve a friendly key (e.g. 'llama3.2') → canonical name (e.g. 'llama3.2:latest')."""
        if model_key is None:
            return self._active_model
        # Try exact match first, then friendly-key lookup
        if model_key in SUPPORTED_MODELS.values():
            return model_key
        return SUPPORTED_MODELS.get(model_key, model_key)

    def _build_messages(self, session_id: str, model: str, user_text: str) -> list[dict]:
        """Build the full message list: system prompt + history + new user turn."""
        system_prompt = SYSTEM_PROMPTS.get(model, SYSTEM_PROMPTS[DEFAULT_MODEL])
        history = self._history.get(session_id, [])

        # Rolling-window: keep only the last MAX_HISTORY messages
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
            self._history[session_id] = history

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _save_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Persist a completed turn in the session history."""
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append({"role": "user",      "content": user_text})
        self._history[session_id].append({"role": "assistant",  "content": assistant_text})

    # ── public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if Ollama is running and reachable."""
        return self.client.is_running()

    def status(self) -> dict:
        """
        Return a status dict for the /api/ollama/status endpoint.
        {
          "running": bool,
          "available_models": [...],
          "supported_models": {...},
          "active_model": "..."
        }
        """
        running = self.client.is_running()
        available: list[str] = []
        if running:
            try:
                available = self.client.model_names()
            except Exception:
                available = []

        return {
            "running":           running,
            "available_models":  available,
            "supported_models":  SUPPORTED_MODELS,
            "active_model":      self._active_model,
        }

    def set_default_model(self, model_key: str) -> bool:
        """
        Switch the default model.
        Returns True if the model is supported, False otherwise.
        """
        resolved = self._resolve_model(model_key)
        if resolved in SUPPORTED_MODELS.values():
            self._active_model = resolved
            return True
        return False

    def chat(
        self,
        user_text: str,
        model: str | None = None,
        session_id: str = "default",
        options: dict | None = None,
    ) -> str:
        """
        Send a message and return the assistant's reply as a string.
        Conversation history is maintained per session_id.

        Parameters
        ----------
        user_text  : the user's message
        model      : 'llama3.2' | 'gemma3' | full canonical name | None (use default)
        session_id : unique key for conversation history (use client IP in production)
        options    : Ollama model options, e.g. {"temperature": 0.7}

        Returns
        -------
        str — assistant reply, or a user-friendly error string on failure
        """
        if not self.is_available():
            return (
                "⚠️ Ollama is not running. Please start it with `ollama serve` "
                "and make sure your models are pulled."
            )

        resolved = self._resolve_model(model)

        # Validate the model is actually available
        try:
            available = self.client.model_names()
        except Exception:
            available = []

        if available and resolved not in available:
            return (
                f"❌ Model **{resolved}** is not available on your Ollama server. "
                f"Run `ollama pull {resolved}` to download it. "
                f"Available: {', '.join(available) or 'none'}"
            )

        messages = self._build_messages(session_id, resolved, user_text)

        try:
            response = self.client.chat(messages, model=resolved, stream=False, options=options)
            reply = response.get("message", {}).get("content", "").strip()
            if not reply:
                return "🤔 The model returned an empty response. Please try again."
            self._save_turn(session_id, user_text, reply)
            return reply

        except requests.exceptions.Timeout:
            return (
                "⏱️ The model took too long to respond. "
                "Try a shorter prompt or switch to a smaller model."
            )
        except requests.exceptions.ConnectionError:
            return "🌐 Lost connection to Ollama. Is it still running on port 11434?"
        except requests.exceptions.RequestException as exc:
            return f"❌ Ollama error: {exc}"

    def stream_chat(
        self,
        user_text: str,
        model: str | None = None,
        session_id: str = "default",
        options: dict | None = None,
    ) -> Iterator[str]:
        """
        Stream the assistant's reply token by token.
        Yields delta strings; saves the full reply to history when done.

        Usage (Flask SSE example):
            def generate():
                full = ""
                for chunk in ollama_manager.stream_chat(msg, model, session_id):
                    full += chunk
                    yield f"data: {json.dumps({'delta': chunk})}\\n\\n"
                yield f"data: {json.dumps({'done': True})}\\n\\n"
        """
        if not self.is_available():
            yield "⚠️ Ollama is not running. Please start it with `ollama serve`."
            return

        resolved = self._resolve_model(model)
        messages = self._build_messages(session_id, resolved, user_text)
        full_reply = ""

        try:
            for chunk in self.client.chat(messages, model=resolved, stream=True, options=options):
                full_reply += chunk
                yield chunk
            if full_reply:
                self._save_turn(session_id, user_text, full_reply)

        except requests.exceptions.Timeout:
            yield "\n\n⏱️ Response timed out."
        except requests.exceptions.ConnectionError:
            yield "\n\n🌐 Lost connection to Ollama."
        except requests.exceptions.RequestException as exc:
            yield f"\n\n❌ Error: {exc}"

    def clear_history(self, session_id: str = "default") -> None:
        """Wipe the conversation history for a session."""
        self._history.pop(session_id, None)

    def clear_all_history(self) -> None:
        """Wipe all sessions (admin use)."""
        self._history.clear()

    def history_length(self, session_id: str = "default") -> int:
        """Return the number of messages in a session's history."""
        return len(self._history.get(session_id, []))


# ── module-level singleton (imported by app.py) ───────────────────────────────
ollama_manager = OllamaManager()

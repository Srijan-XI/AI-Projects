import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import json
import pickle
import random
from pathlib import Path
import numpy as np
import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

from api_manager    import api_manager                    # central API + rate-limit manager
from ollama_manager import ollama_manager, SUPPORTED_MODELS  # local Ollama LLM manager
from database       import db                               # SQLite persistence layer

# ─────────────────────────────────────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
lemmatizer = WordNetLemmatizer()
app        = Flask(__name__)

model = load_model(BASE_DIR / 'chatbot_model.h5', compile=False)
with open(BASE_DIR / 'intents.json', encoding='utf-8') as _f:
    intents = json.load(_f)
with open(BASE_DIR / 'words.pkl', 'rb') as _f:
    words = pickle.load(_f)
with open(BASE_DIR / 'classes.pkl', 'rb') as _f:
    classes = pickle.load(_f)


# ─────────────────────────────────────────────────────────────────────────────
# NLP helpers
# ─────────────────────────────────────────────────────────────────────────────

_IGNORE_CHARS = set('?!.,')

def clean_up_sentence(sentence: str) -> list[str]:
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens if w not in _IGNORE_CHARS]


def bow(sentence: str, vocab: list) -> np.ndarray:
    token_set = set(clean_up_sentence(sentence))
    return np.array([1 if w in token_set else 0 for w in vocab])


def predict_class(sentence: str) -> list[dict]:
    p = bow(sentence, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = sorted(
        [[i, float(r)] for i, r in enumerate(res) if r > ERROR_THRESHOLD],
        key=lambda x: x[1], reverse=True
    )
    return [{"intent": classes[i], "probability": str(prob)} for i, prob in results]


def get_canned_response(ints: list[dict]) -> str:
    if not ints:
        return "🤔 I'm not sure I understood that. Could you rephrase?"
    tag = ints[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "🤔 I'm not sure I understood that. Could you rephrase?"


# ─────────────────────────────────────────────────────────────────────────────
# City extraction helper
# ─────────────────────────────────────────────────────────────────────────────

# Matches patterns like "weather in London", "forecast for New York", etc.
_CITY_RE = re.compile(
    r'(?:weather|temperature|forecast|raining|snowing|sunny|cloudy|humid)'
    r'.*?\b(?:in|at|for|of)\s+([A-Za-z][\w\s]{1,30}?)(?:\?|$|\.|!)',
    re.IGNORECASE
)
# Simpler fallback: "London weather", "weather London" (case-insensitive)
_CITY_SIMPLE_RE = re.compile(
    r'\b([A-Za-z][a-z]{2,}(?:\s[A-Za-z][a-z]+)?)\s+weather\b'
    r'|weather\s+\b([A-Za-z][a-z]{2,}(?:\s[A-Za-z][a-z]+)?)\b',
    re.IGNORECASE,
)


def extract_city(message: str) -> str | None:
    m = _CITY_RE.search(message)
    if m:
        return m.group(1).strip()
    m = _CITY_SIMPLE_RE.search(message)
    if m:
        return (m.group(1) or m.group(2)).strip()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    # 1. Identify client
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"

    # 2. Chat rate-limit check (30 msg / 60 s per IP)
    if not api_manager.chat_allowed(client_ip):
        wait = api_manager.chat_retry_after(client_ip)
        return jsonify({
            "response": f"⚠️ You're sending messages too fast! "
                        f"Please wait **{wait}s** before trying again.",
            "rate_limited": True
        }), 429

    # 3. Parse message
    msg = (request.form.get("msg") or "").strip()
    if not msg:
        return jsonify({"response": "Please type a message!"}), 400

    # 4. Log user message
    ints = predict_class(msg)
    top_intent = ints[0]['intent'] if ints else None
    db.log_message(client_ip, "user", msg, model="nlp", intent=top_intent)

    # 5. Joke intent → call live Jokes API
    if top_intent == "joke":
        reply = api_manager.get_joke(client_ip=client_ip)
        db.log_message(client_ip, "bot", reply, model="nlp", intent=top_intent)
        return jsonify({"response": reply})

    # 6. Weather intent → call live Weather API
    if top_intent == "weather":
        city = extract_city(msg)
        if city:
            weather_text = api_manager.get_weather(city, client_ip=client_ip)
            db.log_message(client_ip, "bot", weather_text, model="nlp", intent=top_intent)
            return jsonify({"response": weather_text})
        else:
            reply = (
                "🌍 Sure! Which city would you like the weather for?\n"
                "Try: _\"What's the weather in London?\"_"
            )
            db.log_message(client_ip, "bot", reply, model="nlp", intent=top_intent)
            return jsonify({"response": reply})

    # 7. All other intents → canned response
    reply = get_canned_response(ints)
    db.log_message(client_ip, "bot", reply, model="nlp", intent=top_intent)
    return jsonify({"response": reply})


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/ollama/chat", methods=["POST"])
def ollama_chat():
    """
    Non-streaming Ollama chat endpoint.
    Body (JSON or form): { msg, model, session_id? }
    Returns: { response, model, session_id }
    """
    client_ip  = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"

    # Chat rate limit (shared with NLP route)
    if not api_manager.chat_allowed(client_ip):
        wait = api_manager.chat_retry_after(client_ip)
        return jsonify({
            "response": f"⚠️ Too many messages! Please wait **{wait}s**.",
            "rate_limited": True
        }), 429

    data  = request.get_json(silent=True) or request.form
    msg   = (data.get("msg") or "").strip()
    model = (data.get("model") or "llama3.2").strip()
    sid   = (data.get("session_id") or client_ip)

    if not msg:
        return jsonify({"response": "Please type a message!"}), 400

    db.log_message(sid, "user", msg, model=model)
    reply = ollama_manager.chat(msg, model=model, session_id=sid)
    db.log_message(sid, "bot",  reply, model=model)
    return jsonify({"response": reply, "model": model, "session_id": sid})


@app.route("/api/ollama/stream", methods=["POST"])
def ollama_stream():
    """
    Server-Sent Events streaming endpoint.
    Body: { msg, model, session_id? }
    Client receives: text/event-stream with data: {"delta":"..."} lines
                     followed by data: {"done": true}
    """
    import json as _json
    from flask import Response, stream_with_context

    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"

    if not api_manager.chat_allowed(client_ip):
        wait = api_manager.chat_retry_after(client_ip)
        def err():
            yield f"data: {_json.dumps({'delta': f'⚠️ Rate limited. Retry in {wait}s.'})}\n\n"
            yield f"data: {_json.dumps({'done': True})}\n\n"
        return Response(stream_with_context(err()), mimetype="text/event-stream")

    data  = request.get_json(silent=True) or request.form
    msg   = (data.get("msg") or "").strip()
    model = (data.get("model") or "llama3.2").strip()
    sid   = (data.get("session_id") or client_ip)

    db.log_message(sid, "user", msg, model=model)

    def generate():
        full = ""
        for chunk in ollama_manager.stream_chat(msg, model=model, session_id=sid):
            full += chunk
            yield f"data: {_json.dumps({'delta': chunk})}\n\n"
        # Log the complete streamed reply once finished
        db.log_message(sid, "bot", full, model=model)
        yield f"data: {_json.dumps({'done': True, 'model': model})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/ollama/status", methods=["GET"])
def ollama_status():
    """Return Ollama server health and available models."""
    return jsonify(ollama_manager.status())


@app.route("/api/ollama/models", methods=["GET"])
def ollama_models():
    """Return just the list of supported model keys for the UI."""
    return jsonify({
        "supported": list(ollama_manager.client.model_names()) if ollama_manager.is_available() else [],
        "configured": {k: v for k, v in SUPPORTED_MODELS.items()},
    })


@app.route("/api/ollama/clear", methods=["POST"])
def ollama_clear():
    """Wipe in-memory Ollama history AND SQLite history for this session."""
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    data = request.get_json(silent=True) or request.form
    sid  = data.get("session_id") or client_ip
    ollama_manager.clear_history(sid)
    deleted = db.clear_history(sid)
    return jsonify({"cleared": True, "session_id": sid, "rows_deleted": deleted})


@app.route("/api/stats", methods=["GET"])
def chat_stats():
    """Return aggregate chat statistics from the SQLite database."""
    return jsonify(db.get_stats())


@app.route("/api/history", methods=["GET"])
def chat_history():
    """Return message history for a session (query param: ?session_id=...). """
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    sid   = request.args.get("session_id", client_ip)
    limit = min(int(request.args.get("limit", 50)), 200)
    return jsonify({"session_id": sid, "messages": db.get_history(sid, limit=limit)})


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug)

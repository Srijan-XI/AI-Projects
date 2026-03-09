/* ── State ────────────────────────────────────────────── */
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const clearBtn = document.getElementById("clear-btn");
const ollamaDot = document.getElementById("ollama-dot");
const statusText = document.getElementById("status-text");

// Current active mode: "nlp" | "llama3.2" | "gemma3"
let activeMode = "nlp";

/* ── Model Switcher ───────────────────────────────────── */

document.querySelectorAll(".model-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".model-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        activeMode = btn.dataset.mode;

        const labels = {
            "nlp": "⚡ NLP — Online & Ready",
            "llama3.2": "🦙 Llama 3.2 — via Ollama",
            "gemma3": "💎 Gemma 3 4B — via Ollama",
        };
        if (statusText) statusText.textContent = labels[activeMode] || "Ready";
    });
});

/* ── Ollama Status Polling ────────────────────────────── */

async function checkOllamaStatus() {
    try {
        const res = await fetch("/api/ollama/status");
        const data = await res.json();
        if (ollamaDot) {
            ollamaDot.className = "ollama-dot " + (data.running ? "online" : "offline");
            ollamaDot.title = data.running
                ? `Ollama online — ${data.available_models.join(", ")}`
                : "Ollama offline (run: ollama serve)";
        }
    } catch {
        if (ollamaDot) {
            ollamaDot.className = "ollama-dot offline";
            ollamaDot.title = "Cannot reach Ollama";
        }
    }
}

// Check on load and every 15 seconds
checkOllamaStatus();
setInterval(checkOllamaStatus, 15_000);

/* ── Helpers ─────────────────────────────────────────── */

function scrollToBottom() {
    chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: "smooth" });
}

function formatTime() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/* ── Append message bubble ───────────────────────────── */

function appendMessage(content, sender, modelLabel = null) {
    const row = document.createElement("div");
    row.className = sender === "user" ? "user-msg" : "bot-msg";

    // Avatar (bot side only)
    if (sender === "bot") {
        const avatarWrap = document.createElement("div");
        avatarWrap.className = "avatar-wrap";
        const avatar = document.createElement("span");
        avatar.className = "msg-avatar";
        avatar.textContent = modelLabel ? (modelLabel.includes("llama") ? "🦙" : "💎") : "🤖";
        avatarWrap.appendChild(avatar);
        row.appendChild(avatarWrap);
    }

    // Bubble
    const bubble = document.createElement("div");
    const isOllama = (sender === "bot" && modelLabel);
    bubble.className = sender === "user"
        ? "bubble"
        : isOllama
            ? "bubble ollama-bubble"
            : "bubble bot-bubble";

    // Model badge (Ollama only)
    if (isOllama) {
        const badge = document.createElement("span");
        badge.className = "model-badge";
        badge.textContent = modelLabel;
        bubble.appendChild(badge);
    }

    const text = document.createElement("p");
    text.style.margin = "0";
    text.textContent = content;
    bubble.appendChild(text);

    // Timestamp
    const ts = document.createElement("span");
    ts.textContent = formatTime();
    ts.style.cssText = `
        display:block; font-size:10.5px; margin-top:6px;
        opacity:0.45; text-align:${sender === "user" ? "right" : "left"}; color:inherit;
    `;
    bubble.appendChild(ts);

    row.appendChild(bubble);
    chatBox.appendChild(row);
    scrollToBottom();
    return text;   // return <p> so streaming can update it
}

/* ── Typing indicator ────────────────────────────────── */

function showTyping() {
    const el = document.getElementById("typing-indicator");
    if (el) {
        el.style.display = "flex";
        el.removeAttribute("aria-hidden");
        chatBox.appendChild(el);
        scrollToBottom();
    }
}

function hideTyping() {
    const el = document.getElementById("typing-indicator");
    if (el) {
        el.style.display = "none";
        el.setAttribute("aria-hidden", "true");
    }
}

/* ── Loading state ───────────────────────────────────── */

function setLoading(on) {
    sendButton.disabled = on;
    userInput.disabled = on;
}

/* ── NLP send (original /get endpoint) ──────────────── */

function sendNLP(text) {
    setLoading(true);
    showTyping();

    fetch("/get", {
        method: "POST",
        body: new URLSearchParams({ msg: text }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
    })
        .then(res => res.json())
        .then(data => {
            hideTyping();
            appendMessage(data.response, "bot");
        })
        .catch(() => {
            hideTyping();
            appendMessage("⚠️ Something went wrong. Please try again.", "bot");
        })
        .finally(() => setLoading(false));
}

/* ── Ollama stream send (/api/ollama/stream SSE) ─────── */

async function sendOllamaStream(text, model) {
    setLoading(true);
    showTyping();

    const modelNames = { "llama3.2": "Llama 3.2", "gemma3": "Gemma 3 4B" };
    const labelText = modelNames[model] || model;

    try {
        const res = await fetch("/api/ollama/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ msg: text, model }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            hideTyping();
            appendMessage(err.response || "❌ Ollama request failed.", "bot", labelText);
            return;
        }

        hideTyping();
        // Create bubble upfront for streaming into
        const textNode = appendMessage("", "bot", labelText);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();   // keep incomplete last line

            for (const line of lines) {
                if (!line.startsWith("data:")) continue;
                const raw = line.slice(5).trim();
                if (!raw) continue;
                try {
                    const chunk = JSON.parse(raw);
                    if (chunk.delta) {
                        textNode.textContent += chunk.delta;
                        scrollToBottom();
                    }
                } catch { /* skip malformed */ }
            }
        }
    } catch (err) {
        hideTyping();
        appendMessage(`🌐 Could not reach Ollama: ${err.message}`, "bot", labelText);
    } finally {
        setLoading(false);
    }
}

/* ── Master send dispatcher ──────────────────────────── */

function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage(text, "user");
    userInput.value = "";
    userInput.focus();

    if (activeMode === "nlp") {
        sendNLP(text);
    } else {
        sendOllamaStream(text, activeMode);
    }
}

/* ── Quick-chip click handler ────────────────────────── */

chatBox.addEventListener("click", (e) => {
    if (e.target.classList.contains("chip")) {
        userInput.value = e.target.dataset.msg || e.target.textContent;
        sendMessage();
    }
});

/* ── Clear chat ──────────────────────────────────────── */

clearBtn.addEventListener("click", () => {
    // Remove all messages except the welcome card
    const welcomeMsg = document.getElementById("welcome-msg");
    chatBox.querySelectorAll(".user-msg, .bot-msg").forEach(m => {
        if (m !== welcomeMsg && !m.id) m.remove();
    });

    // Also clear Ollama server-side history if in Ollama mode
    if (activeMode !== "nlp") {
        fetch("/api/ollama/clear", {
            method: "POST",
            headers: { "Content-Type": "application/json" }, body: "{}"
        });
    }
});

/* ── Key & click listeners ───────────────────────────── */

userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendButton.addEventListener("click", sendMessage);
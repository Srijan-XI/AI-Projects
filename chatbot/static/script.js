const chatBox    = document.getElementById("chat-box");
const userInput  = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const clearBtn   = document.getElementById("clear-btn");

/* ── Helpers ─────────────────────────────────────────── */

function scrollToBottom() {
    chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: "smooth" });
}

function formatTime() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/* ── Append message bubble ───────────────────────────── */

function appendMessage(content, sender) {
    const row = document.createElement("div");
    row.className = sender === "user" ? "user-msg" : "bot-msg";

    // Avatar (bot side only)
    if (sender === "bot") {
        const avatarWrap = document.createElement("div");
        avatarWrap.className = "avatar-wrap";
        const avatar = document.createElement("span");
        avatar.className = "msg-avatar";
        avatar.textContent = "🤖";
        avatarWrap.appendChild(avatar);
        row.appendChild(avatarWrap);
    }

    // Bubble
    const bubble = document.createElement("div");
    bubble.className = sender === "user" ? "bubble" : "bubble bot-bubble";

    const text = document.createElement("p");
    text.textContent = content;
    bubble.appendChild(text);

    // Timestamp
    const ts = document.createElement("span");
    ts.className = "msg-time";
    ts.textContent = formatTime();
    ts.style.cssText = `
        display: block;
        font-size: 10.5px;
        margin-top: 6px;
        opacity: 0.45;
        text-align: ${sender === "user" ? "right" : "left"};
        color: inherit;
    `;
    bubble.appendChild(ts);

    row.appendChild(bubble);
    chatBox.appendChild(row);
    scrollToBottom();
}

/* ── Typing indicator ────────────────────────────────── */

function showTyping() {
    const el = document.getElementById("typing-indicator");
    if (el) {
        el.style.display = "flex";
        el.removeAttribute("aria-hidden");
        chatBox.appendChild(el);   // move to bottom
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

/* ── Send message ────────────────────────────────────── */

function setLoading(on) {
    sendButton.disabled = on;
    userInput.disabled  = on;
}

function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    appendMessage(text, "user");
    userInput.value = "";
    userInput.focus();
    setLoading(true);
    showTyping();

    fetch("/get", {
        method: "POST",
        body: new URLSearchParams({ msg: text }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" }
    })
    .then(res => res.json())
    .then(data => {
        hideTyping();
        appendMessage(data.response, "bot");
    })
    .catch(() => {
        hideTyping();
        appendMessage("⚠️ Sorry, something went wrong. Please try again.", "bot");
    })
    .finally(() => setLoading(false));
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
    // Keep only the welcome message
    const msgs = chatBox.querySelectorAll(".user-msg, .bot-msg");
    const welcomeMsg = document.getElementById("welcome-msg");
    msgs.forEach(m => { if (m !== welcomeMsg && !m.id) m.remove(); });
});

/* ── Event listeners ─────────────────────────────────── */

userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendButton.addEventListener("click", sendMessage);
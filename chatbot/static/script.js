const chatBox = document.getElementById("chat-box");
        const userInput = document.getElementById("user-input");
        const sendButton = document.getElementById("send-button");

        function appendMessage(content, sender) {
            const messageDiv = document.createElement("div");
            messageDiv.className = sender === "user" ? "user-msg" : "bot-msg";
            const bubbleSpan = document.createElement("span");
            bubbleSpan.className = "message bubble";
            bubbleSpan.textContent = content;
            messageDiv.appendChild(bubbleSpan);
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function sendMessage() {
            const userText = userInput.value.trim();
            if (!userText) return;

            appendMessage(userText, "user");
            userInput.value = "";
            userInput.focus();

            fetch("/get", {
                method: "POST",
                body: new URLSearchParams({ "msg": userText }),
                headers: { "Content-Type": "application/x-www-form-urlencoded" }
            })
            .then(res => res.json())
            .then(data => {
                appendMessage(data.response, "bot");
            })
            .catch(() => {
                appendMessage("Sorry, something went wrong. Please try again.", "bot");
            });
        }

        userInput.addEventListener("keydown", function(e) {
            if (e.key === "Enter" && this.value.trim()) {
                sendMessage();
                e.preventDefault();
            }
        });

        sendButton.addEventListener("click", sendMessage);
    
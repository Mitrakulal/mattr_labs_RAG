const chatBox = document.getElementById("chatBox");
const input = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");

sendBtn.addEventListener("click", sendQuestion);
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuestion();
});

async function sendQuestion() {
    const question = input.value.trim();
    if (!question) return;

    input.value = "";
    addMessage("user", question);
    addMessage("loading", "Thinking...");
    sendBtn.disabled = true;

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");

        let isFirstChunk = true;
        let botMsg = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            if (!chunk) continue;

            if (isFirstChunk) {
                // Remove "Thinking..." message ONLY when the very first token arrives
                document.querySelector(".message.loading")?.remove();

                botMsg = document.createElement("div");
                botMsg.className = "message bot";
                botMsg.textContent = "";
                chatBox.appendChild(botMsg);

                isFirstChunk = false;
            }

            botMsg.textContent += chunk;
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        if (isFirstChunk) {
            document.querySelector(".message.loading")?.remove();
            addMessage("bot", "No response received.");
        }
    } catch (err) {
        document.querySelector(".message.loading")?.remove();
        addMessage("bot", "Sorry, something went wrong. Please try again later.");
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}


function addMessage(role, text) {
    const msg = document.createElement("div");
    msg.className = `message ${role}`;
    msg.textContent = text;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}

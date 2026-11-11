// US-05 (Ayan): chat UI behaviour. Sends a natural-language question to /api/chat and shows the answer.
(function () {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  const clearBtn = document.getElementById("chat-clear");
  const log = document.getElementById("chat-log");
  const empty = document.getElementById("chat-empty");
  const count = document.getElementById("chat-count");
  const csrf = form.querySelector("input[name=csrfmiddlewaretoken]").value;
  const maxLen = parseInt(input.getAttribute("maxlength"), 10) || 1000;
  let busy = false;

  function scrollToBottom() {
    log.scrollTop = log.scrollHeight;
  }

  function addMessage(role, text) {
    if (empty) empty.hidden = true;
    const el = document.createElement("div");
    el.className = "msg msg-" + role;
    const body = document.createElement("div");
    body.className = "msg-text";
    body.textContent = text; // textContent, never innerHTML: no injected HTML from answers.
    el.appendChild(body);
    log.appendChild(el);
    scrollToBottom();
    return el;
  }

  function addThinking() {
    const el = addMessage("assistant", "Searching protocols");
    el.classList.add("msg-thinking");
    return el;
  }

  function updateCount() {
    count.textContent = input.value.length + " / " + maxLen;
  }

  function setBusy(value) {
    busy = value;
    sendBtn.disabled = value;
    input.disabled = value;
    sendBtn.textContent = value ? "Sending…" : "Send";
  }

  async function ask(question) {
    addMessage("user", question);
    input.value = "";
    updateCount();
    setBusy(true);
    const thinking = addThinking();
    try {
      const resp = await fetch(form.dataset.apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": csrf },
        body: JSON.stringify({ question: question }),
      });
      let data = {};
      try {
        data = await resp.json();
      } catch (e) {
        data = {};
      }
      thinking.remove();
      if (!resp.ok) {
        addMessage("error", data.error || "Request failed (" + resp.status + ").");
        return;
      }
      const msg = addMessage("assistant", data.text || "(No answer returned.)");
      if (data.latency_ms) {
        const meta = document.createElement("div");
        meta.className = "msg-meta";
        meta.textContent = "Answered in " + (data.latency_ms / 1000).toFixed(1) + "s";
        msg.appendChild(meta);
      }
      // US-07 (Zane) renders the sources; see sources.js.
      if (window.AlertAssistSources) {
        msg.appendChild(window.AlertAssistSources.render(data.sources || []));
      }
      scrollToBottom();
    } catch (err) {
      thinking.remove();
      addMessage("error", "Couldn't reach the server. Check your connection and try again.");
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (busy) return;
    const question = input.value.trim();
    if (!question) return;
    ask(question);
  });

  // Enter sends, Shift+Enter adds a new line.
  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener("input", updateCount);

  document.querySelectorAll(".chat-example").forEach(function (btn) {
    btn.addEventListener("click", function () {
      if (!busy) ask(btn.textContent.trim());
    });
  });

  clearBtn.addEventListener("click", function () {
    log.querySelectorAll(".msg").forEach(function (m) { m.remove(); });
    if (empty) empty.hidden = false;
    input.focus();
  });

  updateCount();
  input.focus();
})();

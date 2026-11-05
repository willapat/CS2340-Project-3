// US-05 (Ayan): chat UI behaviour. Walking skeleton: sends question, shows placeholder answer.
(function () {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const log = document.getElementById("chat-log");
  const csrf = form.querySelector("input[name=csrfmiddlewaretoken]").value;

  function addMessage(role, text) {
    const el = document.createElement("div");
    el.className = "msg msg-" + role;
    el.textContent = text;
    log.appendChild(el);
    return el;
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    addMessage("user", question);
    input.value = "";
    try {
      const resp = await fetch(form.dataset.apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": csrf },
        body: JSON.stringify({ question: question }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        addMessage("error", data.error || "Request failed.");
        return;
      }
      const msg = addMessage("assistant", data.text);
      // US-07 (Zane) renders the sources; see sources.js.
      if (window.AlertAssistSources) {
        msg.appendChild(window.AlertAssistSources.render(data.sources || []));
      }
    } catch (err) {
      addMessage("error", "Network error: " + err);
    }
  });
})();

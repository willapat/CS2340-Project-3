// US-07 (Zane): render an answer's sources. Minimal skeleton so the fake source shows up.
window.AlertAssistSources = {
  render: function (sources) {
    const list = document.createElement("ul");
    list.className = "sources";
    sources.forEach(function (s) {
      const li = document.createElement("li");
      li.textContent = s.source || "Unknown source";
      list.appendChild(li);
    });
    return list;
  },
};

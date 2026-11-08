const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) if (c != null) e.append(c);
  return e;
}

function ago(ts) {
  if (!ts) return "never";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return "just now";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 172800) return Math.round(s / 3600) + " h ago";
  return Math.round(s / 86400) + " days ago";
}

function factList(facts, showWho = false) {
  if (!facts.length) return el("p", { class: "empty" }, "nothing yet");
  return el("ul", { class: "facts" }, ...facts.map((f) =>
    el("li", {}, f.text, showWho && f.person ? el("span", { class: "who" }, ` (${f.person})`) : null)));
}

function render(m) {
  $("review").textContent = m.review_running ? "review running…" : `last review: ${ago(m.last_review)}`;
  $("run").disabled = m.review_running;

  const people = $("people");
  people.replaceChildren(...(m.people.length ? m.people.map((p) => el("div", { class: "card", id: "person-" + p.id },
    el("h3", {}, p.name),
    el("div", { class: "sub" }, `last seen ${ago(p.last_seen)} · ${p.face_samples} face samples`),
    p.profile ? el("p", { class: "profile" }, p.profile) : null,
    el("div", { class: "label" }, "Topics"),
    p.topics.length
      ? el("div", { class: "chips" }, ...p.topics.map((t) => el("a", { class: "chip", href: "#topic-" + t.id }, t.name,
          el("span", { class: "when" }, t.count ? `talked ${ago(t.last_ts)}` : "linked"))))
      : el("p", { class: "empty" }, "none yet"),
    el("div", { class: "label" }, "Facts"),
    factList(p.facts),
  )) : [el("p", { class: "empty" }, "nobody yet")]));

  const topics = $("topics");
  topics.replaceChildren(...(m.topics.length ? m.topics.map((t) => el("div", { class: "card", id: "topic-" + t.id },
    el("h3", {}, t.name),
    t.summary ? el("p", { class: "summary" }, t.summary) : null,
    t.people.length ? el("div", { class: "chips" }, ...t.people.map((p) =>
      el("a", { class: "chip person", href: "#person-" + p.id }, p.name,
        el("span", { class: "when" }, p.count ? `${p.count}× · ${ago(p.last_ts)}` : "linked")))) : null,
    t.related.length ? el("div", {}, el("div", { class: "label" }, "Related"),
      el("div", { class: "chips" }, ...t.related.map((r) => el("a", { class: "chip", href: "#topic-" + r.id }, r.name)))) : null,
    el("div", { class: "label" }, "Facts"),
    factList(t.facts, true),
  )) : [el("p", { class: "empty" }, "no topics yet (they are built by the memory review)")]));

  $("household").replaceChildren(...(m.household.length ? m.household.map((f) => el("li", {}, f.text))
    : [el("li", { class: "empty" }, "nothing yet")]));
  $("questions").replaceChildren(...(m.questions.length ? m.questions.map((q) => el("li", {}, q.text))
    : [el("li", { class: "empty" }, "nothing right now")]));
}

async function load() {
  try {
    render(await (await fetch("/api/memory")).json());
  } catch (err) {
    $("review").textContent = "could not load memory: " + err;
  }
}

$("run").addEventListener("click", async () => {
  $("run").disabled = true;
  $("review").textContent = "review running… (this can take a few minutes)";
  try {
    await fetch("/api/consolidate", { method: "POST" });
  } finally {
    load();
  }
});

load();
setInterval(load, 30000);

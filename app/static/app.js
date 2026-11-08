const $ = (id) => document.getElementById(id);
const log = $("log"), text = $("text"), stateEl = $("state");
const camToggle = $("camera"), speakToggle = $("speak"), quietToggle = $("quiet"), internetToggle = $("internet");

// Remember per-browser toggles (speak defaults to off). Quiet is a server setting.
for (const el of [camToggle, speakToggle]) {
  try {
    const saved = localStorage.getItem("laptopai." + el.id);
    if (saved !== null) el.checked = saved === "1";
  } catch {}
  el.addEventListener("change", () => {
    try { localStorage.setItem("laptopai." + el.id, el.checked ? "1" : "0"); } catch {}
  });
}

const KIND_LABELS = {
  system_start: "start", system_stop: "shutdown", unclean_shutdown: "crash",
  scene: "sees", person_arrived: "arrived", person_left: "left", person_named: "name",
  curious: "wondering", learned: "learned", consolidation: "memory review", lookup: "looked up",
  initiative: "said", initiative_quiet: "would say", decision: "decided", memory: "memory", setting: "setting", reminder: "reminder", screen: "screen",
};

const OWN_LABELS = { initiative: "on its own", reminder: "reminder" };

function setState(s) {
  stateEl.textContent = s;
  stateEl.className = "pill " + s;
}

// Today: 10:43:06. Earlier: 06.11. 22:15
function fmtTime(ts) {
  const d = new Date(ts * 1000);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  if (d.toDateString() === new Date().toDateString()) return time;
  const date = d.toLocaleDateString([], { day: "2-digit", month: "2-digit" });
  return `${date} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

function stamp(ts) {
  const t = document.createElement("time");
  t.className = "stamp";
  t.textContent = fmtTime(ts);
  t.dateTime = new Date(ts * 1000).toISOString();
  t.title = new Date(ts * 1000).toLocaleString();
  return t;
}

function since(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return "just now";
  if (s < 3600) return Math.floor(s / 60) + " min";
  return Math.floor(s / 3600) + " h " + Math.floor((s % 3600) / 60) + " min";
}

// ---------------------------------------------------------------- chat log
function addMsg(cls, content, label, ts) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  if (label) {
    const l = document.createElement("span");
    l.className = "label";
    l.textContent = label;
    div.appendChild(l);
  }
  const body = document.createElement("span");
  body.className = "text";
  body.textContent = content;
  div.appendChild(body);
  if (ts) div.appendChild(stamp(ts));
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function sourceLabel(source, sender) {
  if (source === "voice") return "🎙 spoken";
  if (source === "developer") return "🛠 " + (sender || "developer tool");
  return "⌨ typed";
}

function addUserMsg(text, source, sender, ts) {
  return addMsg(source === "developer" ? "user dev" : "user", text, sourceLabel(source, sender), ts);
}

function renderMessage(m) {
  if (m.role === "user") addUserMsg(m.content, m.source, m.sender, m.ts);
  else if (OWN_LABELS[m.source]) addMsg("own", m.content, OWN_LABELS[m.source], m.ts);
  else addMsg("bot", m.content, null, m.ts);
}

function fmtTimings(t) {
  const parts = [];
  if (t.stt_ms != null) parts.push(`stt ${t.stt_ms} ms`);
  if (t.queue_ms) parts.push(`waited ${t.queue_ms} ms`);
  if (t.ttft_ms != null) parts.push(`first token ${t.ttft_ms} ms`);
  if (t.first_audio_ms != null) parts.push(`first audio ${t.first_audio_ms} ms`);
  if (t.tok_s != null) parts.push(`${t.tok_s} tok/s`);
  if (t.prompt_tokens != null) parts.push(`${t.prompt_tokens} prompt tok`);
  if (t.total_ms != null) parts.push(`total ${t.total_ms} ms`);
  return parts.join(" · ");
}

// ---------------------------------------------------------------- timeline
const timeline = $("timeline");
function addEvent(e) {
  const li = document.createElement("li");
  li.className = e.kind;
  const time = stamp(e.ts);
  const body = document.createElement("div");
  body.className = "body";
  const kind = document.createElement("span");
  kind.className = "kind";
  kind.textContent = KIND_LABELS[e.kind] || e.kind;
  body.append(kind, document.createTextNode(e.text));
  const snap = e.data && e.data.snapshot;
  if (snap) {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = "/snapshots/" + snap;
    img.alt = "Snapshot " + new Date(e.ts * 1000).toLocaleString();
    img.addEventListener("click", () => openViewer(img.src, `${new Date(e.ts * 1000).toLocaleString()}: ${e.text}`));
    body.appendChild(img);
  }
  li.append(time, body);
  timeline.prepend(li);  // newest first
  while (timeline.children.length > 200) timeline.lastChild.remove();
}

function openViewer(src, caption) {
  const d = $("viewer");
  d.querySelector("img").src = src;
  d.querySelector("p").textContent = caption;
  d.showModal();
}
$("viewer").querySelector("button").addEventListener("click", () => $("viewer").close());
$("viewer").addEventListener("click", (e) => { if (e.target === $("viewer")) $("viewer").close(); });

// ---------------------------------------------------------------- people in view
let people = [];
function renderPeople() {
  const ul = $("people");
  // Don't rebuild while someone is typing a name.
  if (ul.contains(document.activeElement) && document.activeElement.tagName === "INPUT") return;
  ul.innerHTML = "";
  if (!people.length) {
    ul.innerHTML = '<li class="empty">nobody recognized</li>';
    return;
  }
  for (const p of people) {
    const li = document.createElement("li");
    li.className = p.known ? "known" : "unknown";
    const dot = document.createElement("span");
    dot.className = "dot";
    li.appendChild(dot);
    if (p.known) {
      li.appendChild(document.createTextNode(p.name));
    } else {
      const form = document.createElement("form");
      const input = document.createElement("input");
      input.placeholder = "unknown: who is this?";
      input.setAttribute("aria-label", "Name for the unknown person");
      const btn = document.createElement("button");
      btn.textContent = "Save";
      form.append(input, btn);
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        const name = input.value.trim();
        if (name) ws.send(JSON.stringify({ type: "name_face", key: p.key, name }));
        input.blur();
      });
      li.appendChild(form);
    }
    const s = document.createElement("span");
    s.className = "since";
    s.textContent = since(p.since);
    li.appendChild(s);
    ul.appendChild(li);
  }
}
setInterval(renderPeople, 30000);

// ---------------------------------------------------------------- open questions
function renderQuestions(qs) {
  const ul = $("questions");
  ul.innerHTML = "";
  if (!qs.length) {
    ul.innerHTML = '<li class="empty">nothing yet</li>';
    return;
  }
  for (const q of qs) {
    const li = document.createElement("li");
    li.textContent = q.text;
    if (q.asked) {
      const s = document.createElement("span");
      s.className = "asked";
      s.textContent = q.asked === 1 ? "asked once" : `asked ${q.asked}×`;
      li.appendChild(s);
    }
    ul.appendChild(li);
  }
}

// ---------------------------------------------------------------- audio out
const audioQueue = [];
let playing = false;
function playNext() {
  if (playing || !audioQueue.length) return;
  playing = true;
  const audio = new Audio(audioQueue.shift());
  const done = () => { URL.revokeObjectURL(audio.src); playing = false; playNext(); };
  audio.onended = done;
  audio.onerror = done;
  audio.play().catch(done);
}
function enqueueWav(b64) {
  if (!speakToggle.checked) return;
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  audioQueue.push(URL.createObjectURL(new Blob([bytes], { type: "audio/wav" })));
  playNext();
}

// ---------------------------------------------------------------- websocket
let ws, botDiv = null, busy = false;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => setState("idle");
  ws.onclose = () => { setState("offline"); setTimeout(connect, 2000); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    switch (m.type) {
      case "hello":
        log.innerHTML = "";
        timeline.innerHTML = "";
        m.messages.forEach(renderMessage);
        m.events.forEach(addEvent);
        people = m.people;
        renderPeople();
        quietToggle.checked = m.quiet;
        internetToggle.checked = !!m.internet;
        renderQuestions(m.questions || []);
        $("model").textContent = m.model;
        break;
      case "state":
        setState(m.state);
        busy = m.state !== "idle";
        break;
      case "user":
        addUserMsg(m.text, m.source, m.sender, m.ts);
        break;
      case "assistant_start":
        botDiv = addMsg("bot", "", null, m.ts);
        break;
      case "token":
        if (!botDiv) botDiv = addMsg("bot", "", null, Date.now() / 1000);
        botDiv.querySelector(".text").textContent += m.text;
        log.scrollTop = log.scrollHeight;
        break;
      case "assistant":
        addMsg(OWN_LABELS[m.source] ? "own" : "bot", m.text, OWN_LABELS[m.source] || null, m.ts);
        break;
      case "audio":
        enqueueWav(m.wav);
        break;
      case "done":
        if (botDiv) {
          const meta = document.createElement("span");
          meta.className = "meta";
          meta.textContent = fmtTimings(m.timings);
          botDiv.appendChild(meta);
        }
        botDiv = null;
        break;
      case "event":
        addEvent(m.event);
        break;
      case "presence":
        people = m.people;
        renderPeople();
        break;
      case "questions":
        renderQuestions(m.questions);
        break;
      case "settings":
        if ("quiet" in m) quietToggle.checked = m.quiet;
        if ("internet" in m) internetToggle.checked = m.internet;
        break;
      case "notice":
        addMsg("notice", m.text);
        break;
      case "error":
        addMsg("error", m.text);
        botDiv = null;
        break;
    }
  };
}

function send(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) { addMsg("error", "not connected"); return false; }
  ws.send(JSON.stringify(obj));
  return true;
}

function sendTurn(msg, source, sttMs = null) {
  msg = msg.trim();
  if (!msg) return;
  send({ type: "turn", text: msg, source, camera: camToggle.checked, speak: speakToggle.checked, stt_ms: sttMs });
}

quietToggle.addEventListener("change", () => send({ type: "set_quiet", quiet: quietToggle.checked }));
internetToggle.addEventListener("change", () => send({ type: "set_internet", internet: internetToggle.checked }));

// ---------------------------------------------------------------- text input
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  sendTurn(text.value, "text");
  text.value = "";
  autosize();
});
text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
});
function autosize() { text.style.height = "auto"; text.style.height = text.scrollHeight + "px"; }
text.addEventListener("input", autosize);

// ---------------------------------------------------------------- push-to-talk
let recorder = null, chunks = [];
async function startRec() {
  if (recorder || busy) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      recorder = null;
      $("mic").classList.remove("rec");
      const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
      if (blob.size < 2000) { setState("idle"); return; }  // accidental tap
      setState("transcribing");
      const form = new FormData();
      form.append("audio", blob, "speech.webm");
      try {
        const r = await (await fetch("/api/transcribe", { method: "POST", body: form })).json();
        if (r.text) sendTurn(r.text, "voice", r.ms);
        else { addMsg("notice", "didn't catch that"); setState("idle"); }
      } catch (err) {
        addMsg("error", "transcription failed: " + err); setState("idle");
      }
    };
    recorder.start();
    $("mic").classList.add("rec");
    setState("listening");
  } catch (err) {
    addMsg("error", "microphone unavailable: " + err.message);
  }
}
function stopRec() { if (recorder && recorder.state === "recording") recorder.stop(); }

const mic = $("mic");
mic.addEventListener("pointerdown", (e) => { e.preventDefault(); startRec(); });
mic.addEventListener("pointerup", stopRec);
mic.addEventListener("pointerleave", stopRec);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && document.activeElement.tagName !== "TEXTAREA"
      && document.activeElement.tagName !== "INPUT" && !e.repeat) { e.preventDefault(); startRec(); }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && document.activeElement.tagName !== "TEXTAREA"
      && document.activeElement.tagName !== "INPUT") stopRec();
});

// ---------------------------------------------------------------- camera preview
const camImg = $("cam");
function refreshCam() {
  if (document.hidden) return;
  const next = new Image();
  next.onload = () => { camImg.src = next.src; camImg.parentElement.classList.remove("off"); };
  next.onerror = () => camImg.parentElement.classList.add("off");
  next.src = "/camera.jpg?t=" + Date.now();
}
setInterval(refreshCam, 1000);
refreshCam();

connect();

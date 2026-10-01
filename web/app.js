const logEl = document.querySelector("#log");
const draftEl = document.querySelector("#draft");
const tasksEl = document.querySelector("#tasks");
const connEl = document.querySelector("#conn");
const floorEl = document.querySelector("#floor");
const thinkEl = document.querySelector("#think");
const micBtn = document.querySelector("#mic");
const cutBtn = document.querySelector("#cut");
const form = document.querySelector("#composer");
const textInput = document.querySelector("#text");
const levelEl = document.querySelector("#level");

const floorName = { idle: "空闲", user: "你在说", assistant: "我在说" };
const tasks = new Map();

let socket = null;
let frameIndex = 0;
let micOn = false;
let audioCtx = null;
let media = null;
let workletNode = null;
let playNode = null;
let playback = false;
let playbackReady = null;
let captureReady = false;
let reconnect = 500;
let sessionId = sessionStorage.getItem("duplex_session") || "";
let acceptPcm = false;
let announced = false;
let micSource = null;

function setStatus(el, text) {
  el.textContent = text;
}

function bubble(role, text) {
  const node = document.createElement("div");
  node.className = `bubble ${role}`;
  node.textContent = text;
  logEl.appendChild(node);
  logEl.scrollTop = logEl.scrollHeight;
}

function upsertTask(id, patch) {
  const key = id || "current";
  const prev = tasks.get(key) || { id: key, status: "running", lines: [] };
  const next = { ...prev, ...patch };
  if (patch.line) next.lines = prev.lines.concat(patch.line);
  tasks.set(key, next);
  renderTasks();
}

function renderTasks() {
  tasksEl.replaceChildren();
  for (const task of tasks.values()) {
    const item = document.createElement("li");
    item.className = `task ${task.status || ""}`;
    const title = document.createElement("strong");
    title.textContent = task.goal || task.id;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = [task.status, task.tool].filter(Boolean).join(" · ");
    item.append(title, meta);
    if (task.lines && task.lines.length) {
      const body = document.createElement("div");
      body.textContent = task.lines.slice(-4).join("\n");
      item.append(body);
    }
    tasksEl.append(item);
  }
}

function reportPlayback(state) {
  sendJson({ type: "playback", state });
}

function ensureAudio() {
  if (playbackReady) return playbackReady;
  playbackReady = (async () => {
    audioCtx = new AudioContext();
    await audioCtx.audioWorklet.addModule("/audio-playback.worklet.js");
    playNode = new AudioWorkletNode(audioCtx, "duplex-playback");
    playNode.connect(audioCtx.destination);
    playNode.port.onmessage = (ev) => {
      if (ev.data.type !== "state") return;
      playback = Boolean(ev.data.playing);
      reportPlayback(playback ? "start" : "end");
    };
  })();
  return playbackReady;
}

async function enqueuePcm(int16) {
  await ensureAudio();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  const samples = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i += 1) samples[i] = int16[i] / 32768;
  playNode.port.postMessage({ cmd: "pcm", samples }, [samples.buffer]);
}

function cutSpeech() {
  acceptPcm = false;
  const wasPlaying = playback;
  playback = false;
  if (playNode) playNode.port.postMessage({ cmd: "cut" });
  if (wasPlaying) reportPlayback("end");
}

function handleBinary(buffer) {
  if (!acceptPcm) return;
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 14 || bytes[0] !== 0x50 || bytes[1] !== 0x43 || bytes[2] !== 0x4d || bytes[3] !== 0x31) {
    return;
  }
  const view = new DataView(buffer);
  const payload = bytes.length - 12;
  if (payload < 2 || payload % 2 !== 0) return;
  const pcm = new Int16Array(payload / 2);
  for (let i = 0; i < pcm.length; i += 1) pcm[i] = view.getInt16(12 + i * 2, true);
  enqueuePcm(pcm).catch((err) => bubble("system", `播放失败：${err.message || err}`));
}

function encodeFrame(index, pcm, hint) {
  const bytes = new Uint8Array(12 + pcm.byteLength);
  const view = new DataView(bytes.buffer);
  bytes[0] = 0x50;
  bytes[1] = 0x43;
  bytes[2] = 0x4d;
  bytes[3] = 0x31;
  bytes[4] = 1;
  bytes[5] = (hint ? 1 : 0) | (playback ? 2 : 0);
  view.setUint32(8, index >>> 0, true);
  bytes.set(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength), 12);
  return bytes;
}

function handleMessage(raw) {
  let msg;
  try {
    msg = JSON.parse(raw);
  } catch {
    return;
  }
  if (msg.type === "hello") {
    setStatus(connEl, `已连接 ${msg.session_id}`);
    if (msg.session_id && sessionId && msg.session_id !== sessionId) tasks.clear();
    sessionId = msg.session_id || sessionId;
    if (sessionId) sessionStorage.setItem("duplex_session", sessionId);
    const engines = [msg.asr, msg.tts].filter(Boolean).join(" / ");
    if (!announced) {
      announced = true;
      bubble(
        "system",
        `${msg.llm ? "思考层模型已配置" : "思考层使用本地工具"}${engines ? ` · 语音 ${engines}` : ""}`,
      );
    }
    return;
  }
  if (msg.type === "transcript" && msg.role === "user") {
    if (msg.final === false) {
      draftEl.textContent = msg.text;
      return;
    }
    draftEl.textContent = "";
    bubble("user", msg.text);
    return;
  }
  if (msg.type === "floor") {
    setStatus(floorEl, floorName[msg.floor] || msg.floor || "空闲");
    return;
  }
  if (msg.type === "speak") {
    acceptPcm = true;
    bubble("assistant", msg.text);
    ensureAudio().catch(() => {});
    return;
  }
  if (msg.type === "cut") {
    cutSpeech();
    bubble("system", "已打断");
    return;
  }
  if (msg.type === "delegation") {
    setStatus(thinkEl, "思考中");
    upsertTask(msg.task_id, { status: "running", goal: msg.goal || msg.task_id, lines: [] });
    return;
  }
  if (msg.type === "fragment") {
    upsertTask(msg.task_id, { line: msg.text, status: "running" });
    return;
  }
  if (msg.type === "task") {
    const status = msg.status || "";
    if (status === "cancelled" || status === "done" || status === "failed") {
      setStatus(thinkEl, status === "done" ? "思考完成" : "思考已停");
    }
    const key = msg.task_id || "current";
    const prev = tasks.get(key);
    const extra = msg.summary || msg.detail || "";
    const seen = prev && prev.lines && prev.lines.includes(extra);
    upsertTask(key, {
      status,
      ...(msg.tool ? { tool: msg.tool } : {}),
      ...(msg.goal ? { goal: msg.goal } : {}),
      ...(extra && !seen ? { line: extra } : {}),
    });
    return;
  }
  if (msg.type === "error") {
    bubble("system", msg.message || msg.code || "错误");
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const params = new URLSearchParams();
  const page = new URLSearchParams(location.search);
  if (page.get("token")) params.set("token", page.get("token"));
  if (sessionId) params.set("session_id", sessionId);
  const query = params.toString();
  const ws = new WebSocket(`${proto}://${location.host}/ws${query ? `?${query}` : ""}`);
  ws.binaryType = "arraybuffer";
  socket = ws;
  setStatus(connEl, "连接中");
  ws.onopen = () => {
    reconnect = 500;
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") handleMessage(ev.data);
    else handleBinary(ev.data);
  };
  ws.onclose = () => {
    setStatus(connEl, "已断开，重连中");
    socket = null;
    window.setTimeout(connect, reconnect);
    reconnect = Math.min(reconnect * 2, 5000);
  };
}

function sendJson(obj) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify(obj));
}

async function startMic() {
  if (micOn) return;
  await ensureAudio();
  if (audioCtx.state === "suspended") await audioCtx.resume();
  if (!captureReady) {
    await audioCtx.audioWorklet.addModule("/audio-capture.worklet.js");
    captureReady = true;
  }
  media = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  micSource = audioCtx.createMediaStreamSource(media);
  workletNode = new AudioWorkletNode(audioCtx, "duplex-capture");
  workletNode.port.onmessage = (ev) => {
    const { pcm, rms } = ev.data;
    levelEl.style.width = `${Math.min(100, rms * 400)}%`;
    frameIndex = (frameIndex + 1) >>> 0;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(encodeFrame(frameIndex, pcm, rms > 0.03));
    }
  };
  micSource.connect(workletNode);
  micOn = true;
  micBtn.classList.add("on");
  micBtn.textContent = "停止听";
}

function stopMic() {
  micOn = false;
  micBtn.classList.remove("on");
  micBtn.textContent = "开始听";
  if (micSource) micSource.disconnect();
  if (workletNode) workletNode.disconnect();
  if (media) media.getTracks().forEach((track) => track.stop());
  micSource = null;
  workletNode = null;
  media = null;
  levelEl.style.width = "0";
}

micBtn.addEventListener("click", () => {
  if (micOn) stopMic();
  else startMic().catch((err) => bubble("system", `麦克风不可用：${err.message || err}`));
});

cutBtn.addEventListener("click", () => {
  cutSpeech();
  sendJson({ type: "barge" });
});

form.addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = "";
  ensureAudio().then(() => audioCtx.resume()).catch(() => {});
  sendJson({ type: "text", text });
});

connect();

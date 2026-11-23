from collections.abc import Callable

import streamlit as st

_HTML = """
<div class="live-wrap">
  <div class="video-box">
    <video class="video" autoplay playsinline muted></video>
    <div class="guide"></div>
    <div class="pill" data-tone="idle">
      <span class="dot"></span>
      <span class="pill-text">Kamera belum aktif</span>
      <span class="pill-meta"></span>
    </div>
  </div>
  <div class="controls">
    <button type="button" class="toggle-btn">Mulai live</button>
    <span class="status-text"></span>
  </div>
</div>
"""

_CSS = """
.live-wrap {
  font-family: var(--st-font, sans-serif);
  color: var(--st-text-color);
  max-width: 520px;
}

.video-box {
  position: relative;
  width: 100%;
  aspect-ratio: 4 / 3;
  background: var(--st-secondary-background-color);
  border: 1px solid var(--st-border-color);
  border-radius: var(--st-base-radius, 8px);
  overflow: hidden;
}

.video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
  display: block;
}

.guide {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 58%;
  aspect-ratio: 1 / 1;
  transform: translate(-50%, -50%);
  border: 1.5px dashed rgba(255, 255, 255, 0.55);
  border-radius: 12px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s ease;
}

.video-box.active .guide {
  opacity: 1;
}

.pill {
  position: absolute;
  top: 10px;
  left: 10px;
  right: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(15, 17, 22, 0.72);
  color: #ffffff;
  font-size: 13px;
  line-height: 1.2;
  backdrop-filter: blur(6px);
  width: fit-content;
  max-width: calc(100% - 20px);
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #9ca3af;
  flex-shrink: 0;
}

.pill[data-tone="ok"] .dot {
  background: #22c55e;
}

.pill[data-tone="warn"] .dot {
  background: #ef4444;
}

.pill[data-tone="live"] .dot {
  background: #f59e0b;
}

.pill-text {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pill-meta {
  opacity: 0.78;
  font-size: 12px;
  white-space: nowrap;
}

.controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 10px;
  flex-wrap: wrap;
}

.toggle-btn {
  appearance: none;
  border: 1px solid var(--st-border-color);
  background: var(--st-background-color);
  color: var(--st-text-color);
  border-radius: var(--st-button-radius, 8px);
  padding: 6px 14px;
  font-family: inherit;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
}

.toggle-btn:hover {
  border-color: var(--st-primary-color);
  color: var(--st-primary-color);
}

.toggle-btn.active {
  background: var(--st-primary-color);
  border-color: var(--st-primary-color);
  color: #ffffff;
}

.status-text {
  font-size: 13px;
  opacity: 0.75;
}
"""

_JS = """
const instances = new WeakMap();

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const videoBox = parentElement.querySelector(".video-box");
  const video = parentElement.querySelector(".video");
  const pill = parentElement.querySelector(".pill");
  const pillText = parentElement.querySelector(".pill-text");
  const pillMeta = parentElement.querySelector(".pill-meta");
  const btn = parentElement.querySelector(".toggle-btn");
  const statusText = parentElement.querySelector(".status-text");

  if (!videoBox || !video || !pill || !pillText || !pillMeta || !btn || !statusText) {
    return;
  }

  let state = instances.get(parentElement);
  if (!state) {
    state = {
      stream: null,
      running: false,
      inFlight: false,
      seq: 0,
      lastSentSeq: 0,
      timerId: null,
      watchdogId: null,
      canvas: document.createElement("canvas"),
      lastModelTag: null,
      error: null,
      setTriggerValue,
      intervalMs: 600,
    };
    state.canvas.width = 256;
    state.canvas.height = 256;
    instances.set(parentElement, state);
  }

  state.setTriggerValue = setTriggerValue;
  state.intervalMs = (data && data.intervalMs) || 600;

  const clearTimers = () => {
    if (state.timerId !== null) {
      clearTimeout(state.timerId);
      state.timerId = null;
    }
    if (state.watchdogId !== null) {
      clearTimeout(state.watchdogId);
      state.watchdogId = null;
    }
  };

  const captureAndSend = () => {
    if (!state.running || state.inFlight) return;
    if (!video.videoWidth || !video.videoHeight) {
      state.timerId = setTimeout(captureAndSend, 200);
      return;
    }

    const side = Math.round(Math.min(video.videoWidth, video.videoHeight) * 0.75);
    const sx = Math.floor((video.videoWidth - side) / 2);
    const sy = Math.floor((video.videoHeight - side) / 2);
    const ctx = state.canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, sx, sy, side, side, 0, 0, 256, 256);
    const image = state.canvas.toDataURL("image/jpeg", 0.8);

    state.seq += 1;
    state.lastSentSeq = state.seq;
    state.inFlight = true;
    state.setTriggerValue("frame", { seq: state.seq, image });

    state.watchdogId = setTimeout(() => {
      if (state.inFlight && state.running) {
        state.inFlight = false;
        captureAndSend();
      }
    }, 5000);
  };

  const scheduleNext = (delay) => {
    if (!state.running) return;
    clearTimers();
    state.timerId = setTimeout(captureAndSend, delay);
  };

  const stopCamera = () => {
    state.running = false;
    state.inFlight = false;
    clearTimers();
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    video.srcObject = null;
    videoBox.classList.remove("active");
    btn.textContent = "Mulai live";
    btn.classList.remove("active");
    pill.dataset.tone = "idle";
    pillText.textContent = "Kamera belum aktif";
    pillMeta.textContent = "";
    state.setTriggerValue("stopped", Date.now());
  };

  const startCamera = async () => {
    state.error = null;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      state.stream = stream;
      state.running = true;
      state.inFlight = false;
      video.srcObject = stream;
      await video.play();
      videoBox.classList.add("active");
      btn.textContent = "Berhenti";
      btn.classList.add("active");
      pill.dataset.tone = "live";
      pillText.textContent = "Mendeteksi...";
      pillMeta.textContent = "";
      statusText.textContent = `Model ${data.modelTag} · posisikan wajah di kotak tengah`;
      scheduleNext(250);
    } catch (err) {
      state.error = err && err.message ? err.message : "Izin kamera ditolak";
      pill.dataset.tone = "warn";
      pillText.textContent = "Kamera tidak tersedia";
      pillMeta.textContent = "";
      statusText.textContent = state.error;
    }
  };

  btn.onclick = () => {
    if (state.running) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  const result = data && data.result;
  const modelChanged = state.lastModelTag !== null && state.lastModelTag !== data.modelTag;
  state.lastModelTag = data.modelTag;

  if (state.running) {
    videoBox.classList.add("active");
    btn.textContent = "Berhenti";
    btn.classList.add("active");
    statusText.textContent = `Model ${data.modelTag} · posisikan wajah di kotak tengah`;

    if (modelChanged) {
      state.inFlight = false;
      scheduleNext(100);
    } else if (result && result.modelTag === data.modelTag) {
      pill.dataset.tone = result.isAdult ? "ok" : "warn";
      pillText.textContent = result.isAdult
        ? `${result.age} th · Dewasa`
        : `${result.age} th · Di bawah ${data.ageLimit}`;
      pillMeta.textContent = `${result.elapsedMs} ms`;
      if (state.inFlight && result.seq >= state.lastSentSeq) {
        state.inFlight = false;
        scheduleNext(state.intervalMs);
      }
    }
  } else {
    videoBox.classList.remove("active");
    btn.textContent = "Mulai live";
    btn.classList.remove("active");
    if (!state.error) {
      statusText.textContent = `Model ${data.modelTag} · ambang T = ${data.recommendedT} th`;
    }
  }

  return () => {
    clearTimers();
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    state.running = false;
    state.inFlight = false;
    instances.delete(parentElement);
  };
}
"""

_LIVE_CAMERA = st.components.v2.component(
    "age_gating_live_camera",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def live_camera(
    *,
    model_tag: str,
    recommended_t: int,
    age_limit: int,
    result: dict | None,
    key: str,
    interval_ms: int = 600,
    on_frame_change: Callable[[], None] | None = None,
    on_stopped_change: Callable[[], None] | None = None,
):
    return _LIVE_CAMERA(
        data={
            "modelTag": model_tag,
            "recommendedT": recommended_t,
            "ageLimit": age_limit,
            "intervalMs": interval_ms,
            "result": result,
        },
        key=key,
        on_frame_change=on_frame_change or (lambda: None),
        on_stopped_change=on_stopped_change or (lambda: None),
    )

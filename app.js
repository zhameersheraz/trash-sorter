/* Trash Sorter - webcam waste classifier
   Model: Teachable Machine image classifier (TensorFlow.js) */

// tmImage.load() wants the model.json file itself. Newer builds whose
// loadFromURL exists expect the containing directory instead.
const MODEL_DIR = "./model/";
const MODEL_FILE = MODEL_DIR + "model.json";

// The CDN build of @teachablemachine/image exposes `tmImage` (lowercase),
// not the `TMImage` that Google's own export snippet assumes. It also has no
// loadFromURL: `load(model, metadata)` is async and returns a Promise, and it
// rejects a null metadata (it reads metadata.packageVersion downstream).
//
// Note imageSize stays a scalar. Internally cropTo() does `size / min` and
// `canvas.width = size`, so passing [224, 224] instead of 224 yields NaN and
// every prediction fails with "Requested texture size [0x0] is invalid".
const TM = window.tmImage || window.TMImage;

async function loadModel() {
  if (!TM) throw new Error("Teachable Machine library did not load");

  const res = await fetch(MODEL_DIR + "metadata.json");
  if (!res.ok) throw new Error("metadata.json not found in ./model/");
  const metadata = await res.json();

  let model;
  if (typeof TM.loadFromURL === "function") {
    model = await TM.loadFromURL(MODEL_DIR);
  } else {
    model = await TM.load(MODEL_FILE, metadata);
  }

  if (!model.labels && metadata.labels) model.labels = metadata.labels;
  return model;
}

// Which bin each detected class belongs to, and how to present it.
// Colour is the information here, not decoration. Each class carries the
// colour of the bin it belongs in, the way municipal recycling signage works:
// blue plastics, slate metal, amber paper, green organics, graphite residual.
const BIN_INFO = {
  "Plastic Bottle":    { bin: "Recycling", colour: "#1D4ED8", note: "plastic bottles" },
  "Aluminum Can":      { bin: "Recycling", colour: "#475569", note: "metal cans" },
  "Paper & Cardboard": { bin: "Recycling", colour: "#B45309", note: "paper & cardboard" },
  "Food Waste":        { bin: "Compost",   colour: "#15803D", note: "food scraps" },
  "Other":             { bin: "General",   colour: "#292524", note: "everything else" },
};

const FALLBACK = { bin: "Unknown", colour: "#292524", note: "" };

const CONFIDENCE_FLOOR = 0.55; // below this, don't claim a confident answer

const el = {
  status: document.getElementById("status"),
  statusText: document.getElementById("statusText"),
  viewport: document.getElementById("viewport"),
  cam: document.getElementById("cam"),
  still: document.getElementById("still"),
  toggleCam: document.getElementById("toggleCam"),
  fileInput: document.getElementById("fileInput"),
  verdict: document.getElementById("verdict"),
  confTop: document.getElementById("confTop"),
  resultName: document.getElementById("resultName"),
  binName: document.getElementById("binName"),
  binNote: document.getElementById("binNote"),
  binFill: document.getElementById("binFill"),
  bars: document.getElementById("bars"),
  binLegend: document.getElementById("binLegend"),
  announcer: document.getElementById("announcer"),
};

let model = null;
let labels = [];
let stream = null;
let rafId = null;
let lastPredict = 0;
let lastAnnounced = "";
// ~5 predictions a second. Any faster wastes battery on a phone and the
// library never disposes its output tensor, so the rate is also the leak rate.
const PREDICT_INTERVAL_MS = 200;


/* ------------------------------------------------------------------ */
/* status                                                               */
/* ------------------------------------------------------------------ */

function setStatus(state, text) {
  el.status.dataset.state = state;
  el.statusText.textContent = text;
}

// The one-line explainer under the bin bar. Anything that needs to say
// something to the user goes here rather than into a separate toast.
function say(text) {
  el.binNote.textContent = text;
}

// Without a live source there is nothing to report. The panel must never
// keep showing a stale answer, because "Plastic Bottle / 0%" reads as a
// confident result when it is not one.
function setIdle() {
  el.verdict.dataset.empty = "true";
  el.verdict.style.removeProperty("--signal");
  el.resultName.textContent = "Nothing yet";
  el.confTop.textContent = "—";
  el.binFill.style.width = "0%";
  el.binName.textContent = "Bin unknown";
  el.bars.innerHTML = "";
  say("Point the camera at an item");
  lastAnnounced = "";
  el.announcer.textContent = "";
}

/* ------------------------------------------------------------------ */
/* legend                                                               */
/* ------------------------------------------------------------------ */

function buildLegend() {
  const seen = new Set();
  el.binLegend.innerHTML = "";

  labels.forEach((label) => {
    const info = BIN_INFO[label] || FALLBACK;
    if (seen.has(info.bin)) return;
    seen.add(info.bin);

    const li = document.createElement("li");
    li.innerHTML =
      '<span class="chip" style="background:' + info.colour + '"></span>' +
      "<span>" + escapeHtml(info.bin) +
      ' <span class="key-note">' + escapeHtml(info.note) + "</span></span>";
    el.binLegend.appendChild(li);
  });
}

/* ------------------------------------------------------------------ */
/* rendering                                                            */
/* ------------------------------------------------------------------ */

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function render(predictions) {
  // Guard on the source, not just the prediction: a stale answer is worse
  // than no answer.
  if (!sourceEl || !predictions || !predictions.length) return;

  // predict() returns classes in TRAINING order, not by score — only
  // predictTopK() sorts. Reading predictions[0] as the winner showed a 0%
  // class as the headline result while the real answer sat at 100%.
  const ranked = predictions
    .slice()
    .sort((a, b) => b.probability - a.probability);

  const top = ranked[0];
  const info = BIN_INFO[top.className] || FALLBACK;
  const confident = top.probability >= CONFIDENCE_FLOOR;
  const pct = Math.round(top.probability * 100);

  // One signal colour drives the whole verdict bay.
  el.verdict.dataset.empty = "false";
  el.verdict.style.setProperty("--signal", info.colour);

  el.resultName.textContent = top.className;
  el.confTop.textContent = pct + "%";

  el.binName.textContent = confident ? info.bin : "Uncertain";
  el.binNote.textContent = confident ? info.note : "hold it closer, better lit";
  el.binFill.style.width = pct + "%";

  el.bars.innerHTML = "";
  ranked.forEach((p, i) => {
    const v = Math.round(p.probability * 100);
    const li = document.createElement("li");
    if (i === 0) li.className = "is-top";
    li.innerHTML =
      '<span class="rd-name">' + escapeHtml(p.className) + "</span>" +
      '<span class="rd-pct">' + v + "%</span>" +
      '<span class="rd-track"><span class="rd-fill"></span></span>';
    const fill = li.querySelector(".rd-fill");
    requestAnimationFrame(() => {
      fill.style.width = v + "%";
      fill.style.background = i === 0 ? info.colour : "";
    });
    el.bars.appendChild(li);
  });

  // Announce on change of winner only. Re-announcing every frame would make a
  // screen reader repeat the same sentence several times a second.
  if (confident && top.className !== lastAnnounced) {
    lastAnnounced = top.className;
    el.announcer.textContent = top.className + ", " + info.bin + " bin";
  }
}

/* ------------------------------------------------------------------ */
/* camera                                                               */
/* ------------------------------------------------------------------ */

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("error", "Camera not supported here");
    say("camera blocked here — load a photo instead");
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 640 } },
      audio: false,
    });

    el.cam.srcObject = stream;
    await el.cam.play();

    sourceEl = el.cam;
    el.viewport.classList.add("is-live");
    el.viewport.classList.remove("photo");
    el.toggleCam.textContent = "Stop camera";
    setStatus("ready", "Camera live");

    clearPhoto();
    loop();
  } catch (err) {
    const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
    setStatus("error", denied ? "Camera blocked" : "Camera unavailable");
    say(denied
      ? "allow camera in the address bar, or load a photo"
      : "no camera found — load a photo instead");
  }
}

function stopCamera() {
  cancelAnimationFrame(rafId);
  rafId = null;

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  el.cam.srcObject = null;
  if (sourceEl === el.cam) sourceEl = null;
  el.viewport.classList.remove("is-live", "photo");
  el.toggleCam.textContent = "Start camera";
  setStatus("ready", "Camera off");
  setIdle();
}

/* ------------------------------------------------------------------ */
/* photo mode                                                           */
/* ------------------------------------------------------------------ */

/* A <video> element cannot decode a still PNG, so photo mode feeds the
   model a real <img>. `sourceEl` is whichever element predict() reads. */
let sourceEl = null;
let lastError = null;

function clearPhoto() {
  el.still.removeAttribute("src");
  el.viewport.classList.remove("photo");
  if (sourceEl === el.still) sourceEl = null;
}

function usePhoto(file) {
  if (!file) return;

  stopCamera();

  const url = URL.createObjectURL(file);
  el.still.onload = () => {
    // tf.browser.fromPixels derives its size from the element's layout box,
    // which is 0x0 for a CSS-sized <img>. Predict from a canvas at the
    // image's intrinsic size instead.
    const canvas = document.createElement("canvas");
    canvas.width = el.still.naturalWidth || el.still.width;
    canvas.height = el.still.naturalHeight || el.still.height;
    canvas.getContext("2d").drawImage(el.still, 0, 0, canvas.width, canvas.height);

    sourceEl = canvas;
    el.viewport.classList.add("is-live", "photo");
    el.toggleCam.textContent = "Start camera";
    setStatus("ready", "Photo loaded");
    loop();
    URL.revokeObjectURL(url);
  };
  el.still.onerror = () => {
    setStatus("error", "Could not read that image");
    URL.revokeObjectURL(url);
  };
  el.still.src = url;
}

/* ------------------------------------------------------------------ */
/* prediction loop                                                      */
/* ------------------------------------------------------------------ */

function loop() {
  cancelAnimationFrame(rafId);
  const step = async () => {
    rafId = requestAnimationFrame(step);

    const now = performance.now();
    if (now - lastPredict < PREDICT_INTERVAL_MS) return;
    lastPredict = now;

    const src = sourceEl;
    if (!model || !src) return;

    // A <video> needs decoded frames; a still <img> needs its intrinsic
    // size. A canvas is already sized, so it is ready as soon as it exists.
    if (src.tagName === "VIDEO") {
      if (src.readyState < 2 || src.videoWidth === 0) return;
    } else if (src.tagName === "IMG") {
      if (!src.naturalWidth) return;
    } else if (!src.width || !src.height) {
      return;
    }

    try {
      const predictions = await model.predict(src);
      render(predictions);
    } catch (e) {
      // A silently blank result looks like "the model is broken". Report it.
      if (lastError !== e.message) {
        lastError = e.message;
        say("prediction failed: " + e.message);
        console.error("prediction failed:", e);
      }
      return;
    }
    lastError = null;
  };
  step();
}

/* ------------------------------------------------------------------ */
/* init                                                                 */
/* ------------------------------------------------------------------ */

async function init() {
  // Attach handlers immediately. The model takes a moment to load, and a user
  // who picks a photo during that window would otherwise have it silently
  // dropped.
  let pendingFile = null;

  el.toggleCam.addEventListener("click", () => {
    if (stream) stopCamera();
    else startCamera();
  });

  el.fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    // Clear the value so re-picking the SAME file still fires change.
    // Otherwise the second pick is silently ignored.
    e.target.value = "";
    if (!file) return;
    if (model) usePhoto(file);
    else pendingFile = file;
  });

  window.addEventListener("beforeunload", stopCamera);

  try {
    setStatus("loading", "Loading model…");

    model = await loadModel();
    labels = model.labels;

    setStatus("ready", "Model ready");
    el.toggleCam.disabled = false;
    buildLegend();
    setIdle();

    if (pendingFile) {
      const file = pendingFile;
      pendingFile = null;
      usePhoto(file);
    }
  } catch (err) {
    setStatus("error", "Model failed to load");
    say("model folder missing — check it sits next to index.html (" + err.message + ")");
  }
}

init();
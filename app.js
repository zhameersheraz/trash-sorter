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
const BIN_INFO = {
  "Plastic Bottle": {
    bin: "Recycling Bin", colour: "#2563eb", icon: "P",
    note: "Plastic & bottles",
  },
  "Aluminum Can": {
    bin: "Recycling Bin", colour: "#d97706", icon: "M",
    note: "Metal cans",
  },
  "Paper & Cardboard": {
    bin: "Recycling Bin", colour: "#ca8a04", icon: "R",
    note: "Paper & cardboard",
  },
  "Food Waste": {
    bin: "Compost Bin", colour: "#16a34a", icon: "C",
    note: "Organic & compostable",
  },
  "Other": {
    bin: "General Waste", colour: "#64748b", icon: "G",
    note: "Non-recyclable",
  },
};

const FALLBACK = { bin: "Unknown", colour: "#94a3b8", icon: "?", note: "" };

const CONFIDENCE_FLOOR = 0.55; // below this, don't claim a confident answer

const el = {
  status: document.getElementById("status"),
  statusText: document.getElementById("statusText"),
  stage: document.querySelector(".video-wrap"),
  cam: document.getElementById("cam"),
  still: document.getElementById("still"),
  toggleCam: document.getElementById("toggleCam"),
  fileInput: document.getElementById("fileInput"),
  resultCard: document.getElementById("resultCard"),
  resultName: document.getElementById("resultName"),
  resultSub: document.getElementById("resultSub"),
  binIcon: document.getElementById("binIcon"),
  binName: document.getElementById("binName"),
  binNote: document.getElementById("binNote"),
  binBox: document.querySelector(".bin"),
  confTop: document.getElementById("confTop"),
  bars: document.getElementById("bars"),
  legend: document.getElementById("binLegend"),
};

let model = null;
let labels = [];
let stream = null;
let rafId = null;
let lastPredict = 0;
const PREDICT_INTERVAL_MS = 120;


/* ------------------------------------------------------------------ */
/* status                                                               */
/* ------------------------------------------------------------------ */

function setStatus(state, text) {
  el.status.dataset.state = state;
  el.statusText.textContent = text;
}

/* ------------------------------------------------------------------ */
/* legend                                                               */
/* ------------------------------------------------------------------ */

function buildLegend() {
  const seen = new Set();
  el.legend.innerHTML = "";

  labels.forEach((label) => {
    const info = BIN_INFO[label] || FALLBACK;
    if (seen.has(info.bin)) return;
    seen.add(info.bin);

    const li = document.createElement("li");
    li.innerHTML =
      '<span class="swatch" style="background:' + info.colour + '"></span>' +
      '<span><span class="lg-name">' + info.bin + "</span>" +
      '<span class="lg-bin">' + escapeHtml(info.note) + "</span></span>";
    el.legend.appendChild(li);
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
  if (!predictions || !predictions.length) return;

  const top = predictions[0];
  const info = BIN_INFO[top.className] || FALLBACK;
  const confident = top.probability >= CONFIDENCE_FLOOR;

  el.resultCard.dataset.empty = "false";
  el.resultName.textContent = top.className;
  el.resultSub.textContent = confident
    ? Math.round(top.probability * 100) + "% sure"
    : "Low confidence - try holding it closer and better lit";

  el.binIcon.textContent = info.icon;
  el.binIcon.style.background = info.colour + "22";
  el.binIcon.style.color = info.colour;
  el.binName.textContent = info.bin;
  el.binNote.textContent = info.note;
  el.binBox.style.background = info.colour + "0f";
  el.binBox.style.borderColor = info.colour + "33";

  el.confTop.textContent = Math.round(top.probability * 100) + "%";
  el.confTop.style.color = info.colour;

  el.bars.innerHTML = "";
  predictions.forEach((p) => {
    const pct = Math.round(p.probability * 100);
    const colour = (BIN_INFO[p.className] || FALLBACK).colour;

    const li = document.createElement("li");
    li.innerHTML =
      '<div class="bar-label"><span>' + escapeHtml(p.className) + "</span>" +
      '<span class="pct">' + pct + "%</span></div>" +
      '<div class="track"><div class="fill"></div></div>';

    const fill = li.querySelector(".fill");
    // set width on next frame so the CSS transition actually runs
    requestAnimationFrame(() => {
      fill.style.width = pct + "%";
      fill.style.background = colour;
    });

    el.bars.appendChild(li);
  });
}

/* ------------------------------------------------------------------ */
/* camera                                                               */
/* ------------------------------------------------------------------ */

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("error", "Camera not supported here");
    el.resultSub.textContent =
      "This browser blocks camera access. Use the “Use a photo” button instead.";
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
    el.stage.classList.add("live");
    el.stage.classList.remove("photo");
    el.toggleCam.textContent = "Stop camera";
    setStatus("ready", "Camera live");

    clearPhoto();
    loop();
  } catch (err) {
    const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
    setStatus("error", denied ? "Camera blocked" : "Camera unavailable");
    el.resultSub.textContent = denied
      ? "Camera permission was denied. Allow it in the address bar, or use “Use a photo”."
      : "No camera found. Use the “Use a photo” button instead.";
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
  el.stage.classList.remove("live", "photo");
  el.toggleCam.textContent = "Start camera";
  setStatus("ready", "Camera off");
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
  el.stage.classList.remove("photo");
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
    el.stage.classList.add("live", "photo");
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
        el.resultSub.textContent = "Prediction failed: " + e.message;
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

    if (pendingFile) {
      const file = pendingFile;
      pendingFile = null;
      usePhoto(file);
    }
  } catch (err) {
    setStatus("error", "Model failed to load");
    el.resultSub.textContent =
      "Could not load ./model/. Make sure this page is served over http://localhost " +
      "and the model folder sits next to index.html. (" + err.message + ")";
  }
}

init();
# Trash Sorter

**Point a camera at litter. Get told which bin it goes in.**

An AI waste-classification web app. A custom image model trained in
[Google Teachable Machine](https://teachablemachine.withgoogle.com/) runs entirely
in the browser on TensorFlow.js — no server, no API calls, no upload of camera
frames.

**Live demo →** **[trash-sorter.enjeanearz.workers.dev](https://trash-sorter.enjeanearz.workers.dev)**

---

## What it does

![Plastic bottle detected](docs/screenshot-plastic-bottle.jpg)

Point the webcam at an item and the app classifies it into one of five waste
classes, then tells you the destination bin. A **Load a photo** button does the
same from a still image, so the demo works even if camera permission is blocked.

| Detected class | Destination | Colour |
|---|---|---|
| Plastic Bottle | Recycling — plastic bottles | Blue |
| Aluminum Can | Recycling — metal cans | Slate |
| Paper & Cardboard | Recycling — paper & cardboard | Amber |
| Food Waste | Compost — food scraps | Green |
| Other | General — everything else | Graphite |

**Colour is the information, not decoration.** Each class carries the colour of
the bin it belongs in, the way municipal recycling signage already works, so the
answer is legible before you finish reading it. The bar under the verdict fills
to the model's confidence in that class's colour.

The mark is a camera viewfinder with the answer bar inside it — which is a
literal picture of what this app does: you point a camera at an item, and a
colour comes back.

<details>
<summary>More screenshots</summary>

![Aluminum can detected](docs/screenshot-aluminum-can.jpg)
![Food waste detected](docs/screenshot-food-waste.jpg)
![Paper and cardboard detected](docs/screenshot-paper-cardboard.jpg)

</details>

---

## How it works

1. The model was **trained in Teachable Machine** as an image classifier over
   five classes, using 817 curated images (50 epochs, batch size 16,
   learning rate 0.001 — the defaults).
2. It was **exported as TensorFlow.js** into [`model/`](model).
3. [`app.js`](app.js) loads it with `@teachablemachine/image` and runs a
   prediction loop over the webcam frame (or the loaded photo) about five
   times a second.
4. The top class drives the verdict panel and the bin colour.

TensorFlow.js and the Teachable Machine library are **vendored into
[`vendor/`](vendor)**, including both webfonts, so the app works with no
internet connection — useful for a classroom demo on unreliable wifi.

## Accuracy

The exported model was re-run in the browser over a 150-image sample of the
dataset (30 per class) through the exact same pipeline the app uses — the same
224-pixel centre crop and the same `predict()` call.

| True class | Correct | Accuracy | Mean confidence |
|---|---|---|---|
| Plastic Bottle | 30/30 | 100% | 100% |
| Aluminum Can | 30/30 | 100% | 100% |
| Other | 30/30 | 100% | 100% |
| Food Waste | 29/30 | 97% | 99% |
| Paper & Cardboard | 29/30 | 97% | 98% |
| **Overall** | **148/150** | **98.7%** | — |

**This is a sample of the images the model was trained on, so it is
optimistic** — it measures how well the model fits its own training data, not
how it generalises to new objects. The honest test is a live camera pointed at
real litter, which is what the demo does.

The confidence numbers are what the app is built around:

| Model confidence | Images | Correct |
|---|---|---|
| 95% – 100% | 147 | 147 |
| 85% – 95% | 1 | 1 |
| 70% – 85% | 1 | 0 |
| below 70% | 1 | 0 |

Every correct answer came in at **85% confidence or above**; both wrong answers
came in below 75%. The app therefore uses a **0.75 confidence floor**: below
that it says **Uncertain** instead of naming a bin, so it never tells you to
recycle something it is only guessing about. That threshold is set from these
measurements, not by taste.

## Stack

- Google Teachable Machine (model training + TensorFlow.js export)
- TensorFlow.js 3.21 (in-browser inference)
- Vanilla HTML, CSS and JavaScript — no build step, no framework
- Archivo + IBM Plex Mono (self-hosted)
- Cloudflare Workers (static hosting, auto-deploy from `main`)

## Run it locally

The camera API requires a secure context, so open it over `http://localhost`
rather than double-clicking `index.html`:

```bash
git clone https://github.com/zhameersheraz/trash-sorter.git
cd trash-sorter
python -m http.server 8000
```

Then open <http://localhost:8000> and allow camera access.

## Project structure

```
├── index.html            markup
├── styles.css            design system + layout
├── app.js                model loading, prediction loop, rendering
├── model/                exported TensorFlow.js model (required)
│   ├── model.json
│   ├── weights.bin
│   └── metadata.json     class labels + image size
├── vendor/               vendored libs and fonts (offline support)
├── docs/                 screenshots used by this README
└── .gitignore
```

---

## Notes for anyone extending this

The CDN build of `@teachablemachine/image` does **not** match Google's own export
snippet, and the mismatch is silent:

- The global is **`tmImage`** (lowercase). `TMImage` is undefined.
- There is no `loadFromURL`. `load(model, metadata)` is **async and
  Promise-returning** — passing callbacks makes it hang forever.
- `metadata` is mandatory, but `imageSize` must stay the **scalar** `224`.
  Passing `[224, 224]` yields NaN and every prediction dies with
  `Requested texture size [0x0] is invalid`.
- `predict()` returns classes in **training order**, not sorted by score. Only
  `predictTopK()` sorts. Sorting in `render()` is required or the headline
  result can show a 0% class while the real answer sits at 100%.

A `<video>` element also cannot decode a still image, so photo mode predicts
from a canvas at the image's intrinsic size.

## Accessibility notes

- The file input is clipped rather than `hidden`, so it stays keyboard-reachable;
  focus is drawn on the label that stands in for it.
- The verdict panel updates about five times a second, so it is **not** itself an
  `aria-live` region. A separate visually-hidden status element announces only
  when the winning class actually changes, instead of narrating every frame.
- `prefers-reduced-motion` disables all transitions.

## Dataset

Training images were curated from the
[Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)
dataset by Alistair King (MIT licence), with unusable samples removed by hand.

What each class actually contains:

| Class | Images | Source sub-categories |
|---|---|---|
| Plastic Bottle | 150 | 113 water bottles, 37 soda bottles |
| Aluminum Can | 150 | 150 soda cans |
| Paper & Cardboard | 136 | 60 newspaper, 53 cardboard boxes, 23 magazine covers |
| Food Waste | 156 | 83 food waste, 45 eggshells, 15 coffee grounds, 13 tea bags |
| Other | 225 | 60 styrofoam cups, 53 plastic cutlery, 45 straws, 37 styrofoam containers, 30 plastic trash bags |

96 images were quarantined during curation and are not in the training set:
71 outright unusable samples (retail tea boxes, text infographics, webpage
screenshots), and 25 more identified later — 12 magazine covers dominated by a
large human face (which teach the model that a face is paper) and 13 remaining
text infographics in Food Waste. Those 25 are removed from the staging folders
and will be dropped in the next training run.

## Known limitations

**Every training image is an isolated product shot on a plain background.** The
model has never seen an object in a room, in a hand, or on a table. That has a
measurable effect: when a real object is cut out and pasted onto a real
background, the model's answer depends heavily on how much of the frame the
object fills.

| Object | Isolated photo | Same object on a real wall |
|---|---|---|
| Water bottle | Plastic Bottle 99% | Plastic Bottle 100% (fills frame) / 53% (small in frame) |
| Food waste | Food Waste 100% | Food Waste 100% |
| Styrofoam cup | Other 100% | Other 100% |
| Cardboard box | Paper & Cardboard 100% | Paper & Cardboard 79% (fills frame) / Food Waste 68% (small in frame) |

In other words: **fill the frame with the item.** The "Uncertain" verdict is
there for the cases where the model is genuinely unsure — a large flat
background with a small object in it produces a confident but wrong answer, and
that is the model's weakest spot.

**The "Other" class is not a true catch-all.** It was trained on single-use
plastics (cups, cutlery, straws, bags), so "Other" means "single-use plastics",
not "everything else". Anything outside the four specific classes is pushed
into it whether or not it belongs.

**There is no "not waste" class.** A person in frame has to be forced into one
of the five, and the model will pick whichever is closest — a face against a
tan wall comes out as Paper & Cardboard. The confidence floor catches this and
answers "Uncertain" instead of naming a bin.

## Team

**Enjeanearz**

- Tampugao, Zhameer Sheraz
- Malolot, Brent Jacob
- Zulieta, Jaypee
- Abay, Aenon John
- Pe&ntilde;aranda, Ajay
- Mosqueda, John Razel

---

Built by **Enjeanearz**.
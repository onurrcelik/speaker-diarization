# Who Is Talking: To-Do Plan

Oct 8, 2026 · @Onur

## Overview

The notebook labels every moment of a video as `me` (Onur talking), `other` (someone else talking) or `silence` (no one talking). The earlier `friend` class was dropped on 2026-10-07.

- **Status:** notebook converted to 3 classes and run on Onur's 14 phone recordings (see `README.md` for the numbers). Still missing: real room tone and the demo video.
- **Notebook:** `notebook.ipynb`. Step 1 Data, Step 2 Clean, Step 3 Models, Step 4 Metrics, Step 5 Video demo.
- **Pipeline:** Silero VAD (is anyone talking?) → ECAPA-TDNN speaker embedding (whose voice?) → logistic regression with an "other" threshold → smoothing → labelled video.
- **Time needed from me:** about 15 min of recording, plus a 15–20 min notebook run on the CPU.

## Part 1: Record (phone is fine, any format)

Record several short sessions rather than one long one; the test split is a whole session the model never saw.

- [x] **Me:** 4 sessions recorded (sessions 1, 2, 3, 5), ~12 min of speech in `recordings/New Recording 50–63.m4a`. Session 1 was recorded in the demo room.
- [x] **Clip → session mapping confirmed:** 50–55 = session 1, 56–61 = session 2, 62 = session 3 (phone-call style), 63 = session 5 (Turkish).
- [x] **Room tone:** taken from the pauses in session 1 (demo room); no separate clips needed.
- [ ] **Other people (optional):** another person on the same phone, or a podcast or YouTube video played aloud in the room. Helps the demo; LibriSpeech strangers are used anyway.
- [ ] **Live demo** instead of a filmed video: run `live_demo.py`, open the proxy link, start the webcam, talk / pause / let someone else talk.

Using the same phone and room as the demo for at least one session matters most: microphone mismatch is the top cause of errors.

## Part 2: Upload, label, run

Drag files into these folders in the Jupyter file browser. Leave the `DRYRUN_*` files alone; they are ignored once my own files are there.

| Recording | Folder |
| --- | --- |
| My sessions | `data/raw/me/` (named `s<N>_<anything>`, one prefix per session) |
| Room tone | `data/raw/silence/` |
| Other people (optional) | `data/raw/other/` |
| Demo video, renamed to `demo.mp4` | `data/demo/` |

- [x] My recordings copied to `data/raw/me/` (done automatically by the notebook from `recordings/`)
- [ ] (Optional) laptop-mic session recorded from the live page if `me` is weak through the webcam mic
- [ ] **(Optional, needed for a demo accuracy score)** Watch the demo and write `data/demo/demo_labels.csv`, one row per stretch of time (template: `demo_labels_TEMPLATE.csv`):

```csv
start,end,label
0.0,4.0,silence
4.0,14.5,me
14.5,25.0,other
```

- [ ] Open `notebook.ipynb` → **Kernel → Restart Kernel and Run All Cells** (15–20 min)
- [x] Check Step 1 output says `DRY_RUN = False`
- [x] Check Step 2 for a warning table; re-record any file flagged as clipped or under 30% speech
- [ ] Listen to the Step 2b sample clips: `me` should be me

## Part 3: Review results

The demo timeline numbers are the honest ones once a real, labelled demo video exists. The window-level scores are measured on a whole held-out session.

| Metric | Where | What it tells me |
| --- | --- | --- |
| Macro-F1, balanced accuracy | Step 4A | Overall quality on the held-out session vs. unseen strangers |
| EER for me | Step 4B | How separable my voice is from strangers (lower is better) |
| VAD miss / false alarm | Step 4C | Speech missed / noise called speech |
| Frame accuracy, DER | Step 4D | Errors on the labelled demo video |
| Real-time factor | Step 5 | Processing time ÷ video length |

- [ ] Try the live page; watch the synthetic `data/demo/synthetic_demo_annotated.mp4` inside the notebook
- [ ] Note the numbers from the Summary cell (also saved to `models/metrics.json`)

**If results are weak, try these in order:**

1. Record more sessions on the same phone and in the same room as the demo.
2. Add people who appear in the demo to `data/raw/other/`.
3. Tune `WIN_S`, `SPEECH_FRAC_MIN` and the smoothing settings.
4. Swap ECAPA for a stronger model (WeSpeaker ResNet293) or fine-tune WavLM; this needs a GPU.

Known limit: when two people talk at once, the louder voice wins.

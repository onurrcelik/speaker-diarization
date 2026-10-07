# Who Is Talking: To-Do Plan

Oct 7, 2026 · @Onur

## Overview

The pipeline is built and tested; the only thing missing is real recordings of me and my friend. The notebook labels every moment of a video as `me`, `friend`, `other` (someone else) or `silence` (no one talking).

- **Status:** dry run passed on 2026-10-07 with two LibriSpeech speakers standing in for me and my friend. On the synthetic demo video: 92.1% frame accuracy, 10.0% DER.
- **Notebook:** `free-lab/notebook.ipynb`. Step 1 Data, Step 2 Clean, Step 3 Models, Step 4 Metrics, Step 5 Video demo.
- **Pipeline:** Silero VAD (is anyone talking?) → ECAPA-TDNN speaker embedding (whose voice?) → logistic regression with an "other" threshold → smoothing → labelled video.
- **Time needed from me:** about 1 hour of recording and uploading, plus a 15–20 min notebook run on the CPU.

## Part 1: Record (phone is fine, any format)

Record several short sessions rather than one long one; the test split is a whole session the model never saw.

What to read and say in each session, plus a timed demo script: Recording Scripts

- [ ] **Me:** 3 or more recordings, 2–3 min each, only my voice. Vary them: one reading aloud, one chatting, one in a different room or distance.
- [ ] **Friend:** 3 or more recordings, 2–3 min each, only their voice. Get their consent first.
- [ ] **Room tone:** 2–3 clips of about 30 s with nobody talking (fan, typing, street noise).
- [ ] **Other people (optional):** another person, or a podcast or YouTube video played aloud in the room.
- [ ] **Demo video (1–2 min, same phone):**
  1. A few seconds of nobody talking
  2. Me talking (\~10 s)
  3. Friend talking (\~10 s)
  4. A pause
  5. Someone else talking (\~10 s)
  6. A few quick back-and-forths between me and my friend

Using the same phone and room as the demo for at least one session matters most: microphone mismatch is the top cause of errors.

## Part 2: Upload, label, run

Drag files into these folders in the Jupyter file browser. Leave the `DRYRUN_*` files alone; they are ignored once my own files are there.

| Recording | Folder |
| --- | --- |
| My sessions | `free-lab/data/raw/me/` |
| Friend's sessions | `free-lab/data/raw/friend/` |
| Room tone | `free-lab/data/raw/silence/` |
| Other people (optional) | `free-lab/data/raw/other/` |
| Demo video, renamed to `demo.mp4` | `free-lab/data/demo/` |

- [ ] Upload all recordings to the folders above
- [ ] **(Optional, needed for a demo accuracy score)** Watch the demo and write `data/demo/demo_labels.csv`, one row per stretch of time (template: `demo_labels_TEMPLATE.csv`):

```csv
start,end,label
0.0,4.0,silence
4.0,14.5,me
14.5,25.0,friend
```

- [ ] Open `notebook.ipynb` → **Kernel → Restart Kernel and Run All Cells** (15–20 min)
- [ ] Check Step 1 output says `DRY_RUN = False`
- [ ] Check Step 2 for a warning table; re-record any file flagged as clipped or under 30% speech
- [ ] Listen to the Step 2b sample clips: `me` should be me, `friend` should be my friend

## Part 3: Review results

The demo timeline numbers are the honest ones; the window-level scores were 1.000 in the dry run only because LibriSpeech is clean studio speech.

| Metric | Where | What it tells me | Dry-run value |
| --- | --- | --- | --- |
| Macro-F1, balanced accuracy | Step 4A | Overall quality on held-out sessions | 1.000 |
| EER for me / friend | Step 4B | How separable our voices are (lower is better) | 0.0% |
| VAD miss / false alarm | Step 4C | Speech missed / noise called speech | 0 / 0 |
| Frame accuracy, DER | Step 4D | Errors on the labelled demo video | 92.1%, 10.0% |
| Real-time factor | Step 5 | Processing time ÷ video length | 1.04 |

- [ ] Watch `data/demo/demo_annotated.mp4` (plays inside the notebook)
- [ ] Note the numbers from the Summary cell (also saved to `models/metrics.json`)

**If results are weak, try these in order:**

1. Record more sessions on the same phone and in the same room as the demo.
2. Add people who appear in the demo to `data/raw/other/`.
3. Tune `WIN_S`, `SPEECH_FRAC_MIN` and the smoothing settings.
4. Swap ECAPA for a stronger model (WeSpeaker ResNet293) or fine-tune WavLM; this needs a GPU.

Known limit: when both people talk at once, the louder voice wins.

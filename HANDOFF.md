# Session handoff — "Who is talking?" (me / someone else / no one)

Written 2026-10-07 at the end of a session. Read this first next time; it is the
full context, including what changed since the plan in `Who Is Talking To-Do Plan.md`.

## Task (original brief)

> identify if I am talking or if someone else is talking
> steps: find/get data (human in the loop) → clean data → recommend pretrained models →
> write down performance metrics → video demo showing *I am talking / no one / someone else*.
> Use a notebook.

## Scope change (decided 2026-10-07)

**The `friend` class is dropped.** Final labels are three:

| label | meaning |
|---|---|
| `me` | Onur is talking |
| `other` | someone who is not Onur is talking |
| `silence` | nobody is talking |

The committed `notebook.ipynb` still has 4 classes (`me, friend, other, silence`). It needs the edits listed under *Next steps*.

## What exists in this repo

| path | state |
|---|---|
| `notebook.ipynb` | 31-cell pipeline, **4-class version**, dry run passed 2026-10-07 (LibriSpeech stand-ins: 92.1 % frame acc, 10.0 % DER on synthetic demo). Not yet run on real data. |
| `recordings/New Recording 50–63.m4a` | 14 clips of **Onur only**, phone, AAC 48 kHz mono. ~12.4 min speech total. Analysed below. |
| `Who Is Talking To-Do Plan.md` | Plan export from the previous session (4-class wording; update after the notebook is converted). |
| `requirements.txt` | torch, torchaudio, speechbrain, silero-vad, soundfile, librosa, imageio-ffmpeg, opencv-python-headless, scikit-learn, seaborn, joblib, pandas, matplotlib, tqdm |
| `.gitignore` | `data/`, `models/`, `.venv/`, checkpoints — so generated data/models are never pushed (by design). |
| `README.md` | placeholder, two lines. |

Lost with the old workspace (all regenerable by re-running): `data/` (LibriSpeech dev-clean, DRYRUN files), `models/` (ECAPA weights, `who_is_talking.joblib`, `metrics.json`), annotated demo video, and a "Recording Scripts" doc that the plan links to (not exported; not needed).

## Recordings analysis (Silero VAD, same settings as notebook Step 2)

| file | length | speech frac | level dBFS | clipping |
|---|---|---|---|---|
| 50 | 1:44 | 0.90 | −17.8 | 0 |
| 51 | 0:37 | 0.87 | −18.7 | 0 |
| 52 | 0:29 | 0.91 | −21.6 | 0 |
| 53 | 0:36 | 0.83 | −20.3 | 0 |
| 54 | 0:29 | 0.90 | −19.8 | 0 |
| 55 | 0:40 | 0.67 | −20.4 | 0 |
| 56 | 1:34 | 0.92 | −17.2 | 0 |
| 57 | 0:31 | 0.92 | −17.6 | 0 |
| 58 | 0:23 | 0.74 | −18.7 | 0 |
| 59 | 0:32 | 0.84 | −18.6 | 0 |
| 60 | 0:20 | 0.80 | −18.5 | 0 |
| 61 | 0:31 | 0.70 | −18.5 | 0 |
| 62 | 3:29 | 0.80 | −17.0 | 0 |
| 63 | 2:48 | 0.87 | −19.3 | 0 |

- All 14 clips are speech; **no room-tone clips exist yet** (notebook falls back to synthetic noise; real room tone from the demo room is a nice-to-have).
- Nothing clipped, levels consistent. Nothing needs re-recording.
- Onur recorded **sessions 1, 2, 3 and 5** (skipped 4). **Session 1 was recorded in the room where the demo will be filmed** → keep session 1 in *train* so the model sees that room/mic.

### OPEN QUESTION 1 — file → session mapping (ask Onur first thing)

The notebook's train/val/test split is **by session** (whole session held out as test). Which `New Recording NN` files belong to session 1 / 2 / 3 / 5 is not recoverable from the audio. Likely shape: ten ~20–40 s clips are script items, four long ones (50, 56, 62, 63) are free talk — but confirm.

### OPEN QUESTION 2 — "someone else" data

LibriSpeech dev-clean strangers only (works out of the box), or also a clip of another person on the same phone (helps the demo)? Also optional: 2 × 30 s room tone of the demo room → `data/raw/silence/`.

## Environment (this JupyterHub box)

- 16 CPU, 61 GB RAM, **no GPU**. Python 3.13.5. No system `pip`, `jupyter` or `ffmpeg`.
- Venv at `/home/coder/workspaces/free-lab/.venv` (one level above this repo) with everything installed and verified:
  torch 2.14.1+cpu, speechbrain 1.1.1, silero-vad, OpenCV, JupyterLab 4.6.4. ffmpeg comes bundled via `imageio_ffmpeg`.
- Recreate if the box was reset:
  ```bash
  cd /home/coder/workspaces/free-lab
  python3 -m venv .venv
  .venv/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/python -m pip install -r speaker-diarization/requirements.txt jupyterlab ipykernel
  ```
- The notebook sets `HF_HUB_OFFLINE=0` itself because the hub defaults it to 1; the ECAPA model download needs network once.
- Gotcha hit this session: a background shell command whose log redirect pointed at a non-existent folder reported success while doing nothing. Run installs in the foreground and verify with an import.

## Next steps (in order)

1. Get answers to the two open questions.
2. **Convert notebook to 3 classes**:
   - `CLASSES = ["me", "other"]`; `LABELS = CLASSES + ["silence"]`; drop `friend` from `COLORS`, folder creation, inventory, overlay `text` dict.
   - `DRY_RUN = not any files in data/raw/me` (currently requires both `me` and `friend`).
   - Step 1 stand-in: one LibriSpeech speaker for `me` (`STANDIN = {"me": "1272"}`); `other_speakers` excludes it.
   - Step 3 `predict`: `me` if `P(me) >= TAU` else `other` (open-set threshold tuned on val as before). Logistic regression is now binary.
   - Step 4B: EER / ROC-AUC for "is this me?" only (drop the friend subplot).
   - Step 5 synthetic demo `plan`: `silence → me → other → silence → me → other → silence`; `analyse()` probs array becomes 3 columns (me, other, silence); smoothing/threshold logic adjusted accordingly.
   - Summary / markdown tables: remove friend rows.
3. **Place data**: copy `recordings/*.m4a` into `data/raw/me/` with session-aware names, e.g. `s1_clip50.m4a`, so the session split logic (last session → test, second-to-last → val) can be driven by the `s{N}` prefix. Adjust `assign_splits` to group by session prefix instead of one-file-one-session, and pin session 1 to train.
4. Run **Kernel → Restart & Run All** (~15–20 min CPU). Check Step 1 prints `DRY_RUN = False`, Step 2 warning table is empty, listen to the Step 2b samples.
5. Record the **demo video** (1–2 min, same phone, demo room): few s nobody → Onur ~10 s → pause → someone else ~10 s (person or YouTube clip played aloud) → Onur again. Save as `data/demo/demo.mp4`; optionally label `data/demo/demo_labels.csv` (`start,end,label`) for measured frame accuracy / DER.
6. Re-run Step 5 on the real video; collect `models/metrics.json`; put the metrics table + model recommendations into `README.md`; update `Who Is Talking To-Do Plan.md` to the 3-class version.

## Pipeline reminder (unchanged design)

`video → ffmpeg 16 kHz mono → Silero VAD (speech?) → 1.5 s windows / 0.25 s hop → SpeechBrain ECAPA-TDNN 192-d embedding → logistic regression + open-set threshold τ → probability smoothing (±0.5 s) + min-run 0.5 s → OpenCV banner + timeline → annotated .mp4`

Metrics reported: per-class precision/recall/F1, macro-F1, balanced accuracy, confusion matrix (window level, held-out session); ROC-AUC + EER for "is this me?"; VAD miss / false-alarm; demo-timeline frame accuracy, macro-F1 and DER; real-time factor.

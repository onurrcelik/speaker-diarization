# Who is talking? — me / someone else / no one

For every moment of a video, label whether **I** (Onur) am talking, **someone else** is talking, or **no one** is talking.
Training and evaluation live in `notebook.ipynb` (Step 1 data → Step 2 cleaning → Step 3 pretrained models → Step 4 metrics → Step 5 video demo).
The **live webcam demo** is `live_demo.py` + `live/index.html` (see below).

```
video → ffmpeg 16 kHz mono → Silero VAD (speech?) → 1.5 s windows / 0.25 s hop
      → SpeechBrain ECAPA-TDNN 192-d embedding → logistic regression + open-set threshold τ
      → probability smoothing (±0.5 s) + min-run 0.5 s → OpenCV banner + timeline → annotated .mp4
```

## Data

| source | what | amount |
|---|---|---|
| `recordings/New Recording 50–63.m4a` (not in git) | Onur only, phone, 4 sessions: s1 = clips 50–55 and s2 = clips 56–61 (English: read aloud, free answers, short lines), s3 = clip 62 (English phone-call style), s5 = clip 63 (Turkish read aloud) | 14 clips, 12.4 min of speech |
| LibriSpeech dev-clean (downloaded by the notebook) | 39 strangers for the `other` class | 29 min |
| pauses between sentences in session 1 (demo room), harvested by Silero VAD | `silence` / room tone | ~22 s |

Split is **by session**: s1 + s2 → train (s1 was recorded in the demo room and is pinned to train), s3 → val (threshold tuning), s5 → test.
`other` is split by speaker: 29 strangers train, 4 val, 6 test, so the test measures behaviour on voices never heard.
The clip → session mapping (`SESSION_OF` in Step 1) was confirmed by Onur on 2026-10-08. Because the test session is Turkish and training is English, the window-level numbers are a cross-language test.

## Results (run of 2026-10-08, CPU only)

**A. Window level** (1.5 s windows, held-out session s5 vs 6 unseen strangers, 406 windows)

| metric | value |
|---|---|
| macro-F1 | 1.000 |
| balanced accuracy | 1.000 |
| ROC-AUC "is this me?" (cosine to enrolment centroid) | 1.000 |
| EER "is this me?" | 0.7 % |

**C. VAD gate** (speech windows vs room-tone windows): precision 1.000, recall 0.998, miss 0.2 %, false alarm 0.0 %.

**D. Synthetic demo timeline** (28 s stitched from held-out test audio: silence → me → other → silence → me → other → silence, measured every 10 ms)

| metric | value |
|---|---|
| frame accuracy | 93.8 % |
| macro-F1 | 0.936 |
| DER | 8.8 % (miss 0.0 s, false alarm 1.2 s, confusion 0.5 s) |
| real-time factor (whole pipeline, 4 CPU threads) | 1.00 |
| embedding real-time factor alone | 0.22 |

Per class on the demo: me F1 0.951, other F1 0.940, silence F1 0.916. Most of the demo error is at speaker boundaries (the 1.5 s window straddles two labels) and speech bleeding ~0.6 s into each silence gap.

**Read the window-level numbers with care.** The strangers are clean studio read speech and Onur's clips are phone recordings, so part of the separation may come from the recording channel rather than the voice. The honest test is the real demo video with someone else speaking into the same phone, which does not exist yet.

## Pretrained model recommendations

| role | used | alternatives |
|---|---|---|
| VAD | **Silero VAD v5** (MIT, 2 MB, CPU real-time, no token) | pyannote `segmentation-3.0` (adds overlap detection, needs HF token); WebRTC VAD (tiny, worse in noise) |
| speaker embedding | **SpeechBrain ECAPA-TDNN** `spkrec-ecapa-voxceleb` (Apache-2.0, 20 MB, ~0.8 % EER on VoxCeleb1) | WeSpeaker ResNet34/293 (slightly better EER, ONNX); `microsoft/wavlm-base-plus-sv` (good to fine-tune); NeMo TitaNet-L |
| classifier on top | logistic regression on 192-d embeddings + open-set threshold τ tuned on val | cosine-to-centroid threshold (what the EER measures); PLDA |
| full diarization baseline | — | `pyannote/speaker-diarization-3.1`, then map clusters to "me" via enrolment |

## Live demo (webcam + microphone in the browser)

```bash
cd /home/coder/workspaces/free-lab/speaker-diarization
../.venv/bin/python live_demo.py --port 8765
# open https://student.exposureai.org/lab/proxy/8765/   (on a laptop: http://localhost:8765/)
```

Press **Start webcam + mic**, allow camera and microphone. The browser shows your webcam locally and streams 16 kHz audio to the server in 250 ms chunks; the server runs the same pipeline as the notebook on the last 1.5 s (Silero VAD → ECAPA → logistic regression → τ → smoothing) and sends back a label about 4× per second. The page draws the banner (ME talking / SOMEONE ELSE talking / no one talking, with confidence) and a 60 s timeline strip. Expect ~0.5–1 s of delay at speaker changes; compute is ~0.1 s per decision on this CPU.

Loopback test (held-out session of Onur, a LibriSpeech stranger and near-silence streamed at real time): 68/74 decisions `me`, 36/41 `other`, 49/53 `silence`; the errors sit at the transitions.

**Microphone mismatch.** The model has only heard Onur through the phone. If the laptop microphone gives weak results, press **Record my voice as training data** on the page, talk for 2–3 minutes (read aloud, then free talk), press Stop: the audio is saved to `data/raw/me/s6_live_<timestamp>.wav` as a new session. Re-run the notebook (it copies nothing over; the new file is simply picked up), then restart `live_demo.py` to load the new classifier.

Server flags: `--hop` (seconds between decisions), `--smooth` (decisions averaged), `--min-run` (consecutive agreeing decisions before switching label), `--speech-frac-min`, `--min-level-db`, `--threads`.

## Running the notebook

```bash
cd /home/coder/workspaces/free-lab
.venv/bin/jupyter lab            # open speaker-diarization/notebook.ipynb, Kernel → Restart & Run All (~20 min on 16 CPU)
# or headless:
cd speaker-diarization && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

Outputs (git-ignored): `data/clean/`, `models/who_is_talking.joblib` (also loaded by `live_demo.py`), `models/metrics.json`, `data/demo/*_annotated.mp4`.

## Still to do

1. Try the live demo with the laptop webcam; if `me` is weak through that microphone, record a laptop session from the page and re-run the notebook.
2. Optional: a recorded, labelled demo video (`data/demo/demo.mp4` + `demo_labels.csv`) gives measured frame accuracy / DER on real footage; the notebook picks it up automatically.

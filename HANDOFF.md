# Session handoff — "Who is talking?" (me / someone else / no one)

Last updated 2026-10-08 ~09:05 UTC, mid-session, by Claude at Onur's request ("update handoff.md with your full context").
Read this first next time. It supersedes the 2026-10-07 handoff; the plan file `Who Is Talking To-Do Plan.md` is the user-facing checklist.

## Task (original brief)

> identify if I am talking or if someone else is talking
> steps: find/get data (human in the loop) → clean data → recommend pretrained models →
> write down performance metrics → video demo showing *I am talking / no one / someone else*.
> Use a notebook.

## Decisions so far

| date | decision |
|---|---|
| 2026-10-07 | `friend` class dropped. Labels: `me` (Onur), `other` (anyone else), `silence` (nobody). |
| 2026-10-08 | **The demo is live from Onur's laptop webcam, not a filmed video.** Onur: "i dont wanna film i want it to be live from my webcam". The notebook's Step 5 still works on a video file (synthetic demo built from held-out audio) but is not the deliverable. |
| 2026-10-08 | Clip → session mapping confirmed by Onur (table below). Session 1 was recorded in the demo room; its inter-sentence pauses are used as room tone (Onur: "session 1 recordings were recorded in demo room so you can use them for room tone"). |
| 2026-10-08 | Laptop-mic sessions recorded from the live page are session `s6`, pinned to train. |

## Repo contents

| path | state |
|---|---|
| `notebook.ipynb` | 31 cells, 3-class pipeline, **executed on real data (run 3, 2026-10-08 08:30)**, outputs saved in the file. Step 1 data, 2 clean, 3 models, 4 metrics, 5 video demo (synthetic). |
| `live_demo.py` | Tornado server: serves `live/index.html`, WebSocket `/ws`, runs VAD → ECAPA → clf every 0.25 s, instant adaptation from recordings, debug dump. |
| `live/index.html` | Browser page: webcam + mic via getUserMedia, AudioWorklet → 16 kHz float32 chunks → WebSocket, banner + 60 s timeline, Record button. |
| `README.md` | Metrics tables, model recommendations, how to run notebook + live demo. |
| `recordings/New Recording 50–63.m4a` | Onur's 14 phone clips (git-ignored, exist on this box). |
| `requirements.txt`, `.gitignore` | unchanged; `data/`, `models/`, `recordings/` are git-ignored. |

Git: last commit `5379eea embedded` (Onur, 08:12) contains the 3-class notebook + first live demo. **Uncommitted** since: README/HANDOFF/plan edits, notebook (embeddings export, s6 pinning, room-tone harvest), `live_demo.py` (latency, adaptation, dump, info), `live/index.html` (adaptation UI, info message, **the buffer-copy fix**). Onur has not asked for a commit.

## Data

| session | clips | content | split |
|---|---|---|---|
| s1 | 50–55 | 50 Part A read aloud; 51–54 Part B free answers (yesterday / the room / disliked food / travel); 55 Part C short lines. **Demo room.** | train (pinned) |
| s2 | 56–61 | 56 read aloud "A Strange Morning at the Station"; 57–60 free answers (late story / movie / recipe / phone annoyance); 61 short lines | train |
| s3 | 62 | phone-call style read aloud "Planning the Weekend", 3:29 | val (τ tuning) |
| s5 | 63 | **Turkish** read aloud "Pazar Kahvaltısı", 2:48 | test |
| s6 | `data/raw/me/s6_live_*.wav` | laptop-mic recordings from the live page (none yet) | train (pinned) |

All 14 clips: ~12.4 min speech, no clipping, −17…−22 dBFS. `other` = LibriSpeech dev-clean, 39 speakers (29 train / 4 val / 6 test), 45 s each. Room tone: ~22 s harvested from s1 pauses (`ROOM_TONE_FROM = "s1"`, gaps ≥ 0.3 s, 0.1 s margin), split in 3 pseudo clips; synthetic noise only if that yields < 15 s. Step 1 copies `recordings/*.m4a` → `data/raw/me/s<N>_clipNN.m4a` via `SESSION_OF`.

## Results (run 3)

| metric | value | note |
|---|---|---|
| window macro-F1 / balanced acc (test = s5 vs 6 unseen strangers, 406 windows) | 1.000 / 1.000 | English-trained → Turkish-tested; caveat: phone vs studio channel helps |
| EER / AUC "is this me?" (cosine to centroid) | 0.7 % / 1.000 | |
| VAD precision / recall / miss / FA | 1.000 / 0.998 / 0.002 / 0.000 | optimistic (pre-trimmed) |
| synthetic demo frame acc / macro-F1 / DER | 92.9 % / 0.926 / 10.0 % | 28 s stitched from held-out audio |
| RTF whole pipeline / embedding only | 1.01 / 0.22 | 4 threads |
| τ | 0.40 | |

Reference scores through the live scoring code: held-out phone Onur → cos 0.54 mean (max 0.67), P(me) 0.96; LibriSpeech stranger → cos −0.02, P(me) 0.02.

## Live demo — architecture

```
browser: getUserMedia(video, audio{EC/NS/AGC off}) → <video> shown locally
         AudioContext(16 kHz) → AudioWorklet "chunker" → 4000-sample float32 chunks → WebSocket (binary)
server:  ring buffer last 1.5 s → every 0.25 s: Silero VAD speech_frac (≥0.3 and level ≥ −55 dBFS) → RMS-normalise −23 dBFS
         → ECAPA 192-d → clf.predict_proba → P(me) → mean of last 2 → ≥ τ "me" else "other" → JSON back
browser: banner (ME / SOMEONE ELSE / no one + %), 60 s timeline, stats line (P(me), τ, cos, speech %, level, ms)
```

- Reachable at **https://student.exposureai.org/lab/proxy/8765/** while the server runs (`VSCODE_PROXY_URI` = `https://student.exposureai.org/lab/proxy/{{port}}/`, code-server strips the prefix; WebSockets pass; page uses relative `ws` URL). Browser: Chrome 154 on macOS; AudioContext 16 kHz honoured, mic track 48 kHz.
- Start: `cd speaker-diarization && PYTHONUNBUFFERED=1 nohup ../.venv/bin/python live_demo.py --port 8765 --threads 6 --dump <dir> > server.log 2>&1 &`. Process is lost on box migration (happened once today); restart it. Kill with `pgrep -f "^/home/coder/workspaces/free-lab/.venv/bin/python live_demo.py"` (a bare `pkill -f live_demo.py` kills the calling shell too).
- Flags: `--hop 0.25 --smooth 2 --min-run 1 --speech-frac-min 0.3 --min-level-db -55 --threads --no-adapt --quiet --dump DIR`.
- Compute ≈ 125 ms per decision on this CPU; the server logs every speech decision (`t= … P(me)= cos= speech= level=`).
- **Instant adaptation**: Record button → `data/raw/me/<session>_live_<ts>.wav`; server embeds VAD-trimmed 1.5 s windows (hop 1 s), appends them as `me` to `models/embeddings.npz` (exported by notebook cell 18: X, y, split, session), refits LogisticRegression (balanced), re-tunes τ on val (incl. last 20 % of the new windows), hot-swaps, saves `who_is_talking.joblib` with `adapted_with`. Refuses if < 10 windows of speech. On start-up it adapts from any `*_live_*.wav` not yet in the cache. Tested with 40 s of held-out phone audio: 39 windows, val F1 0.991, model then restored.

## Live demo — what happened and the bug (IMPORTANT)

1. Onur's first test: 2–3 s lag and "someone else" while he talked (English and Turkish). Lag fixed by `--smooth 5→2`, `--min-run 2→1` (≈1 s now).
2. Second test after fix, with per-decision logging: 31 decisions, **P(me) ≈ 0.06, cos ≈ 0.0**, speech_frac 0.3–0.5 while he talked continuously, level −36…−50 dBFS. cos ≈ 0 is "unrelated to Onur", not mic mismatch (mismatch would give ~0.3).
3. Added `--dump` and a client `info` message; Onur talked 20–40 s again. Diagnosis of the dumped 60 s (`scratchpad/dump/talk.wav`, scratchpad `diag.py`): 40 s at −30 dBFS, **Silero finds 3 s of speech**, speech level = non-speech level, **median f0 = 125 Hz = 16000 / 128**.
4. **Root cause: `live/index.html` AudioWorklet stored a reference to the 128-sample input block (`out = ch`) instead of a copy; the browser reuses that buffer, so each 4000-sample chunk was one 128-sample block repeated ~31×.** A 125 Hz buzz with the spectral colour of the voice reached the server. Fixed 09:04: `out = Float32Array.from(ch)`. The page is re-read on every request, no server restart needed. **Not yet re-tested by Onur** — this is the next step.
5. Dump files from the buggy page are useless as training data. No `*_live_*.wav` was recorded, so nothing bad entered the model (adaptation would have refused anyway: < 10 speech windows).

## Next steps (in order)

1. Onur reloads the page, presses Start, talks 20 s. Expect: speech_frac 0.8–1.0 while talking, cos > 0.3, banner "ME". Check the server log.
2. If "ME" is unstable through the laptop mic (P(me) around τ, cos 0.2–0.4): press Record, talk 1–2 min, Stop → instant adaptation; talk again.
3. Have someone else talk (or play a YouTube voice) to confirm "SOMEONE ELSE", and pause to confirm "no one".
4. Remove `--dump` from the server command once the live path is confirmed (it saves audio to the scratchpad).
5. Re-run the notebook once an `s6_live_*.wav` exists so the notebook metrics include the laptop mic (s6 is pinned to train, so the test stays s5). Refresh README metrics.
6. Commit when Onur asks (he committed `5379eea` himself).

## Environment

- code-server on AWS (kernel 7.0.0-1014-aws after today's migration), 16 CPU, 61 GB RAM, **no GPU**, no system pip/ffmpeg/jupyter. Python 3.13.5.
- Venv `/home/coder/workspaces/free-lab/.venv` (torch 2.14.1+cpu, speechbrain 1.1.1, silero-vad, OpenCV, scikit-learn, librosa, JupyterLab 4.6.4, nbconvert 7.17.1, tornado). ffmpeg via `imageio_ffmpeg`. Survived the migration; `data/`, `models/`, `recordings/` too.
- Headless notebook run: `cd speaker-diarization && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebook.ipynb` (~20 min; embedding 2656 windows ≈ 15 min). **Run from inside the repo**: `ROOT = Path.cwd()`.
- Notebook sets `HF_HUB_OFFLINE=0` (hub defaults to 1 here); ECAPA weights cached in `models/ecapa`.
- Onur's style: wants plain, short status updates; got confused by terse progress-check messages during long background runs. He committed the repo himself mid-session.

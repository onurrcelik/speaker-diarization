"""Live "who is talking?" demo: webcam + microphone in the browser, model on this machine.

    ../.venv/bin/python live_demo.py --port 8765
    then open  https://student.exposureai.org/lab/proxy/8765/   (or http://localhost:8765/ when run locally)

The browser (live/index.html) captures the webcam and microphone, shows the video locally (no lag) and streams
16 kHz mono float32 audio over a WebSocket in 250 ms chunks. This server keeps the last 1.5 s per connection and,
every HOP seconds, runs the same pipeline as the notebook: Silero VAD -> ECAPA embedding -> logistic regression +
open-set threshold tau -> light smoothing, and sends back {label, conf, ...}. The page draws the banner and a timeline.

"Record my voice" on the page saves the incoming audio to data/raw/me/<session>_live_<timestamp>.wav. If the notebook
has exported its training embeddings (models/embeddings.npz), the server immediately re-trains the classifier with
that recording added as "me" (laptop microphone), re-tunes tau on validation data, hot-swaps the model and saves it.
Without the export the file is just saved for the next notebook run. Microphone mismatch (phone vs laptop) is the
#1 cause of errors, so do this once per microphone.
"""
import traceback
import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")
import argparse, asyncio, json, time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import torch
import tornado.ioloop, tornado.web, tornado.websocket
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
MODELS, RAW_ME, PAGE = ROOT / "models", ROOT / "data" / "raw" / "me", ROOT / "live" / "index.html"
BUNDLE, CACHE = MODELS / "who_is_talking.joblib", MODELS / "embeddings.npz"

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--threads", type=int, default=8, help="torch CPU threads")
parser.add_argument("--hop", type=float, default=0.25, help="seconds between decisions (actual hop = max(hop, compute time))")
parser.add_argument("--smooth", type=int, default=2, help="moving average over the last N speech decisions")
parser.add_argument("--min-run", type=int, default=1, help="consecutive agreeing decisions needed to switch label")
parser.add_argument("--speech-frac-min", type=float, default=0.3)
parser.add_argument("--min-level-db", type=float, default=-55.0, help="windows quieter than this (dBFS RMS) are never speech")
parser.add_argument("--no-adapt", action="store_true", help="never re-train from live recordings")
parser.add_argument("--quiet", action="store_true", help="do not log every decision")
parser.add_argument("--dump", default="", help="debug: directory to save each connection's incoming audio (30 s files)")
args = parser.parse_args()
torch.set_num_threads(args.threads)

# ---------------------------------------------------------------- models (identical to the notebook)
from silero_vad import load_silero_vad, get_speech_timestamps
from speechbrain.inference.speaker import EncoderClassifier

vad = load_silero_vad()
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODELS / "ecapa"), run_opts={"device": "cpu"})
encoder.eval()

import threading
MODEL_LOCK = threading.Lock()   # Silero VAD (stateful JIT module) and ECAPA must not be called from two threads at once:
                                # adapt() running next to infer() crashed the process without a Python traceback (2026-10-08).
M = {}   # the live model: clf, tau, i_me, centroid (hot-swappable)

def load_bundle():
    b = joblib.load(BUNDLE)
    M.update(clf=b["clf"], tau=float(b["tau"]), centroid=b["centroids"]["me"], win_s=float(b["win_s"]), sr=int(b["sr"]),
             i_me=list(b["clf"].classes_).index("me"), bundle=b)
    print(f"model loaded: tau={M['tau']:.2f} win={M['win_s']}s classes={list(M['clf'].classes_)}"
          + (f"  (adapted with: {', '.join(b['adapted_with'])})" if b.get("adapted_with") else ""))

load_bundle()
SR, WIN_S = M["sr"], M["win_s"]; WIN = int(WIN_S * SR)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)

def normalise(y, target_db=-23.0):
    return np.clip(y * 10 ** ((target_db - rms_db(y)) / 20), -1, 1)

def speech_segments(y, threshold=0.5):
    ts = get_speech_timestamps(torch.from_numpy(y), vad, sampling_rate=SR, threshold=threshold, min_silence_duration_ms=200, speech_pad_ms=60)
    return [(t["start"], t["end"]) for t in ts]

@torch.no_grad()
def embed(batch, bs=32):
    out = []
    for i in range(0, len(batch), bs):
        e = encoder.encode_batch(torch.from_numpy(np.stack(batch[i:i + bs]))).squeeze(1)
        out.append(torch.nn.functional.normalize(e, dim=1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 192), np.float32)

@torch.no_grad()
def infer(window):
    """window: float32[WIN] -> (speech_frac, p_me or None, cosine to phone 'me' centroid or None)"""
    window = window - window.mean()
    with MODEL_LOCK:
        segs = speech_segments(window)
        sfrac = sum(e - s for s, e in segs) / len(window)
        if sfrac < args.speech_frac_min or rms_db(window) < args.min_level_db:
            return sfrac, None, None
        if rms_db(window) > -60:
            window = normalise(window)
        e = embed([window])
        return sfrac, float(M["clf"].predict_proba(e)[0, M["i_me"]]), float(e[0] @ M["centroid"])


def file_embeddings(path):
    """a recording of ME -> (N, 192) embeddings of its VAD-trimmed, normalised 1.5 s windows (hop 1 s), like the notebook"""
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    assert sr == SR, f"{path}: {sr} Hz, expected {SR}"
    y = y - y.mean()
    with MODEL_LOCK:                                   # pauses live inference for a second or two
        segs = speech_segments(y)
        speech = np.concatenate([y[s:e] for s, e in segs]) if segs else np.zeros(0, np.float32)
        if len(speech) < WIN: return np.zeros((0, 192), np.float32), len(speech) / SR
        speech = normalise(speech)
        ws = [speech[i:i + WIN] for i in range(0, len(speech) - WIN + 1, SR) if rms_db(speech[i:i + WIN]) > -45]
        return embed(ws), len(speech) / SR


def adapt(paths):
    """re-train clf with these ME recordings (laptop mic) added to the notebook's cached training embeddings"""
    if args.no_adapt or not CACHE.exists():
        return {"ok": False, "why": "no models/embeddings.npz yet (run the notebook once) - recording saved for the next run"}
    d = np.load(CACHE, allow_pickle=True)
    X, y, split, session = d["X"], d["y"].astype(str), d["split"].astype(str), d["session"].astype(str)
    Xn, names, speech_s = [], [], 0.0
    for p in paths:
        e, s = file_embeddings(p); speech_s += s
        if len(e): Xn.append(e); names += [f"live__{Path(p).stem}"] * len(e)
    if not Xn or sum(map(len, Xn)) < 10:
        return {"ok": False, "why": f"only {speech_s:.0f}s of speech found - record at least 30 s of talking"}
    Xn = np.concatenate(Xn); names = np.array(names)
    old_p = float(np.mean(M["clf"].predict_proba(Xn)[:, M["i_me"]]))
    k = max(1, int(0.8 * len(Xn)))                                   # first 80% train, last 20% val (time split)
    nsplit = np.array(["train"] * k + ["val"] * (len(Xn) - k))
    X2, y2, s2 = np.vstack([X, Xn]), np.concatenate([y, ["me"] * len(Xn)]), np.concatenate([split, nsplit])
    tr, va = s2 == "train", s2 == "val"
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X2[tr], y2[tr])
    i_me = list(clf.classes_).index("me")
    taus = np.linspace(0.3, 0.95, 27)
    f1s = [f1_score(y2[va], np.where(clf.predict_proba(X2[va])[:, i_me] >= t, "me", "other"), average="macro") for t in taus]
    tau = float(taus[int(np.argmax(f1s))])
    new_p = float(np.mean(clf.predict_proba(Xn)[:, i_me]))
    # hot-swap + persist (notebook outputs are kept; the bundle records what it was adapted with)
    M.update(clf=clf, tau=tau, i_me=i_me)
    b = dict(M["bundle"]); b.update(clf=clf, tau=tau, adapted_with=sorted(set(b.get("adapted_with", [])) | set(Path(p).name for p in paths)))
    b.setdefault("centroids", {})["me_live"] = (lambda v: v / np.linalg.norm(v))(Xn.mean(0))
    joblib.dump(b, BUNDLE); M["bundle"] = b
    np.savez(CACHE, X=X2, y=y2, split=s2, session=np.concatenate([session, names]))
    return {"ok": True, "windows": int(len(Xn)), "speech_s": round(speech_s, 1), "tau": tau, "val_f1": round(max(f1s), 3),
            "p_me_before": round(old_p, 3), "p_me_after": round(new_p, 3)}


# adapt at start-up from live recordings the cache has not seen yet
if CACHE.exists() and not args.no_adapt:
    seen = set(np.load(CACHE, allow_pickle=True)["session"].astype(str))
    pending = [p for p in sorted(RAW_ME.glob("*_live_*.wav")) if f"live__{p.stem}" not in seen]
    if pending:
        print("adapting to", [p.name for p in pending]); print(adapt([str(p) for p in pending]))

t0 = time.time(); infer(np.random.default_rng(0).normal(0, 0.01, WIN).astype(np.float32)); print(f"warm-up inference {time.time() - t0:.2f}s")


# ---------------------------------------------------------------- per-connection state
class Session:
    def __init__(self):
        self.buf = np.zeros(0, np.float32)          # most recent audio (kept to WIN samples)
        self.probs = deque(maxlen=args.smooth)       # recent P(me) for speech decisions
        self.label, self.cand, self.cand_n = "silence", "silence", 0
        self.recording = None                        # list of chunks while recording, else None
        self.dump, self.dump_n = [], 0
        self.rec_session = "s6"

    def push(self, chunk):
        self.buf = np.concatenate([self.buf, chunk])[-WIN:]
        if self.recording is not None:
            self.recording.append(chunk)
        if args.dump:
            self.dump.append(chunk)
            if sum(map(len, self.dump)) >= 30 * SR: self.flush_dump()

    def flush_dump(self):
        if not self.dump: return
        Path(args.dump).mkdir(parents=True, exist_ok=True)
        out = Path(args.dump) / f"dump_{datetime.now():%Y%m%d_%H%M%S}_{self.dump_n}.wav"
        sf.write(out, np.concatenate(self.dump), SR); print("dumped", out); self.dump, self.dump_n = [], self.dump_n + 1

    def decide(self, sfrac, p_me):
        if p_me is None:
            self.probs.clear(); cand, conf = "silence", 1 - sfrac
        else:
            self.probs.append(p_me)
            p = float(np.mean(self.probs))
            cand, conf = ("me" if p >= M["tau"] else "other"), max(p, 1 - p)
        if cand == self.label:
            self.cand_n = 0
        elif cand == self.cand:
            self.cand_n += 1
            if self.cand_n >= args.min_run - 1: self.label = cand; self.cand_n = 0
        else:
            self.cand, self.cand_n = cand, 0
            if args.min_run <= 1: self.label = cand
        return self.label, conf


class WS(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):            # served behind the code-server proxy -> origin differs from host
        return True

    def open(self):
        self.s = Session(); self.busy = False; self.t_open = time.time()
        self.timer = tornado.ioloop.PeriodicCallback(self.tick, args.hop * 1000); self.timer.start()
        print("client connected"); self.write_message({"type": "hello", "tau": M["tau"], "adapt": CACHE.exists() and not args.no_adapt})

    def on_message(self, msg):
        if isinstance(msg, bytes):
            self.s.push(np.frombuffer(msg, dtype=np.float32).copy()); return
        cmd = json.loads(msg)
        if cmd.get("cmd") == "info":
            print("client info:", cmd); return
        if cmd.get("cmd") == "record":
            if cmd.get("on"):
                self.s.rec_session = str(cmd.get("session", "s6")).strip() or "s6"
                self.s.recording = []
                self.write_message({"type": "record", "on": True})
            elif self.s.recording is not None:
                y = np.concatenate(self.s.recording) if self.s.recording else np.zeros(0, np.float32)
                self.s.recording = None
                RAW_ME.mkdir(parents=True, exist_ok=True)
                out = RAW_ME / f"{self.s.rec_session}_live_{datetime.now():%Y%m%d_%H%M%S}.wav"
                sf.write(out, y, SR)
                print("saved", out, f"{len(y) / SR:.1f}s")
                self.write_message({"type": "record", "on": False, "path": str(out.relative_to(ROOT)), "seconds": round(len(y) / SR, 1)})
                asyncio.ensure_future(self.do_adapt(out))

    async def do_adapt(self, path):
        self.write_message({"type": "adapt", "state": "running"})
        try:
            res = await asyncio.get_event_loop().run_in_executor(None, adapt, [str(path)])
        except Exception as e:                        # otherwise the error is swallowed by ensure_future
            traceback.print_exc(); res = {"ok": False, "why": f"{type(e).__name__}: {e}"}
        print("adapt:", res)
        try: self.write_message({"type": "adapt", "state": "done", **res})
        except tornado.websocket.WebSocketClosedError: pass

    async def tick(self):
        if self.busy or len(self.s.buf) < WIN:
            return
        self.busy = True
        try:
            t0 = time.time()
            window = self.s.buf.copy()
            sfrac, p_me, cos = await asyncio.get_event_loop().run_in_executor(None, infer, window)
            label, conf = self.s.decide(sfrac, p_me)
            lvl = round(float(rms_db(window)), 1); ms = int(1000 * (time.time() - t0))
            if not args.quiet and p_me is not None:
                print(f"t={time.time() - self.t_open:6.1f}s {label:8s} P(me)={p_me:.2f} cos={cos:.2f} speech={sfrac:.2f} level={lvl} dBFS {ms} ms")
            self.write_message({"type": "pred", "t": round(time.time() - self.t_open, 2), "label": label, "conf": round(conf, 3),
                                "p_me": None if p_me is None else round(p_me, 3), "cos": None if cos is None else round(cos, 2),
                                "speech_frac": round(sfrac, 2), "ms": ms, "level_db": lvl, "tau": M["tau"]})
        except tornado.websocket.WebSocketClosedError:
            pass
        finally:
            self.busy = False

    def on_close(self):
        self.timer.stop(); self.s.flush_dump(); print("client disconnected")


class Page(tornado.web.RequestHandler):
    def get(self, *_):
        self.set_header("Cache-Control", "no-store")
        self.write(PAGE.read_text())


app = tornado.web.Application([(r"/ws", WS), (r"/.*", Page)])
app.listen(args.port, address="0.0.0.0", max_buffer_size=64 * 1024 * 1024)
print(f"serving on http://localhost:{args.port}/   (code-server: {os.environ.get('VSCODE_PROXY_URI', 'https://<host>/proxy/{{port}}/').replace('{{port}}', str(args.port))})")
tornado.ioloop.IOLoop.current().start()

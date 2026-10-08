"""Live "who is talking?" demo: webcam + microphone in the browser, model on this machine.

    ../.venv/bin/python live_demo.py --port 8765
    then open  https://student.exposureai.org/lab/proxy/8765/   (or http://localhost:8765/ when run locally)

The browser (live/index.html) captures the webcam and microphone, shows the video locally (no lag) and streams
16 kHz mono float32 audio over a WebSocket in 250 ms chunks. This server keeps the last 1.5 s per connection and,
every HOP seconds, runs the same pipeline as the notebook: Silero VAD -> ECAPA embedding -> logistic regression +
open-set threshold tau -> smoothing, and sends back {label, conf, ...}. The page draws the banner and a timeline.

The page also has a "record my voice" button: while it is on, the incoming audio is saved to
data/raw/me/<session>_live_<timestamp>.wav so the laptop microphone can be added as a training session
(re-run the notebook afterwards). Microphone mismatch (phone vs laptop) is the #1 cause of errors.
"""
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

ROOT = Path(__file__).resolve().parent
MODELS, RAW_ME, PAGE = ROOT / "models", ROOT / "data" / "raw" / "me", ROOT / "live" / "index.html"

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--threads", type=int, default=8, help="torch CPU threads")
parser.add_argument("--hop", type=float, default=0.25, help="seconds between decisions (actual hop = max(hop, compute time))")
parser.add_argument("--smooth", type=int, default=5, help="moving average over the last N decisions (5 x 0.25 s ~ +-0.5 s)")
parser.add_argument("--min-run", type=int, default=2, help="consecutive agreeing decisions needed to switch label (~0.5 s)")
parser.add_argument("--speech-frac-min", type=float, default=0.3)
parser.add_argument("--min-level-db", type=float, default=-55.0, help="windows quieter than this (dBFS RMS) are never speech")
args = parser.parse_args()
torch.set_num_threads(args.threads)

# ---------------------------------------------------------------- models (identical to the notebook)
from silero_vad import load_silero_vad, get_speech_timestamps
from speechbrain.inference.speaker import EncoderClassifier

bundle = joblib.load(MODELS / "who_is_talking.joblib")
clf, TAU, WIN_S, SR = bundle["clf"], float(bundle["tau"]), float(bundle["win_s"]), int(bundle["sr"])
I_ME = list(clf.classes_).index("me")
vad = load_silero_vad()
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODELS / "ecapa"), run_opts={"device": "cpu"})
encoder.eval()
WIN = int(WIN_S * SR)
print(f"model loaded: tau={TAU:.2f} win={WIN_S}s classes={list(clf.classes_)}")


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


@torch.no_grad()
def infer(window):
    """window: float32[WIN] -> (speech_frac, p_me or None)"""
    window = window - window.mean()
    ts = get_speech_timestamps(torch.from_numpy(window), vad, sampling_rate=SR, threshold=0.5, min_silence_duration_ms=200, speech_pad_ms=60)
    sfrac = sum(t["end"] - t["start"] for t in ts) / len(window)
    if sfrac < args.speech_frac_min or rms_db(window) < args.min_level_db:
        return sfrac, None
    if rms_db(window) > -60:
        window = np.clip(window * 10 ** ((-23.0 - rms_db(window)) / 20), -1, 1)
    e = encoder.encode_batch(torch.from_numpy(window[None])).squeeze(1)
    e = torch.nn.functional.normalize(e, dim=1).numpy()
    return sfrac, float(clf.predict_proba(e)[0, I_ME])


# warm-up so the first live decision is not slow
t0 = time.time(); infer(np.random.default_rng(0).normal(0, 0.01, WIN).astype(np.float32)); print(f"warm-up inference {time.time() - t0:.2f}s")


# ---------------------------------------------------------------- per-connection state
class Session:
    def __init__(self):
        self.buf = np.zeros(0, np.float32)          # most recent audio (kept to WIN samples)
        self.received = 0                            # total samples received
        self.probs = deque(maxlen=args.smooth)       # recent P(me) for speech decisions
        self.label, self.cand, self.cand_n = "silence", "silence", 0
        self.recording = None                        # list of chunks while recording, else None
        self.rec_session = "s6"

    def push(self, chunk):
        self.received += len(chunk)
        self.buf = np.concatenate([self.buf, chunk])[-WIN:]
        if self.recording is not None:
            self.recording.append(chunk)

    def decide(self, sfrac, p_me):
        if p_me is None:
            self.probs.clear(); cand, conf = "silence", 1 - sfrac
        else:
            self.probs.append(p_me)
            p = float(np.mean(self.probs))
            cand, conf = ("me" if p >= TAU else "other"), max(p, 1 - p)
        # hysteresis: switch only after min_run consecutive agreeing decisions
        if cand == self.label:
            self.cand_n = 0
        elif cand == self.cand:
            self.cand_n += 1
            if self.cand_n >= args.min_run - 1: self.label = cand; self.cand_n = 0
        else:
            self.cand, self.cand_n = cand, 0
        return self.label, conf


class WS(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):            # served behind the code-server proxy -> origin differs from host
        return True

    def open(self):
        self.s = Session(); self.busy = False; self.t_open = time.time()
        self.timer = tornado.ioloop.PeriodicCallback(self.tick, args.hop * 1000); self.timer.start()
        print("client connected")

    def on_message(self, msg):
        if isinstance(msg, bytes):
            self.s.push(np.frombuffer(msg, dtype=np.float32).copy())
            return
        cmd = json.loads(msg)
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
                self.write_message({"type": "record", "on": False, "path": str(out.relative_to(ROOT)), "seconds": round(len(y) / SR, 1)})
                print("saved", out, f"{len(y) / SR:.1f}s")

    async def tick(self):
        if self.busy or len(self.s.buf) < WIN:
            return
        self.busy = True
        try:
            t0 = time.time()
            window = self.s.buf.copy()
            sfrac, p_me = await asyncio.get_event_loop().run_in_executor(None, infer, window)
            label, conf = self.s.decide(sfrac, p_me)
            self.write_message({"type": "pred", "t": round(time.time() - self.t_open, 2), "label": label, "conf": round(conf, 3),
                                "p_me": None if p_me is None else round(p_me, 3), "speech_frac": round(sfrac, 2),
                                "ms": int(1000 * (time.time() - t0)), "level_db": round(float(rms_db(window)), 1)})
        except tornado.websocket.WebSocketClosedError:
            pass
        finally:
            self.busy = False

    def on_close(self):
        self.timer.stop(); print("client disconnected")


class Page(tornado.web.RequestHandler):
    def get(self, *_):
        self.set_header("Cache-Control", "no-store")
        self.write(PAGE.read_text())


app = tornado.web.Application([(r"/ws", WS), (r"/.*", Page)])
app.listen(args.port, address="0.0.0.0", max_buffer_size=64 * 1024 * 1024)
print(f"serving on http://localhost:{args.port}/   (code-server: {os.environ.get('VSCODE_PROXY_URI', 'https://<host>/proxy/{{port}}/').replace('{{port}}', str(args.port))})")
tornado.ioloop.IOLoop.current().start()

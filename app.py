import io
import os
import socket
import sys
import threading
import webbrowser

import numpy as np
import scipy.io.wavfile as wav
from flask import Flask, jsonify, render_template, request, send_file
from scipy.signal import fftconvolve

from colors import MATERIAL_COLORS
from engine import AcousticRayTracer, build_impulse_response
from presets import PRESETS


def _resource_path(relative: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


IS_FROZEN = getattr(sys, "frozen", False)

app = Flask(__name__, template_folder=_resource_path("templates"))


def _rt60(ir: np.ndarray, sr: int) -> int:
    energy = ir ** 2
    edc = np.flip(np.cumsum(np.flip(energy)))
    if edc[0] <= 0:
        return 0
    edc_db = 10 * np.log10(edc / edc[0] + 1e-12)
    above = np.where(edc_db > -60)[0]
    return int(above[-1] / sr * 1000) if len(above) else 0


def _run_trace(preset_name: str, n_rays: int = 2000, max_time: float = 2.5):
    room, source, receiver, desc = PRESETS[preset_name]()
    tracer = AcousticRayTracer(
        room, source, receiver,
        n_rays=n_rays, max_bounces=50, max_time=max_time,
        receiver_radius=0.3,
    )
    hits = tracer.trace()
    return room, source, receiver, desc, hits


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/presets")
def list_presets():
    result = {}
    for key, fn in PRESETS.items():
        room, src, recv, desc = fn()
        result[key] = {
            "description": desc,
            "source": list(src),
            "receiver": list(recv),
            "walls": [
                {
                    "x0": w.x0, "y0": w.y0, "x1": w.x1, "y1": w.y1,
                    "material": w.material.name,
                    "absorption": w.material.absorption,
                    "color": MATERIAL_COLORS.get(w.material.name, "#888888"),
                }
                for w in room.walls
            ],
        }
    return jsonify(result)


@app.route("/api/trace/<preset_name>")
def trace(preset_name: str):
    if preset_name not in PRESETS:
        return jsonify({"error": "Unknown preset"}), 404

    n_rays   = int(request.args.get("rays", 2000))
    max_time = float(request.args.get("max_time", 2.5))

    room, source, receiver, desc, hits = _run_trace(preset_name, n_rays, max_time)
    ir = build_impulse_response(hits, sample_rate=44100, max_time=max_time)
    ir_mono = ir.mean(axis=1)

    direct_ms = round(min((h.time for h in hits), default=0) * 1000, 1)

    n_out = min(2000, len(ir_mono))
    block = max(1, len(ir_mono) // n_out)
    ir_down = [float(np.max(np.abs(ir_mono[i * block:(i + 1) * block]))) for i in range(n_out)]

    return jsonify({
        "n_hits":         len(hits),
        "ir":             ir_down,
        "n_buckets":      n_out,
        "ir_full_len":    len(ir_mono),
        "ir_sample_rate": 44100,
        "max_time":       max_time,
        "rt60_ms":        _rt60(ir_mono, 44100),
        "direct_ms":      direct_ms,
        "description":    desc,
    })


@app.route("/api/process", methods=["POST"])
def process():
    preset_name = request.form.get("preset", "cathedral")
    if preset_name not in PRESETS:
        return jsonify({"error": "Unknown preset"}), 400

    audio_bytes = request.files["audio"].read()
    sr, dry = wav.read(io.BytesIO(audio_bytes))
    dry = dry.astype(np.float64)
    if dry.ndim > 1:
        dry = dry.mean(axis=1)
    dry /= np.max(np.abs(dry)) + 1e-12

    _, _, _, _, hits = _run_trace(preset_name, n_rays=2000)
    ir = build_impulse_response(hits, sample_rate=sr, max_time=3.0)

    direct_ms = min((h.time for h in hits), default=0) * 1000
    boost_start = int((direct_ms / 1000.0 + 0.010) * sr)
    ir_boosted = ir.copy()
    if boost_start < len(ir_boosted):
        n_tail = len(ir_boosted) - boost_start
        ramp = np.linspace(1.0, 10.0, n_tail)
        ir_boosted[boost_start:] *= ramp[:, None]
    peak_ir = np.max(np.abs(ir_boosted))
    if peak_ir > 0:
        ir_boosted /= peak_ir

    wet_l = fftconvolve(dry, ir_boosted[:, 0], mode="full")
    wet_r = fftconvolve(dry, ir_boosted[:, 1], mode="full")
    wet = np.stack([wet_l, wet_r], axis=1)
    peak_w = np.max(np.abs(wet))
    if peak_w > 0:
        wet *= 0.88 / peak_w

    buf = io.BytesIO()
    wav.write(buf, sr, (wet * 32767).astype(np.int16))
    buf.seek(0)
    return send_file(buf, mimetype="audio/wav")


def _run_desktop():
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    print()
    print("  Archaeoacoustics: Virtual Soundscape Reconstruction")
    print(f"  Running at {url}")
    print("  Your browser should open automatically.")
    print("  Close this window to quit.")
    print()

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    if IS_FROZEN:
        _run_desktop()
    else:
        app.run(debug=True, port=5000, use_reloader=False)

import io
import os
import socket
import sys
import threading
import webbrowser

import numpy as np
import scipy.io.wavfile as wav
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
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
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({"error": "Audio upload must be smaller than 32 MB"}), 413


@app.errorhandler(ValueError)
def invalid_input(error):
    return jsonify({"error": str(error)}), 400


def _number(values, name, default, minimum, maximum, integer=False):
    try:
        value = float(values.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    if integer and not value.is_integer():
        raise ValueError(f"{name} must be a whole number")
    return int(value) if integer else value


def _rt60(ir: np.ndarray, sr: int) -> int:
    energy = ir ** 2
    edc = np.flip(np.cumsum(np.flip(energy)))
    if edc[0] <= 0:
        return 0
    edc_db = 10 * np.log10(edc / edc[0] + 1e-12)
    above = np.where(edc_db > -60)[0]
    return int(above[-1] / sr * 1000) if len(above) else 0


def _parse_pos(x, y):
    if x is None and y is None:
        return None
    try:
        position = (float(x), float(y))
    except (TypeError, ValueError):
        raise ValueError("Position requires two numeric coordinates") from None
    if not all(np.isfinite(position)):
        raise ValueError("Position coordinates must be finite")
    return position


def _run_trace(preset_name: str, n_rays: int = 2000, max_time: float = 2.5,
               source=None, receiver=None):
    room, preset_src, preset_recv, desc = PRESETS[preset_name]()
    source = source if source is not None else preset_src
    receiver = receiver if receiver is not None else preset_recv
    xs = [v for w in room.walls for v in (w.x0, w.x1)]
    ys = [v for w in room.walls for v in (w.y0, w.y1)]
    for position in (source, receiver):
        if not min(xs) < position[0] < max(xs) or not min(ys) < position[1] < max(ys):
            raise ValueError("Positions must be inside the room bounds")
    if np.linalg.norm(np.subtract(source, receiver)) < 0.3:
        raise ValueError("Speaker and listener must be at least 0.3 m apart")
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


@app.route("/samples/<path:name>")
def sample_file(name):
    return send_from_directory(_resource_path("samples"), name)


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

    n_rays = _number(request.args, "rays", 2000, 100, 10000, integer=True)
    max_time = _number(request.args, "max_time", 2.5, 0.1, 5)
    src_o    = _parse_pos(request.args.get("sx"), request.args.get("sy"))
    recv_o   = _parse_pos(request.args.get("rx"), request.args.get("ry"))

    room, source, receiver, desc, hits = _run_trace(
        preset_name, n_rays, max_time, source=src_o, receiver=recv_o)
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

    if "audio" not in request.files:
        raise ValueError("Choose a WAV file to process")
    audio_bytes = request.files["audio"].read()
    try:
        sr, dry = wav.read(io.BytesIO(audio_bytes))
    except (ValueError, OSError, EOFError):
        raise ValueError("Audio must be a valid WAV file") from None
    if not 8000 <= sr <= 192000 or dry.size == 0 or dry.ndim not in (1, 2):
        raise ValueError("Audio must contain samples at 8000–192000 Hz")
    if len(dry) / sr > 120:
        raise ValueError("Audio must be 120 seconds or shorter")
    if not np.all(np.isfinite(dry)):
        raise ValueError("Audio contains non-finite samples")
    if dry.dtype == np.uint8:
        dry = dry.astype(np.float64) - 128
    dry = dry.astype(np.float64)
    if dry.ndim > 1:
        dry = dry.mean(axis=1)
    dry /= np.max(np.abs(dry)) + 1e-12

    src_o  = _parse_pos(request.form.get("sx"), request.form.get("sy"))
    recv_o = _parse_pos(request.form.get("rx"), request.form.get("ry"))
    _, _, _, _, hits = _run_trace(preset_name, n_rays=2000,
                                  source=src_o, receiver=recv_o)
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

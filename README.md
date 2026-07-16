# Archaeoacoustics: Virtual Soundscape Reconstruction Engine

A 2D acoustic ray-tracing engine that reconstructs the listening experience of historically significant spaces. Currently only limited to Roman theatres, Renaissance cathedrals, Greek council halls, and medieval great halls. Record or upload audio, and the system convolves it with a physics-based impulse response to simulate how your voice would have sounded in that room.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square)
![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## How It Works

Linear pipeline:

```
Room geometry → Ray tracer → Impulse response → Convolution → .wav
```

### 1. Ray Tracing

`n` rays are emitted uniformly from the source position at angles spanning `[0, 2π)`. Each ray bounces off walls using the law of specular reflection:

```
D_r = D − 2(D · N)N
```

where `D` is the incident direction and `N` is the outward wall normal. Energy decays per bounce:

```
E_new = E * (1 − α) * exp(−κ·d)
```

where `α` is the wall's absorption coefficient, `κ = 0.002 m⁻¹` is air absorption, and `d` is travel distance. Geometric spreading applies a `1/d^1.5` factor. A ray hit is recorded whenever a ray segment passes within `receiver_radius` (default 0.3 m) of the receiver.

### 2. Acoustic Impulse Response (AIR)

Recorded hits `(time, energy, pan)` are mapped to sample indices `⌊t × Fs⌋` in a stereo float64 array. A 7-point Hann window smooths each spike to avoid aliasing. The result is normalised to peak amplitude 1.0. To make room character more audible, a linear ramp `[1×, 10×]` is applied to the IR tail (starting 10 ms after direct sound arrival), then renormalised — this lifts late reflections relative to the direct-sound peak.

### 3. Binaural Rendering

The IR is stereo. Each ray's arrival direction at the receiver is recorded relative to a listener facing the source, and its lateral component becomes a pan position `∈ [−1, 1]`. Two localisation cues are applied per hit:

- **ILD** (Interaural Level Difference): equal-power panning — `gain_L = cos((pan+1)π/4)`, `gain_R = sin((pan+1)π/4)`
- **ITD** (Interaural Time Difference): the far ear receives the spike up to 0.66 ms later (the acoustic path difference across a human head), scaled by `|pan|`

Early reflections therefore arrive from directions that match the room geometry — a wall close behind the listener produces a distinct slap-back on one side of the stereo field. Headphones recommended.

### 4. Convolution

```python
wet_l = fftconvolve(dry, ir_boosted[:, 0], mode="full")
wet_r = fftconvolve(dry, ir_boosted[:, 1], mode="full")
```

Each channel is convolved independently and the result written as a stereo WAV. Full-length output preserves the natural ring-out decay beyond the dry signal length. Output is normalised to −1.2 dBFS headroom.

---

## Architecture

| File | Responsibility |
|------|---------------|
| `engine.py` | `Material`, `Wall`, `Room`, `AcousticRayTracer`, `build_impulse_response` |
| `presets.py` | Four historical room definitions returning `(Room, source, receiver, desc)` |
| `colors.py` | Material display colours, shared by the web UI and the plots |
| `visualize.py` | Matplotlib plots for room geometry and IR (CLI only) |
| `main.py` | CLI entry point — argument parsing, convolution, file I/O |
| `app.py` | Flask REST API serving room data, trace results, and audio processing |
| `templates/index.html` | Single-page UI — canvas visualisers, Web Audio API, client WAV encoder |
| `archaeoacoustics.spec` | PyInstaller build definition for the standalone `.exe` |

The web server deliberately avoids importing `visualize.py` — matplotlib is a CLI-only dependency, and keeping it out of the server's import graph lets the packaged executable exclude it entirely (roughly a 50 MB saving).

---

## Historical Room Presets

| Key | Space | Dimensions | RT₆₀ |
|-----|-------|-----------|-------|
| `amphitheater` | Roman Stone Theatre | 40 × 25 m | ~94 ms |
| `cathedral` | Renaissance Cathedral Nave | 30 × 18 m | ~841 ms |
| `bouleuterion` | Greek Council Hall | 18 × 10 m | ~388 ms |
| `great_hall` | Medieval Castle Great Hall | 25 × 8 m | ~571 ms |

RT₆₀ values are simulation estimates and vary with ray count. Real-world values depend on furnishing density, audience absorption, and 3D geometry not captured by this 2D model.

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/dotuankhoi/archaeoacoustics_reconstruction.git
cd archaeoacoustics_reconstruction
pip install -r requirements.txt
```

---

## Usage

### Web UI (recommended)

```bash
python app.py
```

Open `http://localhost:5000` in your browser. Select a room preset, then record your voice, upload an audio file, or press **Try Sample** to use the bundled demo clip (a dry voice plus three claps — no microphone required). The UI renders:
- Animated ray simulation on the room geometry canvas
- **Draggable source and receiver** — grab either dot on the map and move it; the trace, impulse response, RT₆₀, and stereo image all update for the new positions, and any already-rendered audio is automatically re-rendered
- Acoustic Impulse Response spectrum
- Side-by-side dry/wet waveform comparison with in-browser playback

### CLI

```bash
# Apply cathedral acoustics to a dry recording
python main.py --preset cathedral --input voice.wav --output voice_cathedral.wav

# Visualise room geometry and IR without audio
python main.py --preset amphitheater --plot-only --rays 3000

# Save the raw impulse response
python main.py --preset great_hall --input voice.wav --save-ir ir_great_hall.wav

# List all presets
python main.py --list-presets
```

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | required | Room preset key |
| `--input` | — | Dry input `.wav` file |
| `--output` | auto | Output file path |
| `--rays` | 2000 | Number of rays to trace |
| `--bounces` | 60 | Max reflections per ray |
| `--max-time` | 2.5 | Reverb tail length (seconds) |
| `--plot` | false | Show room + IR plot after processing |
| `--plot-only` | false | Plot without audio convolution |
| `--save-ir` | — | Save raw impulse response as `.wav` |

---

## Building a Standalone Executable

The app can be packaged into a single `.exe` that runs without a Python installation. It starts the Flask server on a free port, opens the default browser, and serves the same UI.

```bash
pip install pyinstaller
pyinstaller archaeoacoustics.spec
```

The result is `dist/Archaeoacoustics.exe` (~52 MB). Double-click to run; close the console window to quit.

Recording works because `getUserMedia` treats `127.0.0.1` as a secure context — the same reason the UI works under `python app.py`. When frozen, `app.py` resolves `templates/` through PyInstaller's `sys._MEIPASS` and disables debug mode; running from source is unaffected.

---

## REST API

The Flask server exposes three endpoints used by the UI:

### `GET /api/presets`
Returns wall geometry, material absorption coefficients, source/receiver positions, and display colours for all presets.

### `GET /api/trace/<preset>?rays=2000`
Runs the ray tracer and returns the results below. Optional `sx`, `sy`, `rx`, `ry` query parameters override the preset's source and receiver positions (metres, room coordinates); omitted or malformed values fall back to the preset defaults.

```json
{
  "ir": [...],        // max-pooled IR magnitudes (2000 buckets)
  "n_hits": 1123,
  "rt60_ms": 634,
  "direct_ms": 49.6,
  "max_time": 2.5
}
```

IR data is max-pooled (not stride-sampled) to preserve narrow energy spikes for accurate visualisation.

### `POST /api/process`
Accepts `multipart/form-data` with `audio` (WAV) and `preset` fields, plus optional `sx`/`sy`/`rx`/`ry` position overrides. Returns the convolved wet audio as stereo `audio/wav`.

---

## Extending the Engine

**Add a material:**

```python
# engine.py → MATERIALS dict
"brick": Material("brick", absorption=0.03),
```

**Add a preset:**

```python
# presets.py
def my_room() -> tuple:
    room = Room()
    room.add_wall(x0, y0, x1, y1, "stone")
    # ... build a closed boundary
    return room, (src_x, src_y), (recv_x, recv_y), "Description"

PRESETS["my_room"] = my_room
```

**Key invariant:** walls must form a *closed boundary*. Any open gap lets rays escape and drastically reduces receiver hits.

---

## Quick Smoke Test

```bash
python -c "
from engine import Room, AcousticRayTracer, build_impulse_response
room = Room(); room.add_rect(0, 0, 10, 6)
tracer = AcousticRayTracer(room, (2,3), (8,3), n_rays=500, max_bounces=20, max_time=1.0)
hits = tracer.trace()
ir = build_impulse_response(hits)
assert len(hits) > 0 and ir.max() > 0
print('PASS', len(hits), 'hits')
"
```

---

I am still a high school student, and relatively inexperienced in doing projects like these. Feedback would be greatly appreciated :)

---

## License

MIT. Do whatever you want with it :)

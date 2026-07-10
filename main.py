import argparse
import sys
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import fftconvolve

from engine import AcousticRayTracer, build_impulse_response, MATERIALS
from presets import PRESETS
from visualize import plot_room_and_rays, plot_impulse_response_only


def load_wav(path: str) -> tuple[int, np.ndarray]:
    sr, data = wav.read(path)
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data /= np.max(np.abs(data)) + 1e-12
    return sr, data


def save_wav(path: str, sample_rate: int, audio: np.ndarray):
    audio = np.clip(audio, -1.0, 1.0)
    out = (audio * 32767).astype(np.int16)
    wav.write(path, sample_rate, out)
    print(f"  Saved -> {path}")


def convolve_with_ir(dry: np.ndarray, ir: np.ndarray) -> np.ndarray:
    wet = fftconvolve(dry, ir, mode="full")
    wet = wet[: len(dry)]
    peak = np.max(np.abs(wet))
    if peak > 0:
        wet *= 0.98 / peak
    return wet


def print_separator():
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Archaeoacoustics: historical room soundscape reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preset",       choices=list(PRESETS.keys()))
    parser.add_argument("--input",        metavar="FILE.wav")
    parser.add_argument("--output",       metavar="OUT.wav")
    parser.add_argument("--rays",         type=int,   default=2000)
    parser.add_argument("--bounces",      type=int,   default=60)
    parser.add_argument("--max-time",     type=float, default=2.5)
    parser.add_argument("--plot",         action="store_true")
    parser.add_argument("--plot-only",    action="store_true")
    parser.add_argument("--save-ir",      metavar="IR.wav")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--sample-rate",  type=int, default=44100)
    args = parser.parse_args()

    if args.list_presets:
        print("\nAvailable historical room presets:\n")
        for key, fn in PRESETS.items():
            _, _, _, desc = fn()
            print(f"  {key:<16}  {desc}")
        print()
        return

    if not args.preset:
        parser.print_help()
        sys.exit(1)

    print_separator()
    room, source, receiver, desc = PRESETS[args.preset]()
    print(f"  Preset   : {desc}")
    print(f"  Source   @ {source}")
    print(f"  Receiver @ {receiver}")
    print(f"  Walls    : {len(room.walls)}")
    print_separator()

    print(f"  Tracing {args.rays} rays  (max {args.bounces} bounces, {args.max_time:.1f}s tail)...")
    tracer = AcousticRayTracer(
        room, source, receiver,
        n_rays=args.rays,
        max_bounces=args.bounces,
        max_time=args.max_time,
        receiver_radius=0.3,
    )
    hits = tracer.trace()
    print(f"  Ray hits recorded : {len(hits)}")

    if not hits:
        print("  WARNING: No rays reached the receiver. Try increasing --rays.")

    sample_rate = args.sample_rate
    if args.input:
        sample_rate, _ = load_wav(args.input)

    ir = build_impulse_response(hits, sample_rate=sample_rate, max_time=args.max_time)
    print(f"  Impulse response  : {len(ir)} samples @ {sample_rate} Hz  "
          f"({len(ir)/sample_rate*1000:.0f} ms)")

    if args.save_ir:
        save_wav(args.save_ir, sample_rate, ir)

    if args.input and not args.plot_only:
        if not args.output:
            args.output = args.input.replace(".wav", f"_{args.preset}.wav")
            if args.output == args.input:
                args.output = "output_" + args.preset + ".wav"

        print(f"  Loading dry audio: {args.input}")
        sr_file, dry = load_wav(args.input)
        if sr_file != sample_rate:
            print(f"  WARNING: file SR ({sr_file} Hz) differs from IR SR ({sample_rate} Hz). "
                  f"Rebuilding IR at {sr_file} Hz.")
            ir = build_impulse_response(hits, sample_rate=sr_file, max_time=args.max_time)
            sample_rate = sr_file

        print(f"  Convolving  ({len(dry)} samples x {len(ir)}-sample IR)...")
        wet = convolve_with_ir(dry, ir)
        save_wav(args.output, sample_rate, wet)

    elif not args.input and not args.plot_only:
        print("  (No --input provided. Use --plot-only to visualise without audio.)")

    if args.plot or args.plot_only:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        fig = plot_room_and_rays(
            room, hits, source, receiver,
            title=desc, sample_rate=sample_rate, max_time=args.max_time
        )
        plt.show()

    print_separator()
    print("  Done.")
    print_separator()


if __name__ == "__main__":
    main()

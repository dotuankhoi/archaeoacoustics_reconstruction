import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from engine import Room, RayHit


MATERIAL_COLORS = {
    "Bare Stone":      "#8B8B83",
    "Brick":           "#CC5533",
    "Marble":          "#E8E8E0",
    "Plaster":         "#D4C9A8",
    "Wood Paneling":   "#8B6914",
    "Wood Floor":      "#A0784A",
    "Heavy Tapestry":  "#6B3A6B",
    "Thick Curtain":   "#4A5A7A",
    "Seated Audience": "#3A7A3A",
    "Compacted Soil":  "#8B7355",
    "Concrete":        "#909090",
    "Open Air":        "#87CEEB",
}


def plot_room_and_rays(room: Room, hits: list[RayHit],
                       source: tuple, receiver: tuple,
                       title: str = "", n_display_rays: int = 60,
                       sample_rate: int = 44100, max_time: float = 2.0):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#1A1A2E")
    for ax in axes:
        ax.set_facecolor("#0F0F1A")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    ax_room, ax_ir = axes

    legend_handles = {}
    for wall in room.walls:
        color = MATERIAL_COLORS.get(wall.material.name, "#AAAAAA")
        ax_room.plot([wall.x0, wall.x1], [wall.y0, wall.y1],
                     color=color, linewidth=3, solid_capstyle="round")
        if wall.material.name not in legend_handles:
            legend_handles[wall.material.name] = mpatches.Patch(
                color=color, label=f"{wall.material.name}  (α={wall.material.absorption:.2f})")

    ax_room.plot(*source,   "o", color="#FFD700", markersize=12, zorder=5, label="Source")
    ax_room.plot(*receiver, "^", color="#FF6B6B", markersize=12, zorder=5, label="Receiver")
    ax_room.annotate("SOURCE",   source,   color="#FFD700", fontsize=8,
                     xytext=(4, 4), textcoords="offset points")
    ax_room.annotate("RECEIVER", receiver, color="#FF6B6B", fontsize=8,
                     xytext=(4, 4), textcoords="offset points")

    ax_room.set_title(f"Room Geometry\n{title}", color="white", fontsize=10, pad=8)
    ax_room.set_xlabel("x (m)", color="#AAAACC")
    ax_room.set_ylabel("y (m)", color="#AAAACC")
    ax_room.set_aspect("equal")
    ax_room.legend(
        handles=list(legend_handles.values()) +
                [mpatches.Patch(color="#FFD700", label="Source"),
                 mpatches.Patch(color="#FF6B6B", label="Receiver")],
        loc="upper left", fontsize=7, facecolor="#1A1A2E", edgecolor="#444466",
        labelcolor="white")

    n_samples = int(max_time * sample_rate)
    ir = np.zeros(n_samples)
    for hit in hits:
        idx = int(hit.time * sample_rate)
        if 0 <= idx < n_samples:
            ir[idx] += hit.energy

    peak = np.max(np.abs(ir))
    if peak > 0:
        ir /= peak

    times_ms = np.arange(n_samples) / sample_rate * 1000
    ax_ir.fill_between(times_ms, ir, 0, color="#4488FF", alpha=0.6, linewidth=0)
    ax_ir.plot(times_ms, ir, color="#88AAFF", linewidth=0.5)

    direct_hits = sorted(hits, key=lambda h: h.time)
    if direct_hits:
        t0_ms = direct_hits[0].time * 1000
        ax_ir.axvline(t0_ms, color="#FFD700", linewidth=1.2, linestyle="--", alpha=0.8)
        ax_ir.text(t0_ms + 5, 0.85, "direct", color="#FFD700", fontsize=8)

    ax_ir.set_xlim(0, max_time * 1000)
    ax_ir.set_ylim(-0.05, 1.1)
    ax_ir.set_xlabel("Time (ms)", color="#AAAACC")
    ax_ir.set_ylabel("Normalised Energy", color="#AAAACC")
    ax_ir.set_title("Acoustic Impulse Response (AIR)", color="white", fontsize=10, pad=8)

    energy = ir ** 2
    edc = np.flip(np.cumsum(np.flip(energy)))
    edc_db = 10 * np.log10(edc / (edc[0] + 1e-12) + 1e-12)
    above_noise = np.where(edc_db > -60)[0]
    if len(above_noise) > 0:
        rt60_ms = times_ms[above_noise[-1]]
        ax_ir.axvspan(0, rt60_ms, alpha=0.06, color="#FFAA00")
        ax_ir.text(rt60_ms * 0.5, 1.0, f"RT60 ~ {rt60_ms:.0f} ms",
                   color="#FFAA44", fontsize=9, ha="center")

    plt.tight_layout(pad=2.0)
    return fig


def plot_impulse_response_only(ir: np.ndarray, sample_rate: int,
                                title: str = "Acoustic Impulse Response"):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#0F0F1A")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    times = np.arange(len(ir)) / sample_rate * 1000
    ax.fill_between(times, ir, 0, color="#4488FF", alpha=0.55)
    ax.plot(times, ir, color="#88CCFF", linewidth=0.6)

    ax.set_xlabel("Time (ms)", color="#AAAACC")
    ax.set_ylabel("Amplitude", color="#AAAACC")
    ax.set_title(title, color="white", fontsize=11)
    plt.tight_layout()
    return fig

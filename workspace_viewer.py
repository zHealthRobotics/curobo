#!/usr/bin/env python3
"""
Interactive workspace explorer for the workspace probe results.
===============================================================

Two-panel layout:
  * Left  — 3D scatter of every grid point (click a point to read its coords).
  * Right — 2D Z-slice grid; drag the slider to change Z, click a square to
            read that cell's coords/status.
  * Bottom — status box showing the last clicked point.

Categories (3-way, matching the original explorer):
  both        — grasp_ok AND pregrasp_ok   (green)
  grasp_only  — grasp_ok, pregrasp fails    (orange)
  neither      — everything else            (light gray)

Usage:
    python workspace_interactive.py
    python workspace_interactive.py /path/to/workspace_results.pkl

If no saved results file exists yet, run ``python workspace_probe.py`` first.
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from workspace_probe import load_results, RESULTS_PKL


# ── Category logic (3-way) ────────────────────────────────────────────────────

def _cat(r):
    if r["grasp_ok"] and r["pregrasp_ok"]:
        return "both"
    if r["grasp_ok"]:
        return "grasp_only"
    return "neither"


CAT_NAME = {
    "both": "Both ok (grasp + pre-grasp)",
    "grasp_only": "Grasp only",
    "neither": "Neither",
}

# 3D scatter styling
STYLE_3D = {
    "both":       dict(c="#2ca02c", s=60, alpha=0.95),
    "grasp_only": dict(c="#ff7f0e", s=45, alpha=0.9),
    "neither":    dict(c="lightgray", s=10, alpha=0.15),
}

# 2D square fill / edge
FILL_2D = {
    "both":       ("#2ca02c", "#1f7a1f"),
    "grasp_only": ("#ff7f0e", "#cc6500"),
    "neither":    ("#eeeeee", "#cccccc"),
}


def _orientation_qw_qy(data):
    """Best-effort extract (qw, qy) for the title; fall back to known values."""
    ori = data.get("orientation")
    qw = qy = None
    if isinstance(ori, dict):
        qw = ori.get("qw", ori.get("w"))
        qy = ori.get("qy", ori.get("y"))
    elif isinstance(ori, (list, tuple, np.ndarray)) and len(ori) >= 4:
        qw, _, qy, _ = ori[0], ori[1], ori[2], ori[3]
    if qw is None:
        qw = 0.7074
    if qy is None:
        qy = -0.7068
    return float(qw), float(qy)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else RESULTS_PKL
    try:
        data = load_results(path)
    except FileNotFoundError:
        print(f"No results file at {path!r}. Run `python workspace_probe.py` first.")
        sys.exit(1)

    results = data["results"]
    rh = data["right_home"]
    lh = data["left_home"]
    qw, qy = _orientation_qw_qy(data)

    # Derive the regular grid axes directly from the data.
    xs = sorted(set(round(r["x"], 6) for r in results))
    ys = sorted(set(round(r["y"], 6) for r in results))
    zs = sorted(set(round(r["z"], 6) for r in results))
    dx = (xs[1] - xs[0]) if len(xs) > 1 else 0.032
    dy = (ys[1] - ys[0]) if len(ys) > 1 else 0.032

    # Index records by (x, y, z) for fast lookup in the slice panel.
    by_xyz = {(round(r["x"], 6), round(r["y"], 6), round(r["z"], 6)): r for r in results}

    # ── Figure & axes layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8.5))
    fig.suptitle(
        f"Right Arm Workspace Explorer  —  grasp orientation qw={qw:.4f}  qy={qy:.4f}\n"
        "3D: click points to read coords   |   2D: drag slider to change Z, click squares",
        fontsize=11,
    )

    ax3d = fig.add_axes([0.02, 0.16, 0.46, 0.74], projection="3d")
    ax2d = fig.add_axes([0.55, 0.24, 0.42, 0.66])
    ax_slider = fig.add_axes([0.55, 0.12, 0.42, 0.03])

    # ── 3D scatter (all points) ───────────────────────────────────────────────
    for cat in ("neither", "grasp_only", "both"):
        rs = [r for r in results if _cat(r) == cat]
        if not rs:
            continue
        h = ax3d.scatter(
            [r["x"] for r in rs], [r["y"] for r in rs], [r["z"] for r in rs],
            label=CAT_NAME[cat], picker=5, **STYLE_3D[cat],
        )
        h._wp_records = rs

    ax3d.scatter([rh[0]], [rh[1]], [rh[2]], c="blue", marker="*", s=320,
                 edgecolors="k", label=f"Right home [{rh[0]:.3f}, {rh[1]:.3f}, {rh[2]:.3f}]")
    ax3d.scatter([lh[0]], [lh[1]], [lh[2]], c="gray", marker="*", s=320,
                 edgecolors="k", label=f"Left  home [{lh[0]:.3f}, {lh[1]:.3f}, {lh[2]:.3f}]")

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D view — click a point")
    ax3d.legend(loc="upper left", fontsize=7)

    # ── 2D Z-slice grid (redrawn on slider change) ────────────────────────────
    half_w, half_h = dx * 0.42, dy * 0.42

    def draw_slice(zi):
        z = zs[zi]
        ax2d.clear()
        for xv in xs:
            for yv in ys:
                r = by_xyz.get((round(xv, 6), round(yv, 6), round(z, 6)))
                cat = _cat(r) if r is not None else "neither"
                fc, ec = FILL_2D[cat]
                ax2d.add_patch(mpatches.Rectangle(
                    (xv - half_w, yv - half_h), 2 * half_w, 2 * half_h,
                    facecolor=fc, edgecolor=ec, linewidth=1.0,
                ))
        ax2d.set_xlim(xs[0] - dx, xs[-1] + dx)
        ax2d.set_ylim(ys[0] - dy, ys[-1] + dy)
        ax2d.set_xticks(xs)
        ax2d.set_xticklabels([f"{v:.2f}" for v in xs], rotation=45, fontsize=8)
        ax2d.set_yticks(ys)
        ax2d.set_yticklabels([f"{v:.3f}" for v in ys], fontsize=8)
        ax2d.set_xlabel("X (torso_link, m)")
        ax2d.set_ylabel("Y (torso_link, m)")
        ax2d.set_title(f"Z-slice  z = {z:.3f} m   (drag slider \u2193)")
        ax2d.set_aspect("equal", adjustable="box")
        legend_handles = [
            mpatches.Patch(facecolor=FILL_2D["both"][0], edgecolor=FILL_2D["both"][1],
                           label="Both ok (grasp + pre-grasp)"),
            mpatches.Patch(facecolor=FILL_2D["grasp_only"][0], edgecolor=FILL_2D["grasp_only"][1],
                           label="Grasp only"),
            mpatches.Patch(facecolor=FILL_2D["neither"][0], edgecolor=FILL_2D["neither"][1],
                           label="Neither"),
        ]
        ax2d.legend(handles=legend_handles, loc="lower right", fontsize=7, framealpha=0.9)
        fig.canvas.draw_idle()

    # ── Status box at the bottom ──────────────────────────────────────────────
    status = fig.text(
        0.5, 0.035, "Click a 3D point or a 2D square to read coordinates",
        ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", ec="steelblue", fc="white", lw=1.8),
    )

    def set_status(prefix, x, y, z, cat):
        status.set_text(
            f"{prefix}  \u2192   x = {x:.3f} m    y = {y:.3f} m    z = {z:.3f} m    "
            f"[{CAT_NAME[cat]}]"
        )
        fig.canvas.draw_idle()

    # ── Interaction ───────────────────────────────────────────────────────────
    def on_pick(event):
        recs = getattr(event.artist, "_wp_records", None)
        if recs is None or len(event.ind) == 0:
            return
        r = recs[event.ind[0]]
        set_status("3D click", r["x"], r["y"], r["z"], _cat(r))
        print(f"3D pick: x={r['x']:.3f} y={r['y']:.3f} z={r['z']:.3f} "
              f"grasp={r['grasp_ok']} pre={r['pregrasp_ok']} [{_cat(r)}]")

    def on_click(event):
        if event.inaxes is not ax2d or event.xdata is None:
            return
        # Snap to nearest grid cell.
        xv = min(xs, key=lambda v: abs(v - event.xdata))
        yv = min(ys, key=lambda v: abs(v - event.ydata))
        z = zs[int(slider.val)]
        r = by_xyz.get((round(xv, 6), round(yv, 6), round(z, 6)))
        cat = _cat(r) if r is not None else "neither"
        set_status("2D click", xv, yv, z, cat)
        print(f"2D click: x={xv:.3f} y={yv:.3f} z={z:.3f} [{cat}]")

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("button_press_event", on_click)

    # ── Slider ────────────────────────────────────────────────────────────────
    slider = Slider(
        ax_slider, "Z index", 0, len(zs) - 1,
        valinit=0, valstep=1,
    )
    slider.on_changed(lambda v: draw_slice(int(v)))

    draw_slice(0)

    print("Interactive explorer ready.")
    print("  Left panel : click any 3D point.")
    print("  Right panel: drag the Z slider, click a square.")
    print("Close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()

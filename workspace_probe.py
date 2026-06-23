#!/usr/bin/env python3
"""
Workspace reachability probe for the minibot RIGHT arm.
=======================================================

Maps the reachable workspace for the RIGHT arm at a fixed grasp orientation,
sweeping a grid of target points in the ``torso_link`` (planner base) frame.
For every grid point it runs two ``plan_pose`` tests with the LEFT (idle) arm
pinned to its home FK pose:

    a) GRASP     — plan_pose straight to the grid point.
    b) PRE-GRASP — plan_pose to (x - 0.05, y, z), i.e. a 5cm pull-back in -X.

This isolates "the grasp is reachable but the approach is not" failures from
"nothing here is reachable" failures.

Outputs
-------
* Text summary printed to stdout.
* /curobo/workspace_3d.png      — 3D scatter of the workspace.
* /curobo/workspace_slices.png  — per-Z 2D grid view (easier to read).
* /curobo/workspace_results.pkl — pickled results for workspace_interactive.py.

The results pickle / the in-memory dict returned by ``run_probe()`` has the
schema consumed by ``workspace_interactive.py``::

    {
        "results":      [ {x, y, z, grasp_ok, pregrasp_ok}, ... ],
        "right_home":   (x, y, z),   # RIGHT tool home FK position
        "left_home":    (x, y, z),   # LEFT  (idle) tool home FK position
        "orientation":  (qw, qx, qy, qz),
        "grid":         {"x": [...], "y": [...], "z": [...]},
    }

Run:
    cd ~/curobo && source .venv/bin/activate
    python workspace_probe.py
"""

import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch

# cuRobo
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, Pose as CuPose, ToolPoseCriteria

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — mirror curobo_v2.py
# ──────────────────────────────────────────────────────────────────────────────
ROBOT_CFG          = "minibot_dual_arm_new.yml"
WARMUP_ITERATIONS  = 10
MAX_ATTEMPTS       = 20

# Fixed grasp orientation (w, x, y, z)
GRASP_QUAT = (0.7074, 0.0, -0.7068, 0.0)

# Sweep grid (torso_link frame, metres). STEP chosen so the sweep lands at
# ~770 grid points (X=7 × Y=11 × Z=10) over the same extents.
STEP    = 0.032
X_RANGE = np.arange(0.10, 0.30 + 1e-9, STEP)
Y_RANGE = np.arange(-0.30, 0.05 + 1e-9, STEP)
Z_RANGE = np.arange(0.30, 0.60 + 1e-9, STEP)

# Pre-grasp pull-back along -X (metres)
PREGRASP_OFFSET = 0.05

RESULTS_PKL = "/curobo/workspace_results.pkl"
PLOT_3D     = "/curobo/workspace_3d.png"
PLOT_SLICES = "/curobo/workspace_slices.png"

DEVICE = "cuda"
DTYPE  = torch.float32


# ──────────────────────────────────────────────────────────────────────────────
# Probe
# ──────────────────────────────────────────────────────────────────────────────
def _round_grid(arr: np.ndarray) -> List[float]:
    return [round(float(v), 4) for v in arr]


def run_probe() -> Dict:
    """Run the full sweep and return the results dict (also pickles it)."""
    print("Initialising cuRobo MotionPlanner …")
    cfg = MotionPlannerCfg.create(robot=ROBOT_CFG)
    planner = MotionPlanner(cfg)

    print(f"Warming up ({WARMUP_ITERATIONS} iters) …")
    planner.warmup(num_warmup_iterations=WARMUP_ITERATIONS)

    joint_names = list(planner.joint_names)
    tool_frames = list(planner.tool_frames)
    right_tool = next(f for f in tool_frames if f.startswith("R"))
    left_tool  = next(f for f in tool_frames if f.startswith("L"))  # idle
    print(f"Tool frames: {tool_frames} | right={right_tool} | left(idle)={left_tool}")

    # ── Start state = planner default joint state ───────────────────────────────
    from curobo.types import JointState as CuJointState
    start_js = CuJointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=joint_names,
    )

    # ── Home FK poses (right home + left/idle pin) ──────────────────────────────
    active_js = planner.kinematics.get_active_js(start_js.clone())
    kin_state = planner.kinematics.compute_kinematics(active_js)
    right_fk = kin_state.tool_poses[right_tool]
    left_fk  = kin_state.tool_poses[left_tool]

    right_home = tuple(right_fk.position.reshape(3).cpu().numpy().tolist())
    left_home  = tuple(left_fk.position.reshape(3).cpu().numpy().tolist())
    print(f"Right home FK pos : [{right_home[0]:.4f}, {right_home[1]:.4f}, {right_home[2]:.4f}]")
    print(f"Left  home FK pos : [{left_home[0]:.4f}, {left_home[1]:.4f}, {left_home[2]:.4f}]")

    # Idle (left) arm pinned to its home FK pose for every plan
    idle_pose = CuPose(
        position=left_fk.position.reshape(1, 3),
        quaternion=left_fk.quaternion.reshape(1, 4),
    )

    qw, qx, qy, qz = GRASP_QUAT
    quat_t = torch.tensor([[qw, qx, qy, qz]], device=DEVICE, dtype=DTYPE)

    # Relaxed-orientation criteria: tight position, loose pitch/yaw on the active
    # (right) arm; idle (left) arm pose cost disabled.
    planner.update_tool_pose_criteria({
        right_tool: ToolPoseCriteria.track_position_and_orientation(
            xyz=[1.0, 1.0, 1.0], rpy=[1.0, 0.05, 0.05], non_terminal_scale=0.0,
        ),
        left_tool: ToolPoseCriteria.disabled(),
    })

    def _plan_to(px: float, py: float, pz: float) -> bool:
        goal = GoalToolPose.from_poses(
            {
                right_tool: CuPose(
                    position=torch.tensor([[px, py, pz]], device=DEVICE, dtype=DTYPE),
                    quaternion=quat_t.clone(),
                ),
                left_tool: idle_pose,
            },
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        )
        result = planner.plan_pose(goal, start_js, max_attempts=MAX_ATTEMPTS)
        return bool(result is not None and result.success.any().item())

    # ── Sweep ───────────────────────────────────────────────────────────────────
    total = len(X_RANGE) * len(Y_RANGE) * len(Z_RANGE)
    print(f"\nSweeping {total} grid points "
          f"(x:{len(X_RANGE)} × y:{len(Y_RANGE)} × z:{len(Z_RANGE)}), "
          f"2 plans each …\n")

    results: List[Dict] = []
    idx = 0
    for z in Z_RANGE:
        for y in Y_RANGE:
            for x in X_RANGE:
                idx += 1
                grasp_ok = _plan_to(float(x), float(y), float(z))
                pregrasp_ok = _plan_to(float(x) - PREGRASP_OFFSET, float(y), float(z))
                results.append({
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                    "z": round(float(z), 4),
                    "grasp_ok": grasp_ok,
                    "pregrasp_ok": pregrasp_ok,
                })
                flag = ("BOTH" if grasp_ok and pregrasp_ok else
                        "grasp-only" if grasp_ok else
                        "pregrasp-only" if pregrasp_ok else "none")
                print(f"  [{idx:4d}/{total}] "
                      f"x={x:5.2f} y={y:6.2f} z={z:5.2f} → {flag}")

    data = {
        "results": results,
        "right_home": right_home,
        "left_home": left_home,
        "orientation": GRASP_QUAT,
        "grid": {
            "x": _round_grid(X_RANGE),
            "y": _round_grid(Y_RANGE),
            "z": _round_grid(Z_RANGE),
        },
    }

    with open(RESULTS_PKL, "wb") as fh:
        pickle.dump(data, fh)
    print(f"\nSaved results → {RESULTS_PKL}")

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Summary + plots
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_range(vals: List[float]) -> str:
    if not vals:
        return "—"
    return f"{min(vals):.2f} … {max(vals):.2f}"


def print_summary(data: Dict) -> None:
    results = data["results"]
    total = len(results)
    grasp = [r for r in results if r["grasp_ok"]]
    both  = [r for r in results if r["grasp_ok"] and r["pregrasp_ok"]]
    grasp_only = [r for r in results if r["grasp_ok"] and not r["pregrasp_ok"]]
    pre_only   = [r for r in results if r["pregrasp_ok"] and not r["grasp_ok"]]
    neither    = [r for r in results if not r["grasp_ok"] and not r["pregrasp_ok"]]

    rh = data["right_home"]
    lh = data["left_home"]
    qw, qx, qy, qz = data["orientation"]

    print("\n" + "=" * 64)
    print("WORKSPACE PROBE SUMMARY — RIGHT arm, fixed grasp orientation")
    print("=" * 64)
    print(f"Grasp orientation (wxyz) : [{qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f}]")
    print(f"Right home FK position   : [{rh[0]:.4f}, {rh[1]:.4f}, {rh[2]:.4f}]")
    print(f"Left  home FK position   : [{lh[0]:.4f}, {lh[1]:.4f}, {lh[2]:.4f}]")
    print(f"Pre-grasp offset         : -{PREGRASP_OFFSET:.2f} m along X")
    print("-" * 64)
    print(f"Total grid points        : {total}")
    print(f"Grasp reachable          : {len(grasp):4d}  ({100*len(grasp)/total:5.1f}%)")
    print(f"BOTH reachable           : {len(both):4d}  ({100*len(both)/total:5.1f}%)")
    print(f"Grasp-only (pre fails)   : {len(grasp_only):4d}  "
          f"({100*len(grasp_only)/total:5.1f}%)  ← approach blocked")
    print(f"Pre-grasp-only           : {len(pre_only):4d}  ({100*len(pre_only)/total:5.1f}%)")
    print(f"Neither                  : {len(neither):4d}  ({100*len(neither)/total:5.1f}%)")
    print("-" * 64)
    print("Ranges where BOTH succeed:")
    print(f"  X : {_fmt_range([r['x'] for r in both])}")
    print(f"  Y : {_fmt_range([r['y'] for r in both])}")
    print(f"  Z : {_fmt_range([r['z'] for r in both])}")
    print("=" * 64 + "\n")


def make_plots(data: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    results = data["results"]
    rh = data["right_home"]
    lh = data["left_home"]

    both = [r for r in results if r["grasp_ok"] and r["pregrasp_ok"]]
    grasp_only = [r for r in results if r["grasp_ok"] and not r["pregrasp_ok"]]
    neither = [r for r in results if not r["grasp_ok"] and not r["pregrasp_ok"]]
    # (pre-grasp-only are rare; group them visually with "neither")
    neither = neither + [r for r in results
                         if r["pregrasp_ok"] and not r["grasp_ok"]]

    def xyz(rs):
        if not rs:
            return [], [], []
        return ([r["x"] for r in rs], [r["y"] for r in rs], [r["z"] for r in rs])

    # ── Plot A: 3D scatter ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = xyz(neither)
    ax.scatter(x, y, z, c="lightgray", s=12, alpha=0.2, label="neither")
    x, y, z = xyz(grasp_only)
    ax.scatter(x, y, z, c="orange", s=55, alpha=0.9,
               label="grasp only (pre-grasp fails)")
    x, y, z = xyz(both)
    ax.scatter(x, y, z, c="green", s=90, alpha=0.95,
               label="both grasp + pre-grasp")

    ax.scatter([rh[0]], [rh[1]], [rh[2]], c="blue", marker="*", s=400,
               edgecolors="k", label="right home (FK)")
    ax.scatter([lh[0]], [lh[1]], [lh[2]], c="gray", marker="*", s=400,
               edgecolors="k", label="left/idle home (FK)")

    ax.set_xlabel("X (torso_link, m)")
    ax.set_ylabel("Y (torso_link, m)")
    ax.set_zlabel("Z (torso_link, m)")
    ax.set_title("Right-arm reachable workspace (fixed grasp orientation)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_3D, dpi=130)
    plt.close(fig)
    print(f"Saved plot → {PLOT_3D}")

    # ── Plot B: per-Z 2D slice grid ─────────────────────────────────────────────
    z_levels = data["grid"]["z"]
    x_vals = data["grid"]["x"]
    y_vals = data["grid"]["y"]

    # category lookup: 2=both, 1=grasp only, 0=neither
    def cat(r):
        if r["grasp_ok"] and r["pregrasp_ok"]:
            return 2
        if r["grasp_ok"]:
            return 1
        return 0

    lut = {(r["x"], r["y"], r["z"]): cat(r) for r in results}

    n = len(z_levels)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows),
                             squeeze=False)

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["white", "orange", "green"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for i, z in enumerate(z_levels):
        ax = axes[i // ncols][i % ncols]
        grid = np.full((len(y_vals), len(x_vals)), 0)
        for yi, yv in enumerate(y_vals):
            for xi, xv in enumerate(x_vals):
                grid[yi, xi] = lut.get((xv, yv, z), 0)
        half = STEP / 2.0
        ax.imshow(grid, origin="lower", cmap=cmap, norm=norm, aspect="auto",
                  extent=[min(x_vals) - half, max(x_vals) + half,
                          min(y_vals) - half, max(y_vals) + half])
        ax.set_title(f"z = {z:.2f} m")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        # mark home if this z is the closest slice to the right home
        if abs(z - rh[2]) <= half:
            ax.plot(rh[0], rh[1], marker="*", color="blue", markersize=16,
                    markeredgecolor="k")

    # hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    legend_handles = [
        Patch(facecolor="green", edgecolor="k", label="both"),
        Patch(facecolor="orange", edgecolor="k", label="grasp only"),
        Patch(facecolor="white", edgecolor="k", label="neither"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Right-arm reachability by Z slice "
                 "(X horizontal, Y vertical, torso_link frame)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(PLOT_SLICES, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {PLOT_SLICES}")


def load_results(path: str = RESULTS_PKL) -> Dict:
    """Load a previously-saved probe result (used by workspace_interactive.py)."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main():
    data = run_probe()
    print_summary(data)
    make_plots(data)


if __name__ == "__main__":
    main()


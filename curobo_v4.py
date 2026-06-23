#!/usr/bin/env python3
"""
cuRobo GPU-accelerated Motion Planning Node for minibot dual arm
================================================================
Subscribes : /curobo/arm_command  (minibot_msgs/ArmCommand)
             /joint_states         (sensor_msgs/JointState)
             /curobo/execute_phase (std_msgs/String)

Publishes  : /dual_arm_controller/joint_trajectory  (trajectory_msgs/JointTrajectory)
             /curobo/status                          (std_msgs/String)

ArmCommand.mode:
    LEFT  (0) → plan left arm to left_pose;   right arm holds current position
    RIGHT (1) → plan right arm to right_pose;  left arm holds current position
    DUAL  (2) → plan both arms simultaneously

Execute phases (/curobo/execute_phase):
    "grasp"      → run the cached grasp-closing trajectory
    "lift"       → run the cached lift trajectory
    "pour_right" → joint-space (cspace) plan of the RIGHT arm to a fixed pour
                   configuration; left arm holds current position
    "pour_left"  → joint-space (cspace) plan of the LEFT arm to a fixed pour
                   configuration; right arm holds current position
    "pouring"    → the actual pour motion: tilt the pourer arm's wrist to a
                   fixed config (POUR_RIGHT_TILT_JOINTS). Open loop, no planning
                   — a single waypoint published to the controller, which
                   interpolates over POURING_DURATION_SEC. Catcher arm held at
                   its current angles. "pouring"/"pouring_right" → right arm;
                   "pouring_left" → left arm.
    "upright_right" / "upright_left" → after pouring, return the pourer arm
                   from its tilt pose back to the pour-hold pose
                   (POUR_RIGHT_JOINTS / POUR_LEFT_JOINTS). Open loop, single
                   waypoint, same duration as pouring. Idle arm held at current
                   angles. Publishes "upright_done" on completion.
    "home_right" / "home_left" → drive the active arm to the all-zeros joint
                   position [0,0,0,0,0,0,0]. Open loop, single waypoint over
                   HOME_DURATION_SEC. Idle arm held at current angles.
                   Publishes "home_done" on completion.

v4 changes vs v3:
    Adds the "pouring" phase — the tilt that actually pours liquid once the
    pourer arm has reached its pour-hold pose. It is intentionally NOT planned:
    both endpoints (pour-hold pose, tilt pose) are fixed constants, so a single
    hardcoded waypoint published to the JointTrajectoryController is fully
    deterministic and repeatable for product demos. The catcher arm holds its
    current angles so the cup it carries stays put.

v3 changes vs v2:
    Adds an open-loop pour milestone driven entirely in joint space. The pour
    pose is a fixed constant (POUR_RIGHT_JOINTS / POUR_LEFT_JOINTS), so no new
    ArmCommand fields or IDL changes are needed — the existing execute_phase
    channel carries the new "pour_right"/"pour_left" phases. Unlike plan_pose,
    plan_cspace has no per-tool pose criteria, so no enable/disable dance.

All three modes publish to /dual_arm_controller (14 DOF) so no controller
switching is ever needed. The idle arm in single-arm modes receives a
trajectory where every waypoint repeats its current joint angles at zero
velocity/acceleration — the JointTrajectoryController holds it still.

Run:
    source /opt/ros/humble/setup.bash
    cd ~/curobo && source .venv/bin/activate
    python curobo_v4.py
"""

import sys
import threading
import time
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")
sys.path.insert(0, "/opt/ros/humble/local/lib/python3.10/dist-packages")

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState as RosJointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from builtin_interfaces.msg import Duration

# cuRobo
# Enable native CUDA-graph reset BEFORE importing/creating any cuRobo objects.
# Default is False; with it on, the trajopt solver re-captures its CUDA graph
# when the goal-buffer shape changes (e.g. switching between pose planning and
# joint-space plan_cspace for the pour phase) instead of raising "CUDA graph
# reset is not available" and leaving a stale graph -> cudaErrorIllegalAddress.
# Requires CUDA 12.0+ (this box has 12.6). Must be set on curobo.runtime (the
# re-export module that curobo reads), not curobo._src.runtime.
import curobo.runtime as _curobo_runtime
_curobo_runtime.cuda_graph_reset = True

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState as CuJointState, ToolPoseCriteria

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — edit to match your setup
# ──────────────────────────────────────────────────────────────────────────────
ROBOT_CFG       = "minibot_dual_arm_new.yml"
SCENE_CFG       = None          # e.g. "collision_scene.yml", or None

CMD_TOPIC       = "/curobo/arm_command"
JS_TOPIC        = "/joint_states"
STATUS_TOPIC    = "/curobo/status"
DUAL_TRAJ_TOPIC = "/dual_arm_controller/joint_trajectory"

# Joint names — must match minibot_dual_arm.yml AND ros2_controllers.yaml
# Order: left first (indices 0-6), right second (indices 7-13)
LEFT_JOINT_NAMES = [
    "LSP_joint", "LSY_joint", "LSR_joint",
    "LEP_joint", "LWR_joint", "LWP_joint", "LWY_joint",
]
RIGHT_JOINT_NAMES = [
    "RSP_joint", "RSY_joint", "RSR_joint",
    "REP_joint", "RWR_joint", "RWP_joint", "RWY_joint",
]
ALL_JOINT_NAMES = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES  # 14 DOF

# Planning knobs
WARMUP_ITERATIONS = 10
MAX_ATTEMPTS      = 20

# Start-state joint-limit clamp margin (rad). The robot's elbow joints (REP/LEP)
# have a hard limit at exactly 0.0 — which is also the home rest pose — so live
# encoder noise tips the real start a few mrad outside the bound. cuRobo's
# feasibility check then rejects the start ("Start or End state in collision")
# and planning fails. We clamp every start/goal joint into [lower+eps, upper-eps]
# so a near-limit pose always sits safely inside the valid box. The margin is far
# below joint resolution, so this never moves the arm meaningfully.
START_CLAMP_EPS = 0.05

# ── Pour targets (joint space) ─────────────────────────────────────────────────
# Fixed tilt configuration the active arm drives to during the pour phase.
# Right-arm joint order is [RSP, RSY, RSR, REP, RWR, RWP, RWY] (cols 7:14),
# which maps 1:1 to SP/SY/SR/EP/WR/WP/WY.
POUR_RIGHT_JOINTS = np.array(
    [0.12, 0.53, -0.44, 1.47, 0.50, -0.38, -1.15], dtype=np.float32
)
# Left-arm pour configuration is not defined yet (mirror it once the left pour
# pose is measured). Left-arm joint order is [LSP, LSY, LSR, LEP, LWR, LWP, LWY]
# (cols 0:7). Leave as None until defined — "pour_left" will be rejected.
POUR_LEFT_JOINTS = np.array(
    [-0.12, 0.13, 0.22, -0.70, -0.35, -0.68, 1.26], dtype=np.float32
)
# ── Pouring TILT targets (the actual pour motion) ──────────────────────────────
# After the pourer arm reaches its pour-hold pose (POUR_*_JOINTS via plan_cspace),
# the "pouring" phase tilts its wrist to this fixed config so liquid pours. The
# controller interpolates from the current (pour-hold) pose to this target over
# POURING_DURATION_SEC, so the motion is smooth and fully deterministic — both
# endpoints are fixed, no planning involved. The catcher arm is held at its
# current angles (NOT commanded to zero) so the cup stays put. Same 7-DOF order
# as the pour constants: right = [RSP, RSY, RSR, REP, RWR, RWP, RWY].
POUR_RIGHT_TILT_JOINTS = np.array(
    [0.12, 0.53, -0.44, 1.47, -0.30, -0.10, -1.15], dtype=np.float32
)
# Left-arm tilt not defined yet — define if the left arm is ever the pourer.
POUR_LEFT_TILT_JOINTS = None
POURING_DURATION_SEC   = 3.0   # controller interpolation time to the tilt pose
HOME_DURATION_SEC      = 4.0   # controller interpolation time for arm return to home (0-joints)

# ArmCommand mode constants (must match ArmCommand.msg)
MODE_LEFT  = 0
MODE_RIGHT = 1
MODE_DUAL  = 2
# ──────────────────────────────────────────────────────────────────────────────


def _extract_pose(pose_stamped_msg):
    """Unpack geometry_msgs/PoseStamped → (px, py, pz, qw, qx, qy, qz)."""
    p = pose_stamped_msg.pose.position
    q = pose_stamped_msg.pose.orientation
    return p.x, p.y, p.z, q.w, q.x, q.y, q.z


def _make_single_arm_traj(
    active_interp: CuJointState,
    idle_pos_np: np.ndarray,    # (7,) current angles of the idle arm
    active_is_left: bool,
    dt: float,
    clock,
) -> JointTrajectory:
    """
    Build a 14-DOF JointTrajectory for dual_arm_controller.

    Active arm  → full cuRobo interpolated trajectory.
    Idle arm    → constant position (current angles) with zero vel/accel
                  repeated for every waypoint → controller holds it still.
    """
    msg = JointTrajectory()
    msg.header.stamp = clock.now().to_msg()
    msg.joint_names  = ALL_JOINT_NAMES  # left joints first, then right

    pos = active_interp.position
    vel = active_interp.velocity
    acc = active_interp.acceleration

    # (1, T, 7) → (T, 7)
    if pos.dim() == 3:
        pos = pos.squeeze(0)
    if vel is not None and vel.dim() == 3:
        vel = vel.squeeze(0)
    if acc is not None and acc.dim() == 3:
        acc = acc.squeeze(0)

    # Planner is 14-DOF (dual arm), so reshape to (-1, 14) then slice the
    # active arm's 7 joints. Left arm = cols 0:7, right arm = cols 7:14.
    n_dof = pos.shape[-1]  # 14 for dual-arm planner
    pos_np = pos.cpu().numpy().reshape(-1, n_dof).astype(np.float64)
    vel_np = vel.cpu().numpy().reshape(-1, n_dof).astype(np.float64) if vel is not None else np.zeros_like(pos_np)
    acc_np = acc.cpu().numpy().reshape(-1, n_dof).astype(np.float64) if acc is not None else np.zeros_like(pos_np)

    # Slice active arm joints; use passed-in idle_pos_np to hold idle arm still
    # (the planner's idle-arm output may drift slightly from current joint state)
    if active_is_left:
        active_pos_np = pos_np[:, 0:7]
        active_vel_np = vel_np[:, 0:7]
        active_acc_np = acc_np[:, 0:7]
    else:
        active_pos_np = pos_np[:, 7:14]
        active_vel_np = vel_np[:, 7:14]
        active_acc_np = acc_np[:, 7:14]

    idle_pos  = idle_pos_np.astype(np.float64)
    idle_zero = np.zeros(7)

    for i in range(active_pos_np.shape[0]):
        pt = JointTrajectoryPoint()

        if active_is_left:
            pt.positions     = [*active_pos_np[i], *idle_pos]
            pt.velocities    = [*active_vel_np[i], *idle_zero]
            pt.accelerations = [*active_acc_np[i], *idle_zero]
        else:
            pt.positions     = [*idle_pos,  *active_pos_np[i]]
            pt.velocities    = [*idle_zero, *active_vel_np[i]]
            pt.accelerations = [*idle_zero, *active_acc_np[i]]

        t = i * dt
        pt.time_from_start = Duration(
            sec=int(t),
            nanosec=int(round((t % 1.0) * 1e9)),
        )
        msg.points.append(pt)

    return msg


def _make_dual_arm_traj(
    interpolated: CuJointState,
    dt: float,
    clock,
) -> JointTrajectory:
    """Build a 14-DOF JointTrajectory directly from a full dual-arm cuRobo result."""
    msg = JointTrajectory()
    msg.header.stamp = clock.now().to_msg()
    msg.joint_names  = ALL_JOINT_NAMES

    pos = interpolated.position
    vel = interpolated.velocity
    acc = interpolated.acceleration

    if pos.dim() == 3:
        pos = pos.squeeze(0)
    if vel is not None and vel.dim() == 3:
        vel = vel.squeeze(0)
    if acc is not None and acc.dim() == 3:
        acc = acc.squeeze(0)

    n_dof = pos.shape[-1]
    pos_np = pos.cpu().numpy().reshape(-1, n_dof)[:, :14].astype(np.float64)
    vel_np = vel.cpu().numpy().reshape(-1, n_dof)[:, :14].astype(np.float64) if vel is not None else None
    acc_np = acc.cpu().numpy().reshape(-1, n_dof)[:, :14].astype(np.float64) if acc is not None else None

    for i in range(pos_np.shape[0]):
        pt = JointTrajectoryPoint()
        pt.positions     = pos_np[i].tolist()
        pt.velocities    = vel_np[i].tolist() if vel_np is not None else []
        pt.accelerations = acc_np[i].tolist() if acc_np is not None else []
        t = i * dt
        pt.time_from_start = Duration(
            sec=int(t),
            nanosec=int(round((t % 1.0) * 1e9)),
        )
        msg.points.append(pt)

    return msg


# ──────────────────────────────────────────────────────────────────────────────

class MinibotCuRoboNode(Node):

    def __init__(self):
        super().__init__("curobo_minibot_planner")

        # ── cuRobo ────────────────────────────────────────────────────────────
        self.get_logger().info("Initialising cuRobo for minibot dual arm…")

        cfg_kwargs = dict(robot=ROBOT_CFG, self_collision_check=False)
        if SCENE_CFG:
            cfg_kwargs["scene_model"] = SCENE_CFG

        cfg = MotionPlannerCfg.create(**cfg_kwargs)
        self.planner = MotionPlanner(cfg)

        self.get_logger().info(f"Warming up ({WARMUP_ITERATIONS} iters)…")
        self.planner.warmup(num_warmup_iterations=WARMUP_ITERATIONS)

        self.joint_names = list(self.planner.joint_names)   # 14 DOF
        self.interp_dt   = self.planner.trajopt_solver.config.interpolation_dt

        # Build per-joint clamp bounds aligned to self.joint_names. position is
        # [2, n] with row 0 = lower, row 1 = upper; reorder to planning order.
        jl = self.planner.kinematics.get_joint_limits()
        lim_lower = jl.position[0, :].cpu().numpy()
        lim_upper = jl.position[1, :].cpu().numpy()
        order = [jl.joint_names.index(n) for n in self.joint_names]
        lo = lim_lower[order]
        hi = lim_upper[order]
        self._clamp_lower = (lo + START_CLAMP_EPS).astype(np.float32)
        self._clamp_upper = (hi - START_CLAMP_EPS).astype(np.float32)
        # Guard: if a joint's range is narrower than 2*eps, the margins cross —
        # collapse that joint to the range midpoint instead.
        mid = (0.5 * (lo + hi)).astype(np.float32)
        crossed = self._clamp_lower > self._clamp_upper
        self._clamp_lower[crossed] = mid[crossed]
        self._clamp_upper[crossed] = mid[crossed]
        self.get_logger().info(
            f"Start-state joint clamp active (eps={START_CLAMP_EPS} rad) on "
            f"{len(self.joint_names)} joints."
        )

        # Tool frames: expect [LWY_link, RWY_link] from minibot_dual_arm.yml
        self.left_tool  = [f for f in self.planner.tool_frames if f.startswith("L")]
        self.right_tool = [f for f in self.planner.tool_frames if f.startswith("R")]

        self.get_logger().info(
            f"Ready | left tool: {self.left_tool} | "
            f"right tool: {self.right_tool} | interp_dt: {self.interp_dt:.3f}s"
        )

        # ── state ─────────────────────────────────────────────────────────────
        self._lock            = threading.Lock()
        self._current_pos_np: Optional[np.ndarray]   = None   # (14,) raw angles, CUDA tensor built lazily on planning thread
        self._is_planning     = False

        # ── grasp-phase state ─────────────────────────────────────────────────
        self._grasp_cache    = None   # GraspPlanResult from the last plan_grasp() call
        self._grasp_idle_pos = None   # (7,) idle arm positions for trajectory building
        self._grasp_arm      = None   # "left" or "right" — which arm is active
        self._lift_available = False  # True only when lift_interpolated_trajectory is not None

        self._default_js     = CuJointState.from_position(
            self.planner.default_joint_state.position.unsqueeze(0),
            joint_names=self.joint_names,
        )
        self._default_pos_np = self.planner.default_joint_state.position.cpu().numpy()

        # NOTE: do NOT warm up the cspace solver here. solve_cspace uses a
        # different batch shape than the pose graph captured by planner.warmup(),
        # and this platform lacks CUDA-graph reset (is_cuda_graph_reset_available()
        # is False — needs CUDA 12.0+). The shape change triggers reset_shape(),
        # which reallocates the shared trajopt_solver's buffers and then raises
        # "CUDA graph reset is not available", leaving the captured pose graph
        # pointing at stale memory. The next plan_grasp then replays that corrupt
        # graph and dies with cudaErrorIllegalAddress. See _plan_pour for the
        # runtime counterpart of this constraint.

        # ── ROS2 ──────────────────────────────────────────────────────────────
        cb = ReentrantCallbackGroup()

        self._js_sub = self.create_subscription(
            RosJointState, JS_TOPIC, self._js_cb, 10, callback_group=cb,
        )

        try:
            from my_robot_interfaces.msg import ArmCommand
            self._cmd_sub = self.create_subscription(
                ArmCommand, CMD_TOPIC, self._cmd_cb, 10, callback_group=cb,
            )
            self.get_logger().info(f"Subscribed to {CMD_TOPIC} (ArmCommand)")
        except ImportError:
            self.get_logger().warn(
                "my_robot_interfaces not found — ArmCommand subscription skipped. "
                "Build the ROS2 package and re-run."
            )

        self._phase_sub = self.create_subscription(
            String, "/curobo/execute_phase", self._phase_cb, 10, callback_group=cb,
        )

        # Single publisher — always dual_arm_controller
        self._dual_pub   = self.create_publisher(JointTrajectory, DUAL_TRAJ_TOPIC, 10)
        self._status_pub = self.create_publisher(String, STATUS_TOPIC, 10)

        self.get_logger().info(
            f"\n"
            f"  Command topic  : {CMD_TOPIC}\n"
            f"  Joint states   : {JS_TOPIC}\n"
            f"  Trajectory out : {DUAL_TRAJ_TOPIC}  (always 14-DOF dual)\n"
            f"  Status         : {STATUS_TOPIC}\n"
            f"\n"
            f"  Modes:\n"
            f"    LEFT  (0) → left arm moves,  right arm holds position\n"
            f"    RIGHT (1) → right arm moves, left  arm holds position\n"
            f"    DUAL  (2) → both arms move simultaneously\n"
            f"  Phases: grasp | lift | pour_right | pour_left | pouring | upright_right | upright_left | home_right | home_left\n"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _js_cb(self, msg: RosJointState):
        """Cache latest joint state reordered to cuRobo's 14-DOF order.

        NOTE: this runs on the (multi-threaded) ROS executor. It must NOT touch
        CUDA — doing so races with CUDA-graph capture/replay on the planning
        thread and corrupts the allocator (cudaErrorIllegalAddress). We only
        stash the reordered NumPy positions here; the CUDA tensor is built
        lazily on the planning thread in _get_start().
        """
        try:
            positions = []
            for name in self.joint_names:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])
            pos_np = np.array(positions, dtype=np.float32)
            with self._lock:
                self._current_pos_np = pos_np
        except ValueError as e:
            self.get_logger().warn(
                f"Joint name mismatch: {e}", throttle_duration_sec=5.0
            )

    def _cmd_cb(self, msg):
        """Dispatch ArmCommand to the correct planning routine."""
        if self._is_planning:
            self.get_logger().warn("Already planning — command ignored.")
            return

        mode   = msg.mode
        action = (msg.action or "").strip().lower()

        if mode == MODE_LEFT:
            arm = "left"
        elif mode == MODE_RIGHT:
            arm = "right"
        else:
            arm = None

        if action == "grasp":
            if arm is None:
                self.get_logger().error(
                    f"action=grasp requires mode LEFT or RIGHT, got mode={mode}"
                )
                return
            self.get_logger().info(f"Action: GRASP ({arm} arm) → plan_grasp pipeline")
            pose = msg.left_pose if arm == "left" else msg.right_pose
            threading.Thread(
                target=self._plan_grasp_single,
                args=(arm, *_extract_pose(pose)),
                daemon=True,
            ).start()
            return

        if mode == MODE_LEFT:
            self.get_logger().info("Mode: LEFT → right arm holds position")
            threading.Thread(
                target=self._plan_single,
                args=("left", *_extract_pose(msg.left_pose)),
                daemon=True,
            ).start()

        elif mode == MODE_RIGHT:
            self.get_logger().info("Mode: RIGHT → left arm holds position")
            threading.Thread(
                target=self._plan_single,
                args=("right", *_extract_pose(msg.right_pose)),
                daemon=True,
            ).start()

        elif mode == MODE_DUAL:
            self.get_logger().info("Mode: DUAL → both arms move")
            threading.Thread(
                target=self._plan_dual,
                args=(*_extract_pose(msg.left_pose), *_extract_pose(msg.right_pose)),
                daemon=True,
            ).start()

        else:
            self.get_logger().error(
                f"Unknown mode {mode} — expected 0=LEFT, 1=RIGHT, 2=DUAL"
            )

    def _phase_cb(self, msg: String):
        """Trigger the next cached/joint-space phase (grasp, lift, pour) on request."""
        phase = msg.data.strip().lower()

        if phase == "grasp":
            if self._is_planning:
                self.get_logger().warn("Phase=grasp requested while planning — ignoring.")
                return
            if self._grasp_cache is None:
                self.get_logger().warn("Phase=grasp requested but no grasp cached — ignoring.")
                return
            threading.Thread(target=self._execute_grasp_phase, daemon=True).start()

        elif phase == "lift":
            if self._is_planning:
                self.get_logger().warn("Phase=lift requested while planning — ignoring.")
                return
            if self._grasp_cache is None:
                self.get_logger().warn("Phase=lift requested but no grasp cached — ignoring.")
                return
            threading.Thread(target=self._execute_lift_phase, daemon=True).start()

        elif phase in ("pour_right", "pour_left"):
            if self._is_planning:
                self.get_logger().warn(f"Phase={phase} requested while planning — ignoring.")
                return
            arm = "right" if phase == "pour_right" else "left"
            self.get_logger().info(f"Phase: POUR ({arm} arm) → plan_cspace to fixed config")
            threading.Thread(target=self._plan_pour, args=(arm,), daemon=True).start()

        elif phase in ("pouring", "pouring_right", "pouring_left"):
            if self._is_planning:
                self.get_logger().warn(f"Phase={phase} requested while planning — ignoring.")
                return
            # "pouring" defaults to the right arm (the usual pourer in the demo).
            arm = "left" if phase == "pouring_left" else "right"
            self.get_logger().info(
                f"Phase: POURING ({arm} arm) → open-loop tilt to fixed config"
            )
            threading.Thread(target=self._execute_pouring, args=(arm,), daemon=True).start()

        elif phase in ("upright_right", "upright_left"):
            if self._is_planning:
                self.get_logger().warn(f"Phase={phase} requested while planning — ignoring.")
                return
            arm = "right" if phase == "upright_right" else "left"
            self.get_logger().info(
                f"Phase: UPRIGHT ({arm} arm) → open-loop return to pour-hold pose"
            )
            threading.Thread(target=self._execute_upright, args=(arm,), daemon=True).start()

        elif phase in ("home_right", "home_left"):
            if self._is_planning:
                self.get_logger().warn(f"Phase={phase} requested while planning — ignoring.")
                return
            arm = "right" if phase == "home_right" else "left"
            self.get_logger().info(
                f"Phase: HOME ({arm} arm) → open-loop drive to all-zero joint position"
            )
            threading.Thread(target=self._execute_home, args=(arm,), daemon=True).start()

        else:
            self.get_logger().warn(f"Unknown phase '{phase}' — ignoring.")

    # ──────────────────────────────────────────────────────────────────────────
    # Planning
    # ──────────────────────────────────────────────────────────────────────────

    def _get_start(self):
        """Return (CuJointState, np.ndarray[14]) for the current robot state.

        Runs on the planning thread. The CUDA tensor is built here (not in the
        ROS callback) so that all CUDA work stays on a single thread and never
        races with CUDA-graph capture/replay during planning.

        The raw /joint_states can sit a few mrad outside the joint limits (the
        elbow joints rest exactly on their 0.0 bound, and encoder noise tips them
        over), which makes cuRobo reject the start as infeasible. We clamp into
        [lower+eps, upper-eps] here so every downstream use — the planning start
        state, the idle/held-arm slices, and the cspace goal built from these
        positions — is guaranteed feasible. The correction is sub-resolution, so
        the arm does not move meaningfully.
        """
        with self._lock:
            pos_np = self._current_pos_np.copy() if self._current_pos_np is not None else None
        if pos_np is None:
            self.get_logger().warn("No /joint_states yet — using robot default.")
            pos_np = self._default_pos_np.copy()
        clamped = np.clip(pos_np, self._clamp_lower, self._clamp_upper)
        if not np.array_equal(clamped, pos_np):
            moved = np.flatnonzero(clamped != pos_np)
            self.get_logger().info(
                "Clamped start joints to limits: "
                + ", ".join(
                    f"{self.joint_names[i]} {pos_np[i]:.4f}->{clamped[i]:.4f}"
                    for i in moved
                )
            )
        pos_np = clamped.astype(np.float32)
        js = CuJointState.from_position(
            torch.as_tensor(pos_np, device="cuda", dtype=torch.float32).unsqueeze(0),
            joint_names=self.joint_names,
        )
        return js, pos_np

    def _plan_single(self, arm: str, px, py, pz, qw, qx, qy, qz):
        """
        Plan one arm. The other arm's joints are frozen at their current angles.
        Publishes a 14-DOF trajectory to dual_arm_controller.
        """
        self._is_planning = True
        self._publish_status("planning")
        plan_ok = False
        try:
            start_js, pos_np = self._get_start()
            self.get_logger().info(
                f"Start JS positions: {start_js.position.cpu().numpy().tolist()}"
            )

            # Idle arm slice from the 14-DOF array (left=0:7, right=7:14)
            idle_pos = pos_np[0:7] if arm == "right" else pos_np[7:14]

            # Identify active and idle tool frame names
            active_link = self.left_tool[0]  if arm == "left"  else self.right_tool[0]
            idle_link   = self.right_tool[0] if arm == "left"  else self.left_tool[0]

            # Compute current FK pose of the idle arm so the planner can hold it still.
            # get_active_js strips any padding; compute_kinematics returns tool poses keyed
            # by link name — same pattern as inverse_kinematics.py lines 375-376.
            active_js  = self.planner.kinematics.get_active_js(start_js.clone())
            kin_state  = self.planner.kinematics.compute_kinematics(active_js)
            idle_fk    = kin_state.tool_poses[idle_link]   # Pose with .position / .quaternion

            # Build goal dict keyed by link name — the correct API (see inverse_kinematics.py:83-93)
            # Active arm: commanded target pose
            # Idle arm:   current FK pose → planner holds it in place
            from curobo.types import Pose as CuPose
            goal_poses = {
                active_link: CuPose(
                    position=torch.tensor([[px, py, pz]], device="cuda", dtype=torch.float32),
                    quaternion=torch.tensor([[qw, qx, qy, qz]], device="cuda", dtype=torch.float32),
                ),
                idle_link: CuPose(
                    position=idle_fk.position.reshape(1, 3),
                    quaternion=idle_fk.quaternion.reshape(1, 4),
                ),
            }

            goal = GoalToolPose.from_poses(
                goal_poses,
                ordered_tool_frames=list(self.planner.tool_frames),  # ['LWY_link', 'RWY_link']
                num_goalset=1,
            )

            self.get_logger().info(
                f"Planning {arm} arm ({active_link}) → "
                f"pos=[{px:.4f},{py:.4f},{pz:.4f}] "
                f"quat(wxyz)=[{qw:.3f},{qx:.3f},{qy:.3f},{qz:.3f}] | "
                f"idle: {idle_link} held at FK pose"
            )

            self.get_logger().info(
                f"Disabling pose cost on idle tool frame {idle_link} "
                f"(soft constraint — its trajectory is overwritten to hold still anyway)"
            )
            self.planner.update_tool_pose_criteria({
                active_link: ToolPoseCriteria.track_position_and_orientation(
                    xyz=[1.0, 1.0, 1.0], rpy=[1.0, 0.05, 0.05],
                ),
                idle_link: ToolPoseCriteria.disabled(),
            })

            t0 = time.perf_counter()
            result = self.planner.plan_pose(goal, start_js, max_attempts=MAX_ATTEMPTS)
            elapsed = time.perf_counter() - t0

            # NOTE: deliberately NO retry from the default seed/start. Re-planning
            # from the home config emits a trajectory that jumps the arm to home
            # first — unsafe (the arm may be holding a payload). Fail and hold
            # position instead.
            if result is None or not result.success.any():
                self.get_logger().error(f"{arm} arm planning FAILED ({elapsed:.2f}s)")
                self._publish_status("failed")
                return

            interpolated = result.get_interpolated_plan()
            n_wp = interpolated.position.shape[-2]
            idle_arm = "right" if arm == "left" else "left"
            self.get_logger().info(
                f"{arm} arm SUCCEEDED | {elapsed:.2f}s | {n_wp} waypoints | "
                f"{n_wp * self.interp_dt:.2f}s | {idle_arm} arm holds position"
            )

            traj = _make_single_arm_traj(
                active_interp=interpolated,
                idle_pos_np=idle_pos,
                active_is_left=(arm == "left"),
                dt=self.interp_dt,
                clock=self.get_clock(),
            )
            self._dual_pub.publish(traj)
            self.get_logger().info("Trajectory published to dual_arm_controller.")

            settle_time = n_wp * self.interp_dt + 1.0
            self.get_logger().info(f"Waiting {settle_time:.2f}s for robot to settle…")
            time.sleep(settle_time)
            plan_ok = True

        except Exception as e:
            self.get_logger().error(f"Error planning {arm} arm: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self.planner.update_tool_pose_criteria({
                self.left_tool[0]: ToolPoseCriteria.track_position_and_orientation(
                    xyz=[1.0, 1.0, 1.0], rpy=[1.0, 0.05, 0.05],
                ),
                self.right_tool[0]: ToolPoseCriteria.track_position_and_orientation(
                    xyz=[1.0, 1.0, 1.0], rpy=[1.0, 0.05, 0.05],
                ),
            })
            self._is_planning = False

        # Publish terminal status after the gate is cleared and criteria restored.
        if plan_ok:
            self._publish_status("success")

    def _plan_dual(self,
                   lpx, lpy, lpz, lqw, lqx, lqy, lqz,
                   rpx, rpy, rpz, rqw, rqx, rqy, rqz):
        """Plan both arms simultaneously — full 14-DOF cuRobo plan."""
        self._is_planning = True
        self._publish_status("planning")
        plan_ok = False
        try:
            start_js, _ = self._get_start()
            self.get_logger().info(
                f"Start JS positions: {start_js.position.cpu().numpy().tolist()}"
            )

            from curobo.types import Pose as CuPose
            left_link, right_link = self.left_tool[0], self.right_tool[0]

            goal = GoalToolPose.from_poses(
                {
                    left_link: CuPose(
                        position=torch.tensor([[lpx, lpy, lpz]], device="cuda", dtype=torch.float32),
                        quaternion=torch.tensor([[lqw, lqx, lqy, lqz]], device="cuda", dtype=torch.float32),
                    ),
                    right_link: CuPose(
                        position=torch.tensor([[rpx, rpy, rpz]], device="cuda", dtype=torch.float32),
                        quaternion=torch.tensor([[rqw, rqx, rqy, rqz]], device="cuda", dtype=torch.float32),
                    ),
                },
                ordered_tool_frames=list(self.planner.tool_frames),
                num_goalset=1,
            )

            self.get_logger().info(
                f"Planning DUAL | "
                f"L=[{lpx:.4f},{lpy:.4f},{lpz:.4f}] "
                f"R=[{rpx:.4f},{rpy:.4f},{rpz:.4f}]"
            )

            t0 = time.perf_counter()
            result = self.planner.plan_pose(goal, start_js, max_attempts=MAX_ATTEMPTS)
            elapsed = time.perf_counter() - t0

            # NOTE: deliberately NO retry from the default seed/start. Re-planning
            # from the home config emits a trajectory that jumps the arms to home
            # first — unsafe (an arm may be holding a payload). Fail and hold
            # position instead.
            if result is None or not result.success.any():
                self.get_logger().error(f"Dual arm planning FAILED ({elapsed:.2f}s)")
                self._publish_status("failed")
                return

            interpolated = result.get_interpolated_plan()
            n_wp = interpolated.position.shape[-2]
            self.get_logger().info(
                f"Dual arm SUCCEEDED | {elapsed:.2f}s | {n_wp} waypoints | "
                f"{n_wp * self.interp_dt:.2f}s"
            )

            traj = _make_dual_arm_traj(interpolated, self.interp_dt, self.get_clock())
            self._dual_pub.publish(traj)
            self.get_logger().info("Dual arm trajectory published.")

            settle_time = n_wp * self.interp_dt + 1.0
            self.get_logger().info(f"Waiting {settle_time:.2f}s for robot to settle…")
            time.sleep(settle_time)
            plan_ok = True

        except Exception as e:
            self.get_logger().error(f"Error planning dual arm: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self._is_planning = False

        # Publish terminal status after the gate is cleared.
        if plan_ok:
            self._publish_status("success")

    def _trim_trajectory(self, interpolated_traj, last_tstep):
        """Trim padded waypoints from interpolated trajectory."""
        if last_tstep is None:
            return interpolated_traj
        last_idx = int(last_tstep.item()) + 1
        return CuJointState(
            position=interpolated_traj.position[..., :last_idx, :],
            velocity=interpolated_traj.velocity[..., :last_idx, :] if interpolated_traj.velocity is not None else None,
            acceleration=interpolated_traj.acceleration[..., :last_idx, :] if interpolated_traj.acceleration is not None else None,
        )

    def _plan_pour(self, arm: str):
        """
        Joint-space pour. Drives the active arm to a fixed pour configuration
        via plan_cspace (configuration-space planning), holding the idle arm at
        its current angles.

        plan_cspace returns a TrajOptSolverResult (not the pose-plan result type),
        so the trajectory is taken from result.interpolated_trajectory /
        result.interpolated_last_tstep and trimmed with _trim_trajectory().
        Unlike plan_pose, cspace planning has no per-tool pose criteria, so there
        is no enable/disable dance to perform here.
        """
        self._is_planning = True
        self._publish_status("planning")
        pour_ok = False
        try:
            # Resolve the pour target for the requested arm.
            if arm == "right":
                pour_joints = POUR_RIGHT_JOINTS
            else:
                pour_joints = POUR_LEFT_JOINTS
            if pour_joints is None:
                self.get_logger().error(
                    f"No pour configuration defined for {arm} arm "
                    f"(POUR_{arm.upper()}_JOINTS is None) — skipping."
                )
                self._publish_status("failed")
                return

            start_js, pos_np = self._get_start()
            self.get_logger().info(
                f"Start JS positions: {start_js.position.cpu().numpy().tolist()}"
            )

            # Idle arm slice from the 14-DOF array (left=0:7, right=7:14)
            idle_pos = pos_np[0:7] if arm == "right" else pos_np[7:14]

            # Build a full 14-DOF goal: hold both arms at current angles, then
            # overwrite the active arm's 7-slice with the pour configuration.
            goal_pos = pos_np.astype(np.float32).copy()
            if arm == "right":
                goal_pos[7:14] = pour_joints
            else:
                goal_pos[0:7] = pour_joints

            goal_js = CuJointState.from_position(
                torch.as_tensor(goal_pos, device="cuda", dtype=torch.float32).unsqueeze(0),
                joint_names=self.joint_names,
            )

            self.get_logger().info(
                f"Planning POUR ({arm} arm) cspace → target 7-DOF={pour_joints.tolist()}"
            )

            t0 = time.perf_counter()
            result = self.planner.plan_cspace(
                goal_js, start_js,
                max_attempts=MAX_ATTEMPTS, enable_graph_attempt=1,
            )
            elapsed = time.perf_counter() - t0

            # NOTE: deliberately NO retry from the default seed/start here. The
            # arm is holding a glass at this point, so re-planning from the home
            # config would emit a trajectory that jumps the arm to home first —
            # unsafe. If planning from the real current state fails, we fail the
            # phase and hold position rather than risk the payload.
            if result is None or not result.success.any():
                self.get_logger().error(f"{arm} pour cspace FAILED ({elapsed:.2f}s)")
                self._publish_status("failed")
                return

            raw_n_wp = result.interpolated_trajectory.position.shape[-2]
            pour_traj = self._trim_trajectory(
                result.interpolated_trajectory, result.interpolated_last_tstep
            )
            n_wp = pour_traj.position.shape[-2]
            self.get_logger().info(
                f"{arm} pour SUCCEEDED | {elapsed:.2f}s | "
                f"trimmed {raw_n_wp}→{n_wp} waypoints | {n_wp * self.interp_dt:.2f}s"
            )

            traj = _make_single_arm_traj(
                active_interp=pour_traj,
                idle_pos_np=idle_pos,
                active_is_left=(arm == "left"),
                dt=self.interp_dt,
                clock=self.get_clock(),
            )
            self._dual_pub.publish(traj)
            self.get_logger().info("Pour trajectory published to dual_arm_controller.")

            settle_time = n_wp * self.interp_dt + 1.0
            self.get_logger().info(f"Waiting {settle_time:.2f}s for robot to settle…")
            time.sleep(settle_time)
            pour_ok = True

        except Exception as e:
            self.get_logger().error(f"Error planning pour for {arm} arm: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self._is_planning = False

        # Publish terminal status after the gate is cleared.
        if pour_ok:
            self._publish_status("pour_done")

    def _execute_pouring(self, arm: str):
        """
        Open-loop POURING tilt — the actual pour motion.

        Runs after the pourer arm is already at its pour-hold pose. Publishes a
        SINGLE-waypoint 14-DOF trajectory: the pourer arm's 7 joints set to the
        fixed POUR_*_TILT_JOINTS, every other joint left at its current angle.
        The JointTrajectoryController interpolates from the current pose to this
        target over POURING_DURATION_SEC, giving a smooth, fully deterministic
        tilt every run (both endpoints are fixed — no planning, no CUDA).

        Crucially the catcher arm is held at its CURRENT angles (read live from
        /joint_states), not commanded to zero, so the cup it is holding stays in
        place while the other arm pours.
        """
        self._is_planning = True
        self._publish_status("executing")
        pouring_ok = False
        try:
            tilt = POUR_RIGHT_TILT_JOINTS if arm == "right" else POUR_LEFT_TILT_JOINTS
            if tilt is None:
                self.get_logger().error(
                    f"No pouring tilt configuration defined for {arm} arm "
                    f"(POUR_{arm.upper()}_TILT_JOINTS is None) — skipping."
                )
                self._publish_status("failed")
                return

            # Current 14-DOF angles (hold everything, then overwrite the pourer).
            with self._lock:
                pos_np = (
                    self._current_pos_np.copy()
                    if self._current_pos_np is not None
                    else None
                )
            if pos_np is None:
                self.get_logger().warn("No /joint_states yet — using robot default.")
                pos_np = self._default_pos_np.copy()

            goal_pos = pos_np.astype(np.float64).copy()
            if arm == "right":
                goal_pos[7:14] = tilt
            else:
                goal_pos[0:7] = tilt

            self.get_logger().info(
                f"Pouring ({arm} arm) → tilt 7-DOF={np.asarray(tilt).tolist()} | "
                f"catcher arm held at current angles | "
                f"controller interpolates over {POURING_DURATION_SEC:.1f}s"
            )

            msg = JointTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names  = ALL_JOINT_NAMES
            pt = JointTrajectoryPoint()
            pt.positions     = goal_pos.tolist()
            pt.velocities    = [0.0] * len(ALL_JOINT_NAMES)
            pt.time_from_start = Duration(
                sec=int(POURING_DURATION_SEC),
                nanosec=int(round((POURING_DURATION_SEC % 1.0) * 1e9)),
            )
            msg.points.append(pt)
            self._dual_pub.publish(msg)
            self.get_logger().info("Pouring trajectory published to dual_arm_controller.")

            settle_time = POURING_DURATION_SEC + 1.0
            self.get_logger().info(f"Waiting {settle_time:.2f}s for robot to settle…")
            time.sleep(settle_time)
            pouring_ok = True

        except Exception as e:
            self.get_logger().error(f"Error executing pouring for {arm} arm: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self._is_planning = False

        # Publish terminal status after the gate is cleared.
        if pouring_ok:
            self._publish_status("pouring_done")

    def _execute_upright(self, arm: str):
        """
        Open-loop UPRIGHT return — after pouring, move the pourer arm back to
        its pour-hold pose (POUR_RIGHT_JOINTS / POUR_LEFT_JOINTS).

        Same open-loop single-waypoint pattern as _execute_pouring but targets
        the pour-hold pose rather than the tilt pose. The idle arm is held at
        its current /joint_states angles so the cup it is carrying stays put.
        Publishes "upright_done" on completion.
        """
        self._is_planning = True
        self._publish_status("executing")
        upright_ok = False
        try:
            hold = POUR_RIGHT_JOINTS if arm == "right" else POUR_LEFT_JOINTS
            if hold is None:
                self.get_logger().error(
                    f"No pour configuration defined for {arm} arm "
                    f"(POUR_{arm.upper()}_JOINTS is None) — skipping."
                )
                self._publish_status("failed")
                return

            with self._lock:
                pos_np = (
                    self._current_pos_np.copy()
                    if self._current_pos_np is not None
                    else None
                )
            if pos_np is None:
                self.get_logger().warn("No /joint_states yet — using robot default.")
                pos_np = self._default_pos_np.copy()

            goal_pos = pos_np.astype(np.float64).copy()
            if arm == "right":
                goal_pos[7:14] = hold
            else:
                goal_pos[0:7] = hold

            self.get_logger().info(
                f"Upright ({arm} arm) → pour-hold 7-DOF={np.asarray(hold).tolist()} | "
                f"idle arm held at current angles | "
                f"controller interpolates over {POURING_DURATION_SEC:.1f}s"
            )

            msg = JointTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names  = ALL_JOINT_NAMES
            pt = JointTrajectoryPoint()
            pt.positions     = goal_pos.tolist()
            pt.velocities    = [0.0] * len(ALL_JOINT_NAMES)
            pt.time_from_start = Duration(
                sec=int(POURING_DURATION_SEC),
                nanosec=int(round((POURING_DURATION_SEC % 1.0) * 1e9)),
            )
            msg.points.append(pt)
            self._dual_pub.publish(msg)
            self.get_logger().info("Upright trajectory published to dual_arm_controller.")

            settle_time = POURING_DURATION_SEC + 1.0
            self.get_logger().info(f"Waiting {settle_time:.2f}s for robot to settle…")
            time.sleep(settle_time)
            upright_ok = True

        except Exception as e:
            self.get_logger().error(f"Error executing upright for {arm} arm: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self._is_planning = False

        if upright_ok:
            self._publish_status("upright_done")

    def _execute_home(self, arm: str):
        """
        Open-loop HOME — drive the active arm to all-zero joint positions
        [0, 0, 0, 0, 0, 0, 0]. Idle arm held at its current /joint_states
        angles. Single waypoint interpolated over HOME_DURATION_SEC.
        Publishes "home_done" on completion.
        """
        self._is_planning = True
        self._publish_status("executing")
        home_ok = False
        try:
            home_joints = np.zeros(7, dtype=np.float64)

            with self._lock:
                pos_np = (
                    self._current_pos_np.copy()
                    if self._current_pos_np is not None
                    else None
                )
            if pos_np is None:
                self.get_logger().warn("No /joint_states yet — using robot default.")
                pos_np = self._default_pos_np.copy()

            goal_pos = pos_np.astype(np.float64).copy()
            if arm == "right":
                goal_pos[7:14] = home_joints
            else:
                goal_pos[0:7] = home_joints

            self.get_logger().info(
                f"Home ({arm} arm) → all-zero joints | "
                f"idle arm held at current angles | "
                f"controller interpolates over {HOME_DURATION_SEC:.1f}s"
            )

            msg = JointTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names  = ALL_JOINT_NAMES
            pt = JointTrajectoryPoint()
            pt.positions     = goal_pos.tolist()
            pt.velocities    = [0.0] * len(ALL_JOINT_NAMES)
            pt.time_from_start = Duration(
                sec=int(HOME_DURATION_SEC),
                nanosec=int(round((HOME_DURATION_SEC % 1.0) * 1e9)),
            )
            msg.points.append(pt)
            self._dual_pub.publish(msg)
            self.get_logger().info("Home trajectory published to dual_arm_controller.")

            settle_time = HOME_DURATION_SEC + 1.0
            self.get_logger().info(f"Waiting {settle_time:.2f}s for robot to settle…")
            time.sleep(settle_time)
            home_ok = True

        except Exception as e:
            self.get_logger().error(f"Error executing home for {arm} arm: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self._is_planning = False

        if home_ok:
            self._publish_status("home_done")

    def _plan_grasp_single(self, arm: str, px, py, pz, qw, qx, qy, qz):
        """
        Plan a grasp (approach → grasp → lift) for one arm via cuRobo's
        plan_grasp(). Both tool frames are included in the goal — the idle
        arm is pinned to its current FK pose (planner requires goal_tool_poses
        to cover every configured tool frame; see plan_pose → solve_pose →
        GoalToolPose.reorder_links, which raises if the planner's tool_frames
        is not a subset of the goal's tool_frames).

        On success, caches the GraspPlanResult and immediately executes the
        approach phase. The grasp and lift phases are executed later on
        request via /curobo/execute_phase (see _phase_cb).
        """
        self._is_planning = True
        self._publish_status("planning")
        approach_ok = False
        try:
            start_js, pos_np = self._get_start()
            self.get_logger().info(
                f"Start JS positions: {start_js.position.cpu().numpy().tolist()}"
            )

            # Idle arm slice from the 14-DOF array (left=0:7, right=7:14)
            idle_pos = pos_np[0:7] if arm == "right" else pos_np[7:14]

            # Identify active and idle tool frame names
            active_link = self.left_tool[0]  if arm == "left"  else self.right_tool[0]
            idle_link   = self.right_tool[0] if arm == "left"  else self.left_tool[0]

            # Compute current FK pose of the idle arm so the planner can hold it still.
            active_js  = self.planner.kinematics.get_active_js(start_js.clone())
            kin_state  = self.planner.kinematics.compute_kinematics(active_js)
            idle_fk    = kin_state.tool_poses[idle_link]   # Pose with .position / .quaternion

            from curobo.types import Pose as CuPose
            goal_poses = {
                active_link: CuPose(
                    position=torch.tensor([[px, py, pz]], device="cuda", dtype=torch.float32),
                    quaternion=torch.tensor([[qw, qx, qy, qz]], device="cuda", dtype=torch.float32),
                ),
                idle_link: CuPose(
                    position=idle_fk.position.reshape(1, 3),
                    quaternion=idle_fk.quaternion.reshape(1, 4),
                ),
            }
            grasp_poses = GoalToolPose.from_poses(
                goal_poses,
                ordered_tool_frames=list(self.planner.tool_frames),
                num_goalset=1,
            )

            self.get_logger().info(
                f"Planning GRASP for {arm} arm ({active_link}) → "
                f"pos=[{px:.4f},{py:.4f},{pz:.4f}] "
                f"quat(wxyz)=[{qw:.3f},{qx:.3f},{qy:.3f},{qz:.3f}]"
            )

            self.get_logger().info(
                f"Disabling pose cost on idle tool frame {idle_link} "
                f"(soft constraint — its trajectory is overwritten to hold still anyway). "
                f"Note: plan_grasp swaps in linear_motion criteria for {active_link} during "
                f"its approach/lift phases and restores standard criteria afterward — our "
                f"disabled() on {idle_link} stays in effect across all of that."
            )
            self.planner.update_tool_pose_criteria({
                active_link: ToolPoseCriteria.track_position_and_orientation(
                    xyz=[1.0, 1.0, 1.0], rpy=[1.0, 0.05, 0.05],
                ),
                idle_link: ToolPoseCriteria.disabled(),
            })

            # approach is along base-frame X (robot forward axis), lift is along base-frame Z (straight up)
            t0 = time.perf_counter()
            result = self.planner.plan_grasp(
                grasp_poses,
                start_js,
                grasp_approach_axis="x",
                grasp_approach_offset=-0.05,           # 5cm pullback in -X (toward robot), then linear move in +X to grasp
                grasp_approach_in_tool_frame=False,     # offset in base/world frame, not tool frame
                grasp_lift_axis="z",
                grasp_lift_offset=0.10,                # 10cm lift straight up
                grasp_lift_in_tool_frame=False,         # lift in world frame
                plan_grasp_to_lift=True,
                grasp_frames=[active_link],
            )
            elapsed = time.perf_counter() - t0

            # NOTE: deliberately NO retry from the default seed/start. Re-planning
            # from the home config emits a trajectory that jumps the arm to home
            # first — unsafe (the arm may be holding a payload). Fail and hold
            # position instead.
            if result is None or not result.approach_success.any():
                self.get_logger().error(f"{arm} arm grasp planning FAILED ({elapsed:.2f}s)")
                self._publish_status("failed")
                return

            # Validate that approach and grasp trajectories are both present — they are required.
            # Lift is optional: plan_grasp_to_lift can succeed on approach+grasp but fail on lift.
            approach_traj_ok = result.approach_interpolated_trajectory is not None
            grasp_traj_ok    = result.grasp_interpolated_trajectory is not None
            lift_traj_ok     = result.lift_interpolated_trajectory is not None

            self.get_logger().info(
                f"{arm} arm grasp plan result | {elapsed:.2f}s | status={result.status} | "
                f"approach={'ok' if approach_traj_ok else 'MISSING'} | "
                f"grasp={'ok' if grasp_traj_ok else 'MISSING'} | "
                f"lift={'ok' if lift_traj_ok else 'missing'}"
            )

            if not approach_traj_ok or not grasp_traj_ok:
                self.get_logger().error(
                    f"{arm} arm grasp planning FAILED — "
                    f"approach_traj={'ok' if approach_traj_ok else 'MISSING'}, "
                    f"grasp_traj={'ok' if grasp_traj_ok else 'MISSING'}"
                )
                self._publish_status("failed")
                return

            if not lift_traj_ok:
                self.get_logger().warn(
                    "Lift phase unavailable (planning failed) — will grasp but not lift."
                )

            # Cache the full grasp plan for later phases (grasp, lift)
            self._grasp_cache    = result
            self._grasp_idle_pos = idle_pos
            self._grasp_arm      = arm
            self._lift_available = lift_traj_ok

            # Execute the approach phase immediately
            approach_interp = result.approach_interpolated_trajectory
            raw_n_wp = approach_interp.position.shape[-2]
            approach_traj = self._trim_trajectory(
                approach_interp, result.approach_interpolated_last_tstep
            )
            n_wp = approach_traj.position.shape[-2]
            self.get_logger().info(
                f"Approach trajectory trimmed from {raw_n_wp} to {n_wp} waypoints."
            )

            traj = _make_single_arm_traj(
                active_interp=approach_traj,
                idle_pos_np=idle_pos,
                active_is_left=(arm == "left"),
                dt=self.interp_dt,
                clock=self.get_clock(),
            )
            self._dual_pub.publish(traj)
            self.get_logger().info("Approach trajectory published to dual_arm_controller.")

            settle_time = n_wp * self.interp_dt + 1.0
            self.get_logger().info(f"Waiting {settle_time:.2f}s for robot to settle…")
            time.sleep(settle_time)
            approach_ok = True

        except Exception as e:
            self.get_logger().error(f"Error planning grasp for {arm} arm: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self.planner.update_tool_pose_criteria({
                self.left_tool[0]: ToolPoseCriteria.track_position_and_orientation(
                    xyz=[1.0, 1.0, 1.0], rpy=[1.0, 0.05, 0.05],
                ),
                self.right_tool[0]: ToolPoseCriteria.track_position_and_orientation(
                    xyz=[1.0, 1.0, 1.0], rpy=[1.0, 0.05, 0.05],
                ),
            })
            self._is_planning = False

        # Publish the handoff signal only AFTER _is_planning is cleared and the
        # default pose criteria are restored, so the TaskManager's follow-up
        # "grasp" phase request isn't rejected by the _is_planning gate.
        if approach_ok:
            self._publish_status("approach_done")

    def _execute_grasp_phase(self):
        """Execute the cached grasp (closing) phase of the grasp plan."""
        self._is_planning = True
        self._publish_status("executing")
        grasp_ok = False
        try:
            grasp_interp = self._grasp_cache.grasp_interpolated_trajectory
            raw_n_wp = grasp_interp.position.shape[-2]
            grasp_traj = self._trim_trajectory(
                grasp_interp, self._grasp_cache.grasp_interpolated_last_tstep
            )
            n_wp = grasp_traj.position.shape[-2]
            self.get_logger().info(
                f"Grasp trajectory trimmed from {raw_n_wp} to {n_wp} waypoints."
            )

            traj = _make_single_arm_traj(
                active_interp=grasp_traj,
                idle_pos_np=self._grasp_idle_pos,
                active_is_left=(self._grasp_arm == "left"),
                dt=self.interp_dt,
                clock=self.get_clock(),
            )
            self._dual_pub.publish(traj)
            self.get_logger().info("Grasp trajectory published to dual_arm_controller.")

            settle_time = n_wp * self.interp_dt + 1.0
            self.get_logger().info(f"Waiting {settle_time:.2f}s for robot to settle…")
            time.sleep(settle_time)
            grasp_ok = True

        except Exception as e:
            self.get_logger().error(f"Error executing grasp phase: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self._is_planning = False

        # Signal completion only after the gate is cleared, so the follow-up
        # "lift" phase request isn't rejected by the _is_planning gate.
        if grasp_ok:
            self._publish_status("grasp_done")

    def _execute_lift_phase(self):
        """Execute the cached lift phase of the grasp plan, then clear the cache."""
        self._is_planning = True
        self._publish_status("executing")
        lift_ok = False
        try:
            if self._grasp_cache is None or self._grasp_cache.lift_interpolated_trajectory is None:
                self.get_logger().warn(
                    "No lift trajectory available — holding position, skipping lift."
                )
                lift_ok = True  # nothing to execute, but the phase is "done"
            else:
                lift_interp = self._grasp_cache.lift_interpolated_trajectory
                raw_n_wp = lift_interp.position.shape[-2]
                lift_traj = self._trim_trajectory(
                    lift_interp, self._grasp_cache.lift_interpolated_last_tstep
                )
                n_wp = lift_traj.position.shape[-2]
                self.get_logger().info(
                    f"Lift trajectory trimmed from {raw_n_wp} to {n_wp} waypoints."
                )

                traj = _make_single_arm_traj(
                    active_interp=lift_traj,
                    idle_pos_np=self._grasp_idle_pos,
                    active_is_left=(self._grasp_arm == "left"),
                    dt=self.interp_dt,
                    clock=self.get_clock(),
                )
                self._dual_pub.publish(traj)
                self.get_logger().info("Lift trajectory published to dual_arm_controller.")

                settle_time = n_wp * self.interp_dt + 1.0
                self.get_logger().info(f"Waiting {settle_time:.2f}s for robot to settle…")
                time.sleep(settle_time)
                lift_ok = True

        except Exception as e:
            self.get_logger().error(f"Error executing lift phase: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self._grasp_cache    = None
            self._grasp_idle_pos = None
            self._grasp_arm      = None
            self._lift_available = False
            self._is_planning    = False

        # Signal completion only after the gate is cleared (consistent with the
        # other phases), so any follow-up phase request isn't spuriously rejected.
        if lift_ok:
            self._publish_status("lift_done")

    # ──────────────────────────────────────────────────────────────────────────

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)


# ──────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = MinibotCuRoboNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down cuRobo minibot node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
cuRobo MPC (Model Predictive Control) Motion Planning Node for mvp_arm
=======================================================================
Replaces the single-shot MotionPlanner with ModelPredictiveControl so the
robot continuously re-optimises toward the goal at every control tick instead
of executing a pre-computed trajectory once.

Architecture
------------
  • One-shot goal subscription  (/curobo/goal_pose)
      └─ Stores the goal; calls mpc.update_goal_tool_poses()  (run_ik=True first time)
  • Joint-state subscription     (/isaac_joint_states)
      └─ Keeps self._current_js up to date from Isaac Sim
  • MPC control timer            (fires every OPTIMIZATION_DT seconds)
      └─ Calls mpc.optimize_action_sequence(current_state)
      └─ Extracts next action → publishes a 1-point JointTrajectory command
      └─ Feeds action back as current_state (warm-start for next tick)
  • Status publisher             (/curobo/status)
      └─ "idle" | "active" | "converged" | "failed"

Key differences from the single-shot MotionPlanner node
---------------------------------------------------------
  Old node (MotionPlanner)                New node (MPC)
  ──────────────────────────────────────  ──────────────────────────────────────
  plan_pose()  →  full trajectory once    optimize_action_sequence() every tick
  Publishes long JointTrajectory          Publishes 1-point command each tick
  Planning in background thread           Control loop in a ROS timer callback
  Goal triggers planning, then done       Goal updates mpc; loop runs forever
  Cannot react to mid-motion goal change  New goal → seamless re-convergence

Subscriptions:
  /curobo/goal_pose    (geometry_msgs/PoseStamped)   — new target EE pose
  /isaac_joint_states  (sensor_msgs/JointState)      — robot feedback

Publishes:
  /mvp_arm_controller/joint_trajectory  (trajectory_msgs/JointTrajectory)
  /curobo/status                        (std_msgs/String)

Run:
    conda activate curobo
    python curobo_mpc_ros2_node.py

Send a goal (same command as before — interface is unchanged):
    ros2 topic pub --once /curobo/goal_pose geometry_msgs/msg/PoseStamped "{
      header: {frame_id: 'base_link'},
      pose: {
        position: {x: 0.0324, y: -0.2098, z: 0.8858},
        orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}
      }
    }"
"""

import sys

# ── ROS2 Python path (adjust if your ROS install differs) ─────────────────────
sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")
sys.path.insert(0, "/opt/ros/humble/local/lib/python3.10/dist-packages")

import threading
import time
from typing import Optional

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import JointState as RosJointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from builtin_interfaces.msg import Duration

# cuRobo — MPC API (from reactive_control.py)
from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg
from curobo.types import GoalToolPose, JointState as CuJointState, Pose


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit to match your setup
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_CFG   = "mvp_arm.yml"
SCENE_CFG   = "collision_test.yml"   # same scene used in interactive_mpc_example
TRAJ_TOPIC  = "/mvp_arm_controller/joint_trajectory"
JS_TOPIC    = "/isaac_joint_states"
GOAL_TOPIC  = "/curobo/goal_pose"
STATUS_TOPIC = "/curobo/status"

# MPC tuning — mirrors reactive_control.py / interactive_mpc_example()
OPTIMIZATION_DT    = 0.025   # seconds — MPC re-plans at this rate (40 Hz)
INTERPOLATION_STEPS = 4      # sub-steps between optimisation knots
COLLISION_ACTIVATION_DISTANCE = 0.03   # metres

# Convergence threshold: stop reporting "active", switch to "converged"
POSITION_ERROR_CONVERGED = 0.01   # metres

# Joint names must match what Isaac Sim publishes on JS_TOPIC
MVP_JOINT_NAMES = [
    "RSP_joint",
    "RSY_joint",
    "RSR_joint",
    "REP_joint",
    "RWR_joint",
    "RWP_joint",
    "RWY_joint",
]

# ─────────────────────────────────────────────────────────────────────────────


def _build_single_point_trajectory(
    position: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    dt: float,
    joint_names: list,
    clock,
) -> JointTrajectory:
    """
    Build a 1-point JointTrajectory from the MPC's next action.

    Stamping rules for streaming / servoing mode
    --------------------------------------------
    The ros2_control JointTrajectoryController interprets trajectories as:

        execute_time = header.stamp + point.time_from_start

    When header.stamp is non-zero (wall time) and time_from_start equals
    one MPC tick (25 ms), the execute_time is only ~25 ms in the future.
    ROS middleware + controller scheduling latency easily exceeds that,
    so the point arrives already in the past → rejected with:
        "Received trajectory … that ends in the past"

    Fix — zero header stamp + small lookahead:
      • header.stamp = 0  tells the controller to interpret time_from_start
        relative to when it *receives* the message (immediate mode), not
        relative to an absolute wall-clock origin.
      • time_from_start = dt + TRANSPORT_LOOKAHEAD gives the controller
        enough runway to execute the point on time even with latency.

    If your controller variant requires a non-zero stamp, switch to the
    alternative below: stamp = now(), time_from_start = TRANSPORT_LOOKAHEAD.

    Args:
        position:     (DOF,) tensor — joint positions for the next step.
        velocity:     (DOF,) tensor — joint velocities for the next step.
        acceleration: (DOF,) tensor — joint accelerations for the next step.
        dt:           MPC time step in seconds.
        joint_names:  Ordered list of joint name strings.
        clock:        ROS2 node clock (kept for signature compatibility).

    Returns:
        JointTrajectory with a single waypoint.
    """
    # Extra time added on top of dt so the point is never stale on arrival.
    # Tune down toward 0 if your controller is very low-latency; increase if
    # you still see "ends in the past" warnings.
    TRANSPORT_LOOKAHEAD = 0.05   # seconds — 2× the MPC tick is a safe start

    msg = JointTrajectory()
    # Zero stamp → controller uses receive-time as t=0 (immediate / streaming)
    msg.header.stamp.sec     = 0
    msg.header.stamp.nanosec = 0
    msg.joint_names = list(joint_names)

    total_dt = dt + TRANSPORT_LOOKAHEAD
    pt = JointTrajectoryPoint()
    pt.positions     = [float(v) for v in position.cpu().numpy()]
    pt.velocities    = [float(v) for v in velocity.cpu().numpy()]
    pt.accelerations = [float(v) for v in acceleration.cpu().numpy()]
    pt.time_from_start = Duration(
        sec=int(total_dt),
        nanosec=int(round((total_dt % 1.0) * 1e9)),
    )
    msg.points.append(pt)
    return msg


# ─────────────────────────────────────────────────────────────────────────────

class CuRoboMPCNode(Node):
    """
    ROS2 node that wraps cuRobo's ModelPredictiveControl.

    Lifecycle
    ---------
    1. __init__  — build MPC, run setup() from robot default config, start timer.
    2. _js_callback  — update self._current_state every joint-state message.
    3. _goal_callback  — call mpc.update_goal_tool_poses(); arm the control loop.
    4. _mpc_timer_cb  — fire every OPTIMIZATION_DT s; call optimize_action_sequence();
                        publish next joint command; feed back as new current_state.
    """

    def __init__(self):
        super().__init__("curobo_mpc_node")

        # ── 1. Build MPC (mirrors reactive_control.py / interactive_mpc_example) ──
        self.get_logger().info("Initialising cuRobo MPC solver…")

        cfg = ModelPredictiveControlCfg.create(
            robot=ROBOT_CFG,
            scene_model=SCENE_CFG,
            use_cuda_graph=True,
            optimization_dt=OPTIMIZATION_DT,
            interpolation_steps=INTERPOLATION_STEPS,
            optimizer_collision_activation_distance=COLLISION_ACTIVATION_DISTANCE,
        )
        self._mpc = ModelPredictiveControl(cfg)
        self._optimization_dt = OPTIMIZATION_DT
        self._joint_names = list(self._mpc.joint_names)

        # ── 2. Setup MPC from robot retract config (from reactive_control.py) ──
        #    Sets the initial warm-start state so the solver is ready immediately.
        _default_pos = self._mpc.default_joint_position.clone().unsqueeze(0)
        _init_state = CuJointState.from_position(
            _default_pos, joint_names=self._joint_names
        )
        _init_state.velocity     = torch.zeros_like(_init_state.position)
        _init_state.acceleration = torch.zeros_like(_init_state.position)
        self._mpc.setup(_init_state)   # warm-starts solver internals

        # ── 3. Internal state ─────────────────────────────────────────────────
        self._lock = threading.Lock()

        # Latest joint feedback — stored as plain Python lists by _js_callback,
        # converted to CUDA tensors by the timer thread only.
        # Starts as robot default until first feedback arrives.
        _default_pos_list = _default_pos.squeeze(0).tolist()
        _default_vel_list = [0.0] * len(self._joint_names)
        self._pending_js: tuple = (_default_pos_list, _default_vel_list)

        # Whether a goal has been set at least once
        self._goal_active = False

        # Track goal changes to know when to call update_goal_tool_poses()
        self._pending_goal: Optional[tuple] = None   # (px,py,pz, qw,qx,qy,qz)
        self._goal_changed = False

        self.get_logger().info(
            f"MPC ready  |  joints: {self._joint_names}  |  dt: {OPTIMIZATION_DT:.3f}s"
        )

        # ── 4. ROS2 pub / sub ─────────────────────────────────────────────────
        # Use separate callback groups:
        #   • timer runs in its own group so it never blocks on I/O callbacks
        _io_cb_group    = ReentrantCallbackGroup()
        _timer_cb_group = MutuallyExclusiveCallbackGroup()

        self._js_sub = self.create_subscription(
            RosJointState, JS_TOPIC, self._js_callback, 10,
            callback_group=_io_cb_group,
        )
        self._goal_sub = self.create_subscription(
            PoseStamped, GOAL_TOPIC, self._goal_callback, 10,
            callback_group=_io_cb_group,
        )

        self._traj_pub   = self.create_publisher(JointTrajectory, TRAJ_TOPIC, 10)
        self._status_pub = self.create_publisher(String, STATUS_TOPIC, 10)

        # MPC control timer — fires at OPTIMIZATION_DT (e.g. 40 Hz)
        self._mpc_timer = self.create_timer(
            OPTIMIZATION_DT, self._mpc_timer_cb,
            callback_group=_timer_cb_group,
        )

        self._publish_status("idle")
        self.get_logger().info(
            f"\n"
            f"  Listening for goals on : {GOAL_TOPIC}\n"
            f"  Joint states from      : {JS_TOPIC}\n"
            f"  Publishing commands to : {TRAJ_TOPIC}\n"
            f"  Status on              : {STATUS_TOPIC}\n"
            f"  MPC rate               : {1.0/OPTIMIZATION_DT:.0f} Hz"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Subscription callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _js_callback(self, msg: RosJointState):
        """
        Update current joint feedback from Isaac Sim.

        IMPORTANT: Must NOT allocate any CUDA tensors.  This callback runs on
        a ROS I/O thread that may be concurrent with the MPC timer thread.
        Any torch.tensor(..., device='cuda') call here can race with a CUDA
        graph capture in the timer, causing cudaErrorStreamCaptureUnsupported.

        Fix: store plain Python lists under the lock.  The timer thread
        converts them to CUDA tensors at the start of each tick.
        """
        try:
            positions  = [msg.position[msg.name.index(n)] for n in self._joint_names]
            velocities = (
                [msg.velocity[msg.name.index(n)] for n in self._joint_names]
                if len(msg.velocity) == len(self._joint_names)
                else [0.0] * len(self._joint_names)
            )
            with self._lock:
                # Plain Python lists — zero CUDA involvement
                self._pending_js = (positions, velocities)

        except ValueError as e:
            self.get_logger().warn(
                f"Joint name mismatch in JS callback: {e}",
                throttle_duration_sec=5.0,
            )

    def _goal_callback(self, msg: PoseStamped):
        """
        Receive a new EE goal pose and arm the MPC control loop.

        IMPORTANT: This callback must NOT call any cuRobo / CUDA code directly.
        The MPC timer (_mpc_timer_cb) runs on a separate thread and may be in
        the middle of a CUDA graph capture when this callback fires.  Calling
        update_goal_tool_poses() (which internally builds CUDA graphs via IK)
        from a concurrent thread invalidates the in-progress capture and causes
        `cudaErrorStreamCaptureInvalidated`.

        Fix: store the raw goal data under the lock and let _mpc_timer_cb pick
        it up at the start of the next tick, where all CUDA work is serialized
        on the timer thread.
        """
        p = msg.pose.position
        q = msg.pose.orientation  # geometry_msgs: x y z w

        self.get_logger().info(
            f"New goal received  "
            f"pos=[{p.x:.4f}, {p.y:.4f}, {p.z:.4f}]  "
            f"quat(wxyz)=[{q.w:.3f}, {q.x:.3f}, {q.y:.3f}, {q.z:.3f}]"
        )

        # Store raw floats only — no CUDA tensors, no cuRobo objects.
        # _mpc_timer_cb will convert these and call update_goal_tool_poses().
        with self._lock:
            self._pending_goal = (p.x, p.y, p.z, q.w, q.x, q.y, q.z)
            self._goal_changed = True

        self._publish_status("active")

    # ─────────────────────────────────────────────────────────────────────────
    # MPC control timer — the heart of the node
    # ─────────────────────────────────────────────────────────────────────────

    def _mpc_timer_cb(self):
        """
        Core MPC control loop — fires every OPTIMIZATION_DT seconds.

        Implements exactly the loop from reactive_control.py Step 4:

            result = mpc.optimize_action_sequence(current_state)
            next_position = result.action_sequence.position[:, -1, :]
            current_state = JointState(next_position, vel, acc)

        The key difference from reactive_control.py is that instead of
        collecting trajectory_positions for a plot, we publish a 1-point
        JointTrajectory command to the robot controller on every tick.

        Because the solver warm-starts from the previous solution,
        consecutive calls naturally produce smooth, consistent motion.

        All cuRobo / CUDA work (including update_goal_tool_poses) happens here
        on the timer thread so that CUDA graph operations are never concurrent
        with each other.  _goal_callback only writes plain Python data under
        the lock; this method reads and processes it.
        """
        with self._lock:
            goal_active    = self._goal_active
            goal_changed   = self._goal_changed
            pending_goal   = self._pending_goal
            pending_js     = self._pending_js          # plain Python lists
            self._goal_changed = False   # consume the flag

        # ── Build CuJointState from latest feedback (CUDA only on this thread) ─
        pos_list, vel_list = pending_js
        current_state = CuJointState.from_position(
            torch.tensor([pos_list], device="cuda", dtype=torch.float32),
            joint_names=self._joint_names,
        )
        current_state.velocity     = torch.tensor([vel_list], device="cuda", dtype=torch.float32)
        current_state.acceleration = torch.zeros_like(current_state.position)

        # ── Apply a pending goal update (all CUDA work on this thread) ────────
        if goal_changed and pending_goal is not None:
            px, py, pz, qw, qx, qy, qz = pending_goal

            target_pose = Pose(
                position=torch.tensor(
                    [[px, py, pz]], device="cuda", dtype=torch.float32
                ),
                quaternion=torch.tensor(
                    [[qw, qx, qy, qz]], device="cuda", dtype=torch.float32
                ),
            )

            goal_poses = {self._mpc.tool_frames[0]: target_pose}
            goal = GoalToolPose.from_poses(
                goal_poses,
                ordered_tool_frames=self._mpc.tool_frames,
                num_goalset=1,
            )

            # run_ik=True on first goal so MPC gets a valid seed trajectory;
            # subsequent goals use run_ik=False for lower latency.
            run_ik = not goal_active
            try:
                self._mpc.update_goal_tool_poses(goal, run_ik=run_ik)
            except Exception as e:
                self.get_logger().error(
                    f"update_goal_tool_poses raised: {e}", throttle_duration_sec=2.0
                )
                self._publish_status("failed")
                return

            with self._lock:
                self._goal_active = True
            goal_active = True

            self.get_logger().info(
                f"MPC goal updated (run_ik={run_ik})  |  target: {self._mpc.tool_frames[0]}"
            )

        if not goal_active:
            # No goal yet — stay idle; do not call MPC (it has no goal IK seed).
            return

        # ── Optimise action sequence (warm-started) ───────────────────────────
        try:
            result = self._mpc.optimize_action_sequence(current_state)
        except Exception as e:
            self.get_logger().error(
                f"optimize_action_sequence raised: {e}", throttle_duration_sec=2.0
            )
            self._publish_status("failed")
            return

        # ── Validate result ───────────────────────────────────────────────────
        if (
            result is None
            or result.action_sequence is None
            or result.action_sequence.position.shape[1] == 0
        ):
            self.get_logger().warn(
                "MPC returned empty action sequence.", throttle_duration_sec=1.0
            )
            return

        # ── Extract LAST action in the optimised horizon (same as reactive_control.py)
        #    shape: (1, T, DOF) → take [:, -1, :] → (1, DOF) → squeeze → (DOF,)
        seq      = result.action_sequence
        next_pos = seq.position[:, -1, :].squeeze(0)    # (DOF,)
        next_vel = seq.velocity[:, -1, :].squeeze(0)    # (DOF,)
        next_acc = seq.acceleration[:, -1, :].squeeze(0) # (DOF,)

        # ── Publish single-point trajectory command ───────────────────────────
        traj_msg = _build_single_point_trajectory(
            next_pos, next_vel, next_acc,
            self._optimization_dt,
            self._joint_names,
            self.get_clock(),
        )
        self._traj_pub.publish(traj_msg)

        # ── Feed action back as current_state for next tick (warm-start) ──────
        #    Write back as plain Python lists so no CUDA tensor crosses the
        #    lock boundary.  _js_callback will overwrite this with real
        #    feedback whenever a joint_states message arrives.
        with self._lock:
            self._pending_js = (
                next_pos.cpu().tolist(),
                next_vel.cpu().tolist(),
            )

        # ── Status / convergence reporting ────────────────────────────────────
        pos_err = result.position_error
        if pos_err is not None:
            err_val = pos_err.item()
            if err_val < POSITION_ERROR_CONVERGED:
                self._publish_status("converged")
            else:
                self._publish_status("active")

            self.get_logger().debug(
                f"MPC tick  pos_err={err_val:.4f} m",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = CuRoboMPCNode()
    # 4 threads: timer thread + JS callback + goal callback + spare
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down cuRobo MPC node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

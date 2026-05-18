#!/usr/bin/env python3
"""
cuRobo GPU-accelerated Motion Planning Node for mvp_arm
========================================================
Subscribes : /curobo/goal_pose  (geometry_msgs/PoseStamped)
             /joint_states       (sensor_msgs/JointState)
Publishes  : /mvp_arm_controller/joint_trajectory (trajectory_msgs/JointTrajectory)
             /curobo/status      (std_msgs/String)  -- "success" | "failed" | "planning"

Run from conda env:
    conda activate curobo
    python curobo_ros2_node.py

Send a goal:
    ros2 topic pub --once /curobo/goal_pose geometry_msgs/msg/PoseStamped "{
      header: {frame_id: 'base_link'},
      pose: {
        position: {x: 0.0324, y: -0.2098, z: 0.8858},
        orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}
      }
    }"
"""

import sys
import os

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
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import JointState as RosJointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from builtin_interfaces.msg import Duration

# cuRobo ── newer high-level API (same as your motion_planning.py)
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState as CuJointState, Pose


# ─────────────────────────────────────────────────────────────────────────────
# Constants  ── edit these to match your setup
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_CFG       = "mvp_arm.yml"
SCENE_CFG       = "collision_test.yml"   # has the table; make sure z is -0.15
TRAJ_TOPIC      = "/mvp_arm_controller/joint_trajectory"
JS_TOPIC        = "/isaac_joint_states"
GOAL_TOPIC      = "/curobo/goal_pose"
STATUS_TOPIC    = "/curobo/status"

# Joint names in the order mvp_arm.yml defines them
# (must match what Isaac Sim publishes on /joint_states)
MVP_JOINT_NAMES = [
    "RSP_joint",
    "RSY_joint",
    "RSR_joint",
    "REP_joint",
    "RWR_joint",
    "RWP_joint",
    "RWY_joint",
]

# Planning quality knobs
WARMUP_ITERATIONS   = 10     # more = faster first real plan
MAX_ATTEMPTS        = 5      # retry seeds if first attempt fails
INTERPOLATION_DT    = 0.02   # seconds between waypoints (50 Hz) → smooth motion
# ─────────────────────────────────────────────────────────────────────────────


def _build_trajectory_msg(
    interpolated: CuJointState,
    dt: float,
    joint_names: list,
    clock,
) -> JointTrajectory:
    """Convert a cuRobo interpolated JointState into a ROS2 JointTrajectory."""
    msg = JointTrajectory()
    msg.header.stamp = clock.now().to_msg()
    msg.joint_names = list(joint_names)

    pos = interpolated.position
    vel = interpolated.velocity
    acc = interpolated.acceleration

    # Squeeze batch dim: (1, T, DOF) → (T, DOF)
    if pos.dim() == 3:
        pos = pos.squeeze(0)
    if vel is not None and vel.dim() == 3:
        vel = vel.squeeze(0)
    if acc is not None and acc.dim() == 3:
        acc = acc.squeeze(0)

    # After the squeeze block, add this debug line temporarily:
    print(f"pos shape after squeeze: {pos.shape}")

    # And change the numpy conversion to force 2D:
    pos_np = pos.cpu().numpy().reshape(-1, len(joint_names)).astype(np.float64)
    vel_np = vel.cpu().numpy().reshape(-1, len(joint_names)).astype(np.float64) if vel is not None else None
    acc_np = acc.cpu().numpy().reshape(-1, len(joint_names)).astype(np.float64) if acc is not None else None
    n_pts  = pos_np.shape[0]

    for i in range(n_pts):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in pos_np[i]]  # explicit Python float

        if vel_np is not None:
            pt.velocities = [float(v) for v in vel_np[i]]
        if acc_np is not None:
            pt.accelerations = [float(v) for v in acc_np[i]]

        t_sec = i * dt
        pt.time_from_start = Duration(
            sec=int(t_sec),
            nanosec=int(round((t_sec % 1.0) * 1e9)),
        )
        msg.points.append(pt)

    return msg


# ─────────────────────────────────────────────────────────────────────────────

class CuRoboNode(Node):
    def __init__(self):
        super().__init__("curobo_motion_planner")

        # ── cuRobo init ───────────────────────────────────────────────────────
        self.get_logger().info("Initialising cuRobo motion planner…")

        cfg = MotionPlannerCfg.create(
            robot=ROBOT_CFG,
            scene_model=SCENE_CFG,
        )
        self.planner = MotionPlanner(cfg)

        self.get_logger().info(
            f"Warming up ({WARMUP_ITERATIONS} iterations, CUDA graphs enabled)…"
        )
        self.planner.warmup(
            enable_graph=True,
            num_warmup_iterations=WARMUP_ITERATIONS,
        )

        self.joint_names = self.planner.joint_names
        self.interp_dt   = self.planner.trajopt_solver.config.interpolation_dt
        self.get_logger().info(
            f"cuRobo ready | joints: {list(self.joint_names)} | dt: {self.interp_dt:.3f}s"
        )

        # ── internal state ────────────────────────────────────────────────────
        self._lock        = threading.Lock()
        self._current_js: Optional[CuJointState] = None   # updated every /joint_states msg
        self._is_planning = False

        # Start state falls back to robot default until first /joint_states arrives
        self._default_js = CuJointState.from_position(
            self.planner.default_joint_state.position.unsqueeze(0),
            joint_names=list(self.joint_names),
        )

        # ── ROS2 pub/sub ──────────────────────────────────────────────────────
        cb = ReentrantCallbackGroup()

        self._js_sub = self.create_subscription(
            RosJointState,
            JS_TOPIC,
            self._js_callback,
            10,
            callback_group=cb,
        )
        self._goal_sub = self.create_subscription(
            PoseStamped,
            GOAL_TOPIC,
            self._goal_callback,
            10,
            callback_group=cb,
        )
        self._traj_pub   = self.create_publisher(JointTrajectory, TRAJ_TOPIC, 10)
        self._status_pub = self.create_publisher(String, STATUS_TOPIC, 10)

        self.get_logger().info(
            f"\n"
            f"  Listening for goals on : {GOAL_TOPIC}\n"
            f"  Joint states from      : {JS_TOPIC}\n"
            f"  Publishing traj to     : {TRAJ_TOPIC}\n"
            f"  Status on              : {STATUS_TOPIC}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _js_callback(self, msg: RosJointState):
        """Keep latest joint state from Isaac Sim, reordered to cuRobo's joint order."""
        try:
            positions = []
            for name in self.joint_names:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])

            with self._lock:
                self._current_js = CuJointState.from_position(
                    torch.tensor([positions], device="cuda", dtype=torch.float32),
                    joint_names=list(self.joint_names),
                )
        except ValueError as e:
            self.get_logger().warn(
                f"Joint name mismatch: {e}", throttle_duration_sec=5.0
            )

    def _goal_callback(self, msg: PoseStamped):
        """Receive a PoseStamped goal and trigger planning in a background thread."""
        if self._is_planning:
            self.get_logger().warn("Already planning — goal ignored.")
            return

        p = msg.pose.position
        q = msg.pose.orientation   # geometry_msgs: x y z w

        self.get_logger().info(
            f"Goal received  pos=[{p.x:.4f}, {p.y:.4f}, {p.z:.4f}]  "
            f"quat(wxyz)=[{q.w:.3f}, {q.x:.3f}, {q.y:.3f}, {q.z:.3f}]"
        )

        # Kick off planning in a separate thread so the callback returns fast
        threading.Thread(
            target=self._plan_and_publish,
            args=(p.x, p.y, p.z, q.w, q.x, q.y, q.z),
            daemon=True,
        ).start()

    # ─────────────────────────────────────────────────────────────────────────
    # Planning
    # ─────────────────────────────────────────────────────────────────────────

    def _plan_and_publish(self, px, py, pz, qw, qx, qy, qz):
        """GPU motion planning + trajectory publish (runs in background thread)."""
        self._is_planning = True
        self._publish_status("planning")

        try:
            # ── 1. Grab current joint state (or use default) ─────────────────
            with self._lock:
                start_js = (
                    self._current_js.clone()
                    if self._current_js is not None
                    else self._default_js.clone()
                )
            if self._current_js is None:
                self.get_logger().warn(
                    "No /joint_states received yet — using robot default config."
                )

            # ── 2. Build goal pose ────────────────────────────────────────────
            # GoalToolPose shape: (1, 1, 1, 1, 3) for position, (1,1,1,1,4) for quat
            goal = GoalToolPose(
                tool_frames=self.planner.tool_frames,
                position=torch.tensor(
                    [[[[[px, py, pz]]]]],
                    device="cuda", dtype=torch.float32,
                ),
                quaternion=torch.tensor(
                    [[[[[qw, qx, qy, qz]]]]],   # cuRobo expects wxyz
                    device="cuda", dtype=torch.float32,
                ),
            )

            # ── 3. Plan on GPU ────────────────────────────────────────────────
            t0 = time.perf_counter()
            result = self.planner.plan_pose(
                goal,
                start_js,
                max_attempts=MAX_ATTEMPTS,
            )
            elapsed = time.perf_counter() - t0

            # ── 4. Handle result ──────────────────────────────────────────────
            if result is None or not result.success.any():
                self.get_logger().error(
                    f"Motion planning FAILED after {elapsed:.2f}s"
                )
                self._publish_status("failed")
                return

            interpolated = result.get_interpolated_plan()
            n_wp = interpolated.position.shape[-2]
            duration = n_wp * self.interp_dt

            self.get_logger().info(
                f"Planning SUCCEEDED  |  {elapsed:.2f}s planning  |  "
                f"{n_wp} waypoints  |  {duration:.2f}s trajectory"
            )

            # ── 5. Build and publish JointTrajectory ──────────────────────────
            traj_msg = _build_trajectory_msg(
                interpolated,
                self.interp_dt,
                self.joint_names,
                self.get_clock(),
            )
            self._traj_pub.publish(traj_msg)
            self._publish_status("success")
            self.get_logger().info("Trajectory published.")

        except Exception as e:
            self.get_logger().error(f"Unexpected error during planning: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()

        finally:
            self._is_planning = False

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
    node = CuRoboNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down cuRobo node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
cuRobo GPU-accelerated Motion Planning Node for minibot dual arm
================================================================
Subscribes : /curobo/arm_command  (minibot_msgs/ArmCommand)
             /joint_states         (sensor_msgs/JointState)

Publishes  : /dual_arm_controller/joint_trajectory  (trajectory_msgs/JointTrajectory)
             /curobo/status                          (std_msgs/String)

ArmCommand.mode:
    LEFT  (0) → plan left arm to left_pose;   right arm holds current position
    RIGHT (1) → plan right arm to right_pose;  left arm holds current position
    DUAL  (2) → plan both arms simultaneously

All three modes publish to /dual_arm_controller (14 DOF) so no controller
switching is ever needed. The idle arm in single-arm modes receives a
trajectory where every waypoint repeats its current joint angles at zero
velocity/acceleration — the JointTrajectoryController holds it still.

Run:
    source /opt/ros/humble/setup.bash
    cd ~/curobo && source .venv/bin/activate
    python curobo_minibot_node.py
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
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState as CuJointState

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
MAX_ATTEMPTS      = 5

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

        cfg_kwargs = dict(robot=ROBOT_CFG)
        if SCENE_CFG:
            cfg_kwargs["scene_model"] = SCENE_CFG

        cfg = MotionPlannerCfg.create(**cfg_kwargs)
        self.planner = MotionPlanner(cfg)

        self.get_logger().info(f"Warming up ({WARMUP_ITERATIONS} iters)…")
        self.planner.warmup(enable_graph=True, num_warmup_iterations=WARMUP_ITERATIONS)

        self.joint_names = list(self.planner.joint_names)   # 14 DOF
        self.interp_dt   = self.planner.trajopt_solver.config.interpolation_dt

        # Tool frames: expect [LWY_link, RWY_link] from minibot_dual_arm.yml
        self.left_tool  = [f for f in self.planner.tool_frames if f.startswith("L")]
        self.right_tool = [f for f in self.planner.tool_frames if f.startswith("R")]

        self.get_logger().info(
            f"Ready | left tool: {self.left_tool} | "
            f"right tool: {self.right_tool} | interp_dt: {self.interp_dt:.3f}s"
        )

        # ── state ─────────────────────────────────────────────────────────────
        self._lock            = threading.Lock()
        self._current_js:     Optional[CuJointState] = None
        self._current_pos_np: Optional[np.ndarray]   = None   # (14,) raw angles
        self._is_planning     = False

        self._default_js     = CuJointState.from_position(
            self.planner.default_joint_state.position.unsqueeze(0),
            joint_names=self.joint_names,
        )
        self._default_pos_np = self.planner.default_joint_state.position.cpu().numpy()

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
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _js_cb(self, msg: RosJointState):
        """Cache latest joint state reordered to cuRobo's 14-DOF order."""
        try:
            positions = []
            for name in self.joint_names:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])
            pos_np = np.array(positions, dtype=np.float32)
            with self._lock:
                self._current_js = CuJointState.from_position(
                    torch.tensor([positions], device="cuda", dtype=torch.float32),
                    joint_names=self.joint_names,
                )
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

        mode = msg.mode

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

    # ──────────────────────────────────────────────────────────────────────────
    # Planning
    # ──────────────────────────────────────────────────────────────────────────

    def _get_start(self):
        """Return (CuJointState, np.ndarray[14]) for the current robot state."""
        with self._lock:
            js     = self._current_js.clone()    if self._current_js     is not None else None
            pos_np = self._current_pos_np.copy() if self._current_pos_np is not None else None
        if js is None:
            self.get_logger().warn("No /joint_states yet — using robot default.")
            js     = self._default_js.clone()
            pos_np = self._default_pos_np.copy()
        return js, pos_np

    def _plan_single(self, arm: str, px, py, pz, qw, qx, qy, qz):
        """
        Plan one arm. The other arm's joints are frozen at their current angles.
        Publishes a 14-DOF trajectory to dual_arm_controller.
        """
        self._is_planning = True
        self._publish_status("planning")
        try:
            start_js, pos_np = self._get_start()

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

            t0 = time.perf_counter()
            result = self.planner.plan_pose(goal, start_js, max_attempts=MAX_ATTEMPTS)
            elapsed = time.perf_counter() - t0

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
            self._publish_status("success")
            self.get_logger().info("Trajectory published to dual_arm_controller.")

        except Exception as e:
            self.get_logger().error(f"Error planning {arm} arm: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self._is_planning = False

    def _plan_dual(self,
                   lpx, lpy, lpz, lqw, lqx, lqy, lqz,
                   rpx, rpy, rpz, rqw, rqx, rqy, rqz):
        """Plan both arms simultaneously — full 14-DOF cuRobo plan."""
        self._is_planning = True
        self._publish_status("planning")
        try:
            start_js, _ = self._get_start()

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
            self._publish_status("success")
            self.get_logger().info("Dual arm trajectory published.")

        except Exception as e:
            self.get_logger().error(f"Error planning dual arm: {e}")
            self._publish_status("failed")
            import traceback; traceback.print_exc()
        finally:
            self._is_planning = False

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

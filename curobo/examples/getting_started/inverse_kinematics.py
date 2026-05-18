# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Dual-arm inverse kinematics for minibot_dual_arm using cuRobo.

Solves IK simultaneously for both left and right arms at specified
reachable poses. Both arms share the same orientation: RPY(0, -1.57, 0).

Left arm target:
    position : (0.254,  0.149, 0.464)
Right arm target:
    position : (0.254, -0.149, 0.464)

Run headless (single + batched + collision-free IK):

    python inverse_kinematics_dual_arm.py

Interactive Viser viewer (drag gizmos per arm):

    python inverse_kinematics_dual_arm.py --visualize

Reachability map:

    python inverse_kinematics_dual_arm.py --reachability
"""

import argparse
import copy
import sys
import time

import numpy as np
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.scene import Cuboid, Scene
from curobo.types import ContentPath, GoalToolPose, Pose
from curobo.viewer import ViserVisualizer

# ---------------------------------------------------------------------------
# Robot config
# ---------------------------------------------------------------------------
ROBOT_FILE = "minibot_dual_arm_new.yml"

# ---------------------------------------------------------------------------
# Target poses  (RPY 0, -1.57, 0  →  wxyz quaternion)
# ---------------------------------------------------------------------------
# w=0.7074, x=0.0, y=-0.7068, z=0.0
_QUAT_WXYZ = [0.7073882691671998, 0.0, -0.706825181105366, 0.0]

LEFT_POSITION  = [0.254,  0.149, 0.464]
RIGHT_POSITION = [0.254, -0.149, 0.464]


def _make_pose(position: list) -> Pose:
    """Return a single-batch Pose on CUDA."""
    return Pose(
        position=torch.tensor([position], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([_QUAT_WXYZ], device="cuda", dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# 1. Single IK
# ---------------------------------------------------------------------------
def single_ik_example():
    """Solve IK for both arms simultaneously at the fixed reachable poses."""
    config = InverseKinematicsCfg.create(
        robot=ROBOT_FILE,
        num_seeds=32,
    )
    ik = InverseKinematics(config)

    # Identify left/right tool frames from the robot config.
    # Assumes tool_frames are ordered [left, right] in the YAML.
    tool_frames = ik.tool_frames
    assert len(tool_frames) >= 2, (
        f"Expected at least 2 tool frames, got: {tool_frames}"
    )
    left_link, right_link = tool_frames[0], tool_frames[1]
    print(f"Tool frames  ->  left: '{left_link}'  right: '{right_link}'")

    goal_poses = {
        left_link:  _make_pose(LEFT_POSITION),
        right_link: _make_pose(RIGHT_POSITION),
    }

    result = ik.solve_pose(
        GoalToolPose.from_poses(
            goal_poses,
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        )
    )

    print("\n=== Single Dual-Arm IK ===")
    if result.success.item():
        print("IK solved for both arms!")
        print(f"  Joint angles : {result.js_solution.position}")
        print(f"  Position error: {result.position_error.item() * 1000:.3f} mm")
        return True
    else:
        print("IK failed — check that the poses are reachable and the robot YAML is correct.")
        return False


# ---------------------------------------------------------------------------
# 2. Batched IK
# ---------------------------------------------------------------------------
def batched_ik_example():
    """Solve IK for a batch of pose pairs (left arm swept, right arm fixed)."""
    n_poses = 50

    config = InverseKinematicsCfg.create(
        robot=ROBOT_FILE,
        num_seeds=32,
        max_batch_size=n_poses,
    )
    ik = InverseKinematics(config)
    tool_frames = ik.tool_frames
    left_link, right_link = tool_frames[0], tool_frames[1]

    # Sweep left arm X position; keep right arm fixed
    left_positions = torch.zeros(n_poses, 3, device="cuda", dtype=torch.float32)
    left_positions[:, 0] = torch.linspace(0.20, 0.30, n_poses)   # X sweep
    left_positions[:, 1] =  0.149
    left_positions[:, 2] =  0.464

    right_positions = torch.zeros(n_poses, 3, device="cuda", dtype=torch.float32)
    right_positions[:, 0] = 0.254
    right_positions[:, 1] = -0.149
    right_positions[:, 2] =  0.464

    quats = torch.tensor(
        [_QUAT_WXYZ], device="cuda", dtype=torch.float32
    ).expand(n_poses, -1).contiguous()

    goal_poses = {
        left_link:  Pose(position=left_positions,  quaternion=quats),
        right_link: Pose(position=right_positions, quaternion=quats),
    }

    result = ik.solve_pose(
        GoalToolPose.from_poses(
            goal_poses,
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        )
    )

    n_success = result.success.sum().item()
    print(f"\n=== Batched Dual-Arm IK ({n_poses} poses) ===")
    print(f"Solved {n_success}/{n_poses} ({100 * n_success / n_poses:.0f}% success)")
    if n_success > 0:
        successful = result.success.squeeze()
        pos_errors = result.position_error[successful]
        print(f"Mean position error: {pos_errors.mean().item() * 1000:.3f} mm")
        print(f"Max  position error: {pos_errors.max().item()  * 1000:.3f} mm")
    return n_success > 0


# ---------------------------------------------------------------------------
# 3. Collision-free IK
# ---------------------------------------------------------------------------
def collision_free_ik_example():
    """Solve collision-free dual-arm IK with a table scene."""
    config = InverseKinematicsCfg.create(
        robot=ROBOT_FILE,
        scene_model="collision_table.yml",
        num_seeds=32,
        self_collision_check=True,
    )
    ik = InverseKinematics(config)
    tool_frames = ik.tool_frames
    left_link, right_link = tool_frames[0], tool_frames[1]

    goal_poses = {
        left_link:  _make_pose(LEFT_POSITION),
        right_link: _make_pose(RIGHT_POSITION),
    }

    result = ik.solve_pose(
        GoalToolPose.from_poses(
            goal_poses,
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        )
    )

    print("\n=== Collision-Free Dual-Arm IK ===")
    if result.success.item():
        print("Collision-free IK solved!")
        print(f"  Position error: {result.position_error.item() * 1000:.3f} mm")
    else:
        print("Collision-free IK failed.")

    # Add an obstacle and retry
    obstacle = Cuboid(
        name="box_1",
        pose=[0.254, 0.0, 0.6, 1, 0, 0, 0],
        dims=[0.05, 0.05, 0.1],
    )
    config2 = InverseKinematicsCfg.create(
        robot=ROBOT_FILE,
        scene_model="collision_table.yml",
        num_seeds=64,
        self_collision_check=True,
        collision_cache={"cuboid": 10},
    )
    ik2 = InverseKinematics(config2)
    ik2.update_world(Scene(cuboid=[obstacle]))

    result2 = ik2.solve_pose(
        GoalToolPose.from_poses(
            goal_poses,
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        )
    )
    if result2.success.item():
        print("After adding obstacle — still solved!")
        print(f"  Position error: {result2.position_error.item() * 1000:.3f} mm")
    else:
        print("After adding obstacle — IK failed (obstacle may block the path).")
    return True


# ---------------------------------------------------------------------------
# 4. Interactive Viser IK
# ---------------------------------------------------------------------------
def interactive_ik_example(robot_file=ROBOT_FILE, port=8080):
    """Real-time dual-arm IK with per-arm gizmos in Viser."""

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )

    config = InverseKinematicsCfg.create(
        robot=robot_file,
        optimizer_configs=["ik/lbfgs_ik.yml"],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model="collision_test.yml",
        use_cuda_graph=True,
        num_seeds=1,
        seed_solver_num_seeds=1,
    )
    config.scene_collision_cfg.use_warp_collision = True
    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    ik_solver = InverseKinematics(config)
    ik_solver.config.use_lm_seed = False
    ik_solver.config.exit_early = False

    tool_frames = ik_solver.tool_frames
    left_link, right_link = tool_frames[0], tool_frames[1]
    print(f"\nTool frames  ->  left: '{left_link}'  right: '{right_link}'")

    goal_state = ik_solver.default_joint_state.clone()
    kin_state = ik_solver.compute_kinematics(goal_state).clone()
    goal_tool_poses = kin_state.tool_poses.to_dict()

    current_state = ik_solver.get_active_js(ik_solver.default_joint_state.clone())
    current_state = current_state.unsqueeze(0)

    # Warm-up solve
    ik_solver.solve_pose(
        goal_tool_poses=GoalToolPose.from_poses(
            goal_tool_poses,
            ordered_tool_frames=tool_frames,
            num_goalset=1,
        ),
        current_state=current_state.clone(),
        return_seeds=1,
    )

    print(f"Interactive dual-arm IK running at http://localhost:{port}")
    print(f"Drag the '{left_link}' and '{right_link}' gizmos to update goals.")
    print("Press Ctrl+C to exit.\n")

    previous_target_poses = None
    pose_changed = False

    while True:
        # Update obstacle poses
        obstacle_poses = {
            k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            for k in obstacle_frames.keys()
        }
        for k in obstacle_poses.keys():
            if obstacle_poses[k] != old_obstacle_poses[k]:
                ik_solver.scene_collision_checker.update_obstacle_pose(k, obstacle_poses[k])
                pose_changed = True
        old_obstacle_poses = {k: v.clone() for k, v in obstacle_poses.items()}

        # Check for gizmo movement
        target_poses = viser_viz.get_control_frame_pose()
        if previous_target_poses is None:
            previous_target_poses = copy.deepcopy(target_poses)
        else:
            for frame_name in target_poses.keys():
                if target_poses[frame_name] != previous_target_poses[frame_name]:
                    previous_target_poses = {k: v.clone() for k, v in target_poses.items()}
                    pose_changed = True
                    break

        if pose_changed:
            active_js = ik_solver.get_active_js(current_state)
            target_link_poses = {
                k.replace("target_", ""): v for k, v in target_poses.items()
            }
            result = ik_solver.solve_pose(
                goal_tool_poses=GoalToolPose.from_poses(
                    target_link_poses,
                    ordered_tool_frames=tool_frames,
                    num_goalset=1,
                ),
                current_state=active_js.squeeze(1).clone(),
                return_seeds=1,
                run_optimizer=True,
            )
            if result.success:
                pose_changed = False
                current_state = result.js_solution.clone()
                viser_viz.set_joint_state(result.js_solution.squeeze(0).squeeze(0))

        time.sleep(0.001)


# ---------------------------------------------------------------------------
# 5. Reachability map
# ---------------------------------------------------------------------------
def reachability_example(robot_file=ROBOT_FILE, port=8080):
    """Interactive reachability slice viewer for the dual-arm robot."""
    import viser.transforms as vtf

    BATCH_TARGET = 500
    n_per_axis = int(BATCH_TARGET ** 0.5)
    actual_batch = n_per_axis * n_per_axis

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=False,
        visualize_robot_spheres=False,
        add_robot_to_scene=True,
    )
    server = viser_viz._server

    config = InverseKinematicsCfg.create(
        robot=robot_file,
        self_collision_check=True,
        scene_model="collision_test.yml",
    )
    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)

    ik = InverseKinematics(config)
    ik.exit_early = False
    tool_frames = ik.tool_frames
    primary_link = tool_frames[0]    # left arm drives the reachability grid
    secondary_link = tool_frames[1]  # right arm stays fixed

    kin_state = ik.compute_kinematics(ik.default_joint_state)
    tool_pose = kin_state.tool_poses[primary_link]
    center = tool_pose.position.squeeze().cpu().numpy()

    slice_gizmo = server.scene.add_transform_controls(
        "/reachability_gizmo",
        scale=0.15,
        position=tuple(center.tolist()),
        wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    # Secondary (right arm) gizmo
    secondary_pose = kin_state.tool_poses[secondary_link]
    secondary_pos = secondary_pose.position.squeeze().cpu().numpy()
    secondary_quat = secondary_pose.quaternion.squeeze().cpu().numpy()
    secondary_gizmo = server.scene.add_transform_controls(
        f"/tool_frame_{secondary_link}",
        scale=0.10,
        position=tuple(secondary_pos.tolist()),
        wxyz=tuple(secondary_quat.tolist()),
    )

    with server.gui.add_folder("Reachability"):
        grid_extent_slider = server.gui.add_slider(
            "Grid Extent (m)", min=0.1, max=2.0, step=0.05, initial_value=0.5,
        )

    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }
    prev_gizmo_pos   = np.array(slice_gizmo.position, dtype=np.float32)
    prev_gizmo_wxyz  = np.array(slice_gizmo.wxyz,     dtype=np.float32)
    prev_sec_pos     = np.array(secondary_gizmo.position, dtype=np.float32)
    prev_sec_wxyz    = np.array(secondary_gizmo.wxyz,     dtype=np.float32)
    prev_extent      = grid_extent_slider.value

    print(f"\nReachability viewer running at http://localhost:{port}")
    print(f"Primary (left) arm:  '{primary_link}'")
    print(f"Secondary (right) arm: '{secondary_link}'")
    print(f"Slice: {n_per_axis}x{n_per_axis} = {actual_batch} IK queries per update")
    print("Drag the reachability gizmo to move/rotate the slice plane.")
    print("Drag the right-arm gizmo to change its fixed goal.")
    print("Press Ctrl+C to exit.\n")

    needs_update = True

    while True:
        obstacle_poses = {
            k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            for k in obstacle_frames.keys()
        }
        for k in obstacle_poses.keys():
            if obstacle_poses[k] != old_obstacle_poses[k]:
                ik.scene_collision_checker.update_obstacle_pose(k, obstacle_poses[k])
                needs_update = True
        old_obstacle_poses = {k: v.clone() for k, v in obstacle_poses.items()}

        cur_pos    = np.array(slice_gizmo.position, dtype=np.float32)
        cur_wxyz   = np.array(slice_gizmo.wxyz,     dtype=np.float32)
        cur_sec_pos  = np.array(secondary_gizmo.position, dtype=np.float32)
        cur_sec_wxyz = np.array(secondary_gizmo.wxyz,     dtype=np.float32)
        cur_extent = grid_extent_slider.value

        if (
            not np.allclose(cur_pos,   prev_gizmo_pos)
            or not np.allclose(cur_wxyz,  prev_gizmo_wxyz)
            or not np.allclose(cur_sec_pos,  prev_sec_pos)
            or not np.allclose(cur_sec_wxyz, prev_sec_wxyz)
            or cur_extent != prev_extent
        ):
            needs_update = True
            prev_gizmo_pos  = cur_pos
            prev_gizmo_wxyz = cur_wxyz
            prev_sec_pos    = cur_sec_pos
            prev_sec_wxyz   = cur_sec_wxyz
            prev_extent     = cur_extent

        if not needs_update:
            time.sleep(0.02)
            continue
        needs_update = False

        extent = cur_extent
        half   = extent / 2.0
        rot    = vtf.SO3(cur_wxyz).as_matrix().astype(np.float32)
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = rot
        pose_matrix[:3, 3]  = cur_pos
        pose_t = torch.tensor(pose_matrix, device="cuda", dtype=torch.float32)

        lin = torch.linspace(-half, half, n_per_axis, device="cuda", dtype=torch.float32)
        uu, vv = torch.meshgrid(lin, lin, indexing="xy")
        local_pts = torch.stack(
            [uu.reshape(-1), vv.reshape(-1),
             torch.zeros(actual_batch, device="cuda", dtype=torch.float32),
             torch.ones(actual_batch,  device="cuda", dtype=torch.float32)],
            dim=-1,
        )
        grid_world_pts = (pose_t @ local_pts.T).T[:, :3]

        total_batch = actual_batch + 1

        # Primary (left) arm: grid + the gizmo tip
        primary_pos = torch.tensor(cur_pos, device="cuda", dtype=torch.float32).unsqueeze(0)
        all_primary_positions  = torch.cat([grid_world_pts, primary_pos], dim=0)
        primary_quat = torch.tensor(_QUAT_WXYZ, device="cuda", dtype=torch.float32)
        all_primary_quats = primary_quat.unsqueeze(0).expand(total_batch, -1).contiguous()

        # Secondary (right) arm: fixed at gizmo position for all batch entries
        sec_pos  = torch.tensor(cur_sec_pos,  device="cuda", dtype=torch.float32)
        sec_quat = torch.tensor(cur_sec_wxyz, device="cuda", dtype=torch.float32)
        all_sec_positions = sec_pos.unsqueeze(0).expand(total_batch, -1).contiguous()
        all_sec_quats     = sec_quat.unsqueeze(0).expand(total_batch, -1).contiguous()

        goal_dict = {
            primary_link:   Pose(position=all_primary_positions, quaternion=all_primary_quats),
            secondary_link: Pose(position=all_sec_positions,     quaternion=all_sec_quats),
        }

        result = ik.solve_pose(
            GoalToolPose.from_poses(
                goal_dict,
                ordered_tool_frames=tool_frames,
                num_goalset=1,
            ),
        )

        all_success  = result.success.squeeze().cpu().numpy().astype(bool)
        grid_success = all_success[:actual_batch].reshape(n_per_axis, n_per_axis)
        gizmo_success = all_success[actual_batch]

        img = np.zeros((n_per_axis, n_per_axis, 3), dtype=np.uint8)
        img[grid_success]  = [0,   200, 0]
        img[~grid_success] = [200, 0,   0]

        if gizmo_success:
            viser_viz.set_joint_state(result.js_solution[actual_batch].squeeze(0))

        server.scene.add_image(
            name="/reachability_gizmo/slice_image",
            image=img,
            render_width=extent,
            render_height=extent,
        )

        corners_local = np.array(
            [[-half, -half, 0], [half, -half, 0],
             [half,  half,  0], [-half, half, 0]], dtype=np.float32,
        )
        corners_world = (rot @ corners_local.T).T + cur_pos
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        lines = np.array([[corners_world[i], corners_world[j]] for i, j in edges], dtype=np.float32)
        server.scene.add_line_segments(
            "/reachability_bounds", points=lines,
            colors=np.array([255, 255, 0], dtype=np.uint8), line_width=3.0,
        )

        n_success = int(grid_success.sum())
        print(
            f"Reachability (left arm): {n_success}/{actual_batch} "
            f"({100 * n_success / actual_batch:.0f}%) | "
            f"Grid: {n_per_axis}x{n_per_axis} | Extent: {extent:.2f} m"
        )


# ---------------------------------------------------------------------------
# Test / main
# ---------------------------------------------------------------------------
def test():
    assert single_ik_example(),        "Single dual-arm IK failed"
    assert batched_ik_example(),       "Batched dual-arm IK failed"
    assert collision_free_ik_example(), "Collision-free dual-arm IK failed"


def main():
    parser = argparse.ArgumentParser(description="Dual-Arm IK with cuRobo (minibot_dual_arm)")
    parser.add_argument("--test",         action="store_true")
    parser.add_argument("--visualize",    action="store_true", help="Interactive Viser IK viewer")
    parser.add_argument("--reachability", action="store_true", help="Interactive reachability map")
    parser.add_argument("--robot",  type=str, default=ROBOT_FILE)
    parser.add_argument("--port",   type=int, default=8080)
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "collision_free", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)

    if args.reachability:
        reachability_example(robot_file=args.robot, port=args.port)
        return

    if args.visualize:
        interactive_ik_example(robot_file=args.robot, port=args.port)
        return

    if args.mode in ("single", "all"):
        single_ik_example()

    if args.mode in ("batch", "all"):
        batched_ik_example()

    if args.mode in ("collision_free", "all"):
        collision_free_ik_example()


if __name__ == "__main__":
    main()

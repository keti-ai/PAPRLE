import pyroki as pk

import os, sys
from paprle.ik.base import BaseIKSolver
from yourdfpy.urdf import URDF

"""
Solves the basic IK problem.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import numpy as np
from pytransform3d.transformations import transform_from_pq


def solve_ik_with_multiple_targets(
    robot: pk.Robot,
    target_link_indices: Sequence[int],
    target_wxyzs: onp.ndarray,
    target_positions: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_names: Sequence[str]. List of link names to be controlled.
        target_wxyzs: onp.ndarray. Shape: (num_targets, 4). Target orientations.
        target_positions: onp.ndarray. Shape: (num_targets, 3). Target positions.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    num_targets = len(target_link_indices)
    assert target_positions.shape == (num_targets, 3)
    #assert target_wxyzs.shape == (num_targets, 4)
    #target_link_indices = [robot.links.names.index(name) for name in target_link_names]

    cfg = _solve_ik_jax(
        robot,
        target_wxyzs,
        jnp.array(target_positions),
        jnp.array(target_link_indices, dtype=jnp.int32),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_wxyz: jaxlie.SO3,
    target_position: jax.Array,
    target_joint_indices: jax.Array,
) -> jax.Array:
    JointVar = robot.joint_var_cls

    # Get the batch axes for the variable through the target pose.
    # Batch axes for the variables and cost terms (e.g., target pose) should be broadcastable!
    target_pose = jaxlie.SE3.from_rotation_and_translation(
       target_wxyz, target_position
    )
    batch_axes = target_pose.get_batch_axes()

    factors = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch_axes, 0, dtype=jnp.int32)),
            target_pose,
            target_joint_indices,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=JointVar.default_factory(),
            weight=1.0,
        ),
        pk.costs.limit_cost(
            robot,
            JointVar(0),
            jnp.array([100.0] * robot.joints.num_joints),
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [JointVar(0)])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[JointVar(0)]


class PyrokiIKSolver(BaseIKSolver):
    def __init__(self, robot):
        jax.config.update("jax_enable_x64", False)
        self.robot_name = robot.name
        self.config = robot.ik_config

        self.limb_names = robot.limb_names

        urdf_file, asset_dir = robot.urdf_file, robot.asset_dir
        actuated_joint_names, ee_links = [], []
        for limb_name in self.limb_names:
            joint_names = robot.robot_config.limb_joint_names
            actuated_joint_names.extend(joint_names[limb_name])
            ee_links.append(robot.eef_names[limb_name])

        self.out_joint_names = actuated_joint_names
        self.ctrl_joint_names = robot.joint_names
        self.out_idx_mapping = self.get_ik_idx_mappings()

        urdf = URDF.load(urdf_file)
        for joint_name, joint_info in urdf.joint_map.items():
            if joint_name not in self.out_joint_names:
                joint_info.type = 'fixed'
                if 'mimic' in joint_info.__dict__:
                    joint_info.mimic = None
        urdf._update_actuated_joints()
        self.robot = pk.Robot.from_urdf(urdf)
        self.target_link_names = ee_links
        self.target_link_ids = [self.robot.links.names.index(name) for name in self.target_link_names]
        self.urdf_actuated_names = self.robot.joints.actuated_names
        self.urdf_to_ctrl_idx_mapping = [self.urdf_actuated_names.index(name) for name in self.out_joint_names]
        self.qpos = np.zeros(len(self.out_joint_names))

    def solve(self, Rts):
        Rts = np.array(Rts)
        solution = solve_ik_with_multiple_targets(
            robot=self.robot,
            target_link_indices=self.target_link_ids,
            target_positions=Rts[:,:3,3],
            target_wxyzs=jaxlie.SO3.from_matrix(Rts[:,:3,:3]),
        )
        return solution[self.urdf_to_ctrl_idx_mapping]

    def reset(self):

        return

    def get_ik_idx_mappings(self):
        out_indices = []
        for name in self.out_joint_names:
            out_indices.append(self.ctrl_joint_names.index(name))
        self.out_idx_mapping = out_indices
        return out_indices

    def set_qpos(self, ctrl_qpos):
        self.qpos = ctrl_qpos[self.out_idx_mapping]
        return

    def compute_ee_poses(self, ctrl_qpos):
        ee_poses = {}
        transforms = self.robot.forward_kinematics(ctrl_qpos[self.out_idx_mapping])
        for limb_name, link_id in zip(self.limb_names, self.target_link_ids):
            qp = np.array(transforms[link_id])
            ee_poses[limb_name] = transform_from_pq(qp[[4,5,6,0,1,2,3]])  # wxyz to xyzw
        return ee_poses


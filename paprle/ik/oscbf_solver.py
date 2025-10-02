import os
from typing import List

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from paprle.ik.base import BaseIKSolver
from cbfpy import CBF
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from paprle.ik.oscbf_utils.controller import PoseTaskVelocityController
from paprle.ik.oscbf_utils.manipulator import CustomManipulator
from yourdfpy.urdf import URDF
from pytransform3d.rotations import matrix_from_quaternion
from pytransform3d.transformations import pq_from_transform
import copy
from typing import List, Optional, Dict

@jax.tree_util.register_static
class CombinedConfig(OSCBFVelocityConfig):

    def __init__(
        self,
        robot: CustomManipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-3
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        assert len(collision_positions) == len(collision_radii)
        self.num_collision_bodies = len(collision_positions)
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        objectives = []

        # Extract values
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # EE Set Containment
        h_ee_safe_set = jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        objectives = [h_ee_safe_set, h_joint_limits, h_singularity]

        # Collision Avoidance
        if len(self.robot.collision_radii) > 0:
            robot_collision_pos_rad = self.robot.link_collision_data(q)
            robot_collision_positions = robot_collision_pos_rad[:, :3]
            robot_collision_radii = robot_collision_pos_rad[:, 3, None]
            robot_num_pts = robot_collision_positions.shape[0]
            center_deltas = (
                robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
            ).reshape(-1, 3)
            radii_sums = (
                robot_collision_radii[:, None] + self.collision_radii[None, :]
            ).reshape(-1)
            h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

            # Whole-body Set Containment
            h_whole_body_upper = (
                jnp.tile(self.whole_body_pos_max, (robot_num_pts, 1))
                - robot_collision_positions
                - robot_collision_radii
            ).ravel()
            h_whole_body_lower = (
                robot_collision_positions
                - jnp.tile(self.whole_body_pos_min, (robot_num_pts, 1))
                - robot_collision_radii
            ).ravel()

            objectives.extend([h_collision, h_whole_body_upper, h_whole_body_lower])

        return jnp.concatenate(objectives)

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2

    def _P(self, z):
        q = z
        transforms = self.robot.joint_to_world_transforms(q)
        J = self.robot._ee_jacobian(transforms)
        M = self.robot._mass_matrix(transforms)
        M_inv = jnp.linalg.inv(M)
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv + 1e-6 * jnp.eye(task_inertia_inv.shape[0]))
        J_bar = M_inv @ J.T @ task_inertia
        J_hash = J_bar
        N = jnp.eye(self.num_joints) - J_hash @ J
        W_T_W_joint = jnp.diag(jnp.asarray(self.W_T_W_joint_diag))
        W_T_W_task = jnp.diag(jnp.asarray(self.W_T_W_task_diag))
        return N.T @ W_T_W_joint @ N + J.T @ W_T_W_task @ J


class OSCBFIKSolverMultiWrapper(BaseIKSolver):
    def __init__(self, robot):
        self.robot_name = robot.name
        self.config = robot.ik_config
        self.solvers = {}
        self.limb_names = robot.limb_names
        self.base2world = {}
        for limb_name, cfg in self.config.items():
            self.solvers[limb_name] = OSCBFIKSolver(self.robot_name, cfg)
            self.base2world[limb_name] = robot.urdf.get_transform(robot.urdf.base_link,cfg.attached_to)
        self.limb_names = list(self.solvers.keys())
        self.total_dof = sum([self.solvers[limb_name].get_dof() for limb_name in self.limb_names])
        self.joint_names, self.out_joint_names = [], []
        for limb_name in self.limb_names:
            self.joint_names += self.solvers[limb_name].config.joint_names
            self.out_joint_names += np.array(robot.joint_names)[robot.ctrl_arm_joint_idx_mapping[limb_name]].tolist()
        self.ctrl_joint_names = robot.joint_names
        self.out_idx_mapping = self.get_ik_idx_mappings()
        self.init_qpos_set = False
        self.init_qpos = robot.init_qpos
        self.set_qpos(self.init_qpos)

    def solve(self, Rts):
        total_qpos = []
        for id, limb_name in enumerate(self.limb_names):
            Rt = self.base2world[limb_name] @ Rts[id]
            out_q = self.solvers[limb_name].solve(Rt)
            total_qpos.append(out_q)
        total_qpos = np.concatenate(total_qpos)
        return total_qpos

    def reset(self):
        for limb_name in self.limb_names:
            self.solvers[limb_name].reset()
        self.init_qpos_set = False
        self.set_qpos(self.init_qpos)
        return

    def get_ik_idx_mappings(self):
        out_indices = []
        for joint_name in self.out_joint_names:
            out_indices.append(self.ctrl_joint_names.index(joint_name))
        self.out_idx_mapping = out_indices
        return out_indices

    def set_qpos(self, ctrl_qpos):
        qpos = ctrl_qpos[self.out_idx_mapping]
        qpos_idx = 0
        # [TODO] Better solution?
        for limb_name in self.limb_names:
            dof = len(self.solvers[limb_name].config.joint_names)
            self.solvers[limb_name].set_current_qpos(qpos[qpos_idx:qpos_idx+dof])
            if not self.init_qpos_set:
                self.solvers[limb_name].des_q = qpos[qpos_idx:qpos_idx + dof]
            qpos_idx += dof
        if not self.init_qpos_set:
            self.init_qpos_set = True

    def compute_ee_poses(self, ctrl_qpos: np.ndarray) -> Dict:
        qpos = ctrl_qpos[self.out_idx_mapping]
        qpos_idx = 0
        ee_poses = {}
        # [TODO] Better solution?
        for limb_name in self.limb_names:
            dof = len(self.solvers[limb_name].config.joint_names)
            self.solvers[limb_name].set_current_qpos(qpos[qpos_idx:qpos_idx+dof])
            qpos_idx += dof
            ee_pose = self.solvers[limb_name].compute_ee_pose(self.solvers[limb_name].qpos)
            ee_pose = np.linalg.inv(self.base2world[limb_name]) @ ee_pose
            ee_poses[limb_name] = ee_pose
        return ee_poses


class OSCBFIKSolver(BaseIKSolver):
    def __init__(self, robot_name, ik_config):


        self.config = ik_config
        urdf_path = os.path.abspath(ik_config.urdf_path)
        self.urdf = URDF.load(urdf_path)

        offset = self.urdf.get_transform(ik_config.ee_link,self.urdf.joint_map[ik_config.joint_names[-1]].child)
        self.robot_collision_data = self.load_collision_data(ik_config)
        self.robot = CustomManipulator.from_urdf(str(urdf_path),
                                                 asset_path=ik_config.asset_dir,
                                                 ee_offset=offset,
                                                 joint_names=ik_config.joint_names,
                                                 collision_data=self.robot_collision_data)

        if 'oscbf' in ik_config:
            self.ee_pos_min = ik_config.oscbf.get('ee_pos_min', [-5.65, -5.25, -5.65])
            self.ee_pos_max = ik_config.oscbf.get('ee_pos_max', [5.65, 5.25, 5.65])
            self.wb_pos_min = ik_config.oscbf.get('wb_pos_min', [-5.25, -5.25, -5.25])
            self.wb_pos_max = ik_config.oscbf.get('wb_pos_max', [5.25, 5.25, 5.25])
            self.collision_pos = ik_config.oscbf.get('collision_pos', [[-5.65, -5.25, -5.65]])
            self.collision_radii = ik_config.oscbf.get('collision_radii', [0.1])
        else:
            self.ee_pos_min = [-5.65, -5.25, -5.65]
            self.ee_pos_max = [5.65, 5.25, 5.65]
            self.wb_pos_min = [-5.25, -5.25, -5.25]
            self.wb_pos_max = [5.25, 5.25, 5.25]
            self.collision_pos = [[-5.65, -5.25, -5.65]]
            self.collision_radii = [0.1]

        self.collision_data = {"positions": self.collision_pos, "radii": self.collision_radii}
        self.oscbf_config = CombinedConfig(
                    self.robot,
                    self.ee_pos_min,
                    self.ee_pos_max,
                    self.collision_pos,
                    self.collision_radii,
                    self.wb_pos_min,
                    self.wb_pos_max,
                )

        self.cbf = CBF.from_config(self.oscbf_config)

        kp_pos = 10.0
        kp_rot = 5.0
        kp_joint = 5.0
        self.osc_controller = PoseTaskVelocityController(
            n_joints=self.robot.num_joints,
            kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
            kp_joint=kp_joint,
            # Note: velocity limits will be enforced via the QP
            qdot_min=None,
            qdot_max=None,
        )

        self.qpos = np.zeros(self.robot.num_joints)
        joint_names = self.get_joint_names()
        self.idx_mapping = [joint_names.index(name) for name in ik_config.joint_names]
        self.des_q = None

        # @partial(jax.jit, static_argnums=(0, 1, 2))
        def compute_control(
                robot: CustomManipulator,
                osc_controller: PoseTaskVelocityController,
                cbf: CBF,
                z: ArrayLike,
                z_ee_des: ArrayLike,
                des_q: Optional[ArrayLike] = None,
        ):
            q = z[: robot.num_joints]
            M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
            pos = ee_tmat[:3, 3]
            rot = ee_tmat[:3, :3]
            des_pos = z_ee_des[:3]
            des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
            des_vel = z_ee_des[12:15]
            des_omega = z_ee_des[15:18]
            # Set nullspace desired joint position
            u_nom = osc_controller(
                q, pos, rot, des_pos, des_rot, des_vel, des_omega, des_q, J, M_inv
            )
            return cbf.safety_filter(q, u_nom)

        @jax.jit
        def compute_control_jit(z, z_des, des_q=None):
            return compute_control(self.robot, self.osc_controller, self.cbf, z, z_des, des_q)

        self.compute_control = compute_control_jit


    def solve(self, Rt):
        pos = Rt[:3,3]
        rot = Rt[:3,:3].ravel()
        vel = np.zeros(3)
        omega = np.zeros(3)
        ee_state = np.array([*pos, *rot, *vel, *omega])

        qpos = self.qpos.copy()
        err = 0.0
        for k in range(10):
            curr_qpos = np.concatenate([qpos, np.zeros_like(qpos)])
            tau = self.compute_control(curr_qpos, ee_state, des_q=self.des_q)
            if jnp.any(jnp.isnan(tau)): break
            qpos += tau * 0.01
            curr_ee_state = jnp.array(self.robot.ee_transform(qpos))
            err = np.linalg.norm(curr_ee_state[:3, 3] - pos)
            if np.linalg.norm(err) < 1e-3:
                break
        self.qpos = qpos
        return np.array(qpos)[self.idx_mapping]

    def reset(self):
        return

    def set_current_qpos(self, qpos: np.ndarray):
        new_qpos = np.zeros(self.robot.num_joints)
        new_qpos[self.idx_mapping] = qpos
        self.qpos = new_qpos

    def get_joint_names(self) -> List[str]:
        return self.robot.joint_names

    def load_collision_data(self, ik_config):
        urdf_name = os.path.basename(ik_config.urdf_path)
        if 'papras_7dof.urdf' in urdf_name:
            from paprle.ik.collision_models.papras_7dof_collision_model import papras_collision_data as collision_data
        else:
            collision_data = None
        return collision_data

    def get_dof(self): return self.robot.num_joints

    def compute_ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        curr_ee_state = self.robot.ee_transform(qpos)
        return np.array(curr_ee_state)


import os
from typing import List, Dict

import mujoco
import numpy as np
from paprle.ik.base import BaseIKSolver
from paprle.envs.mujoco_env_utils.mujoco_parser import MuJoCoParserClass
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr


class MujocoIKSolverMultiWrapper(BaseIKSolver):
    def __init__(self, robot):
        self.robot_name = robot.name
        self.config = robot.ik_config
        self.solvers = {}
        self.limb_names = robot.limb_names
        self.base2world = {}
        for limb_name, cfg in self.config.items():
            self.solvers[limb_name] = MujocoIKSolver(self.robot_name, cfg)
            self.base2world[limb_name] = robot.urdf.get_transform(robot.urdf.base_link,cfg.attached_to)
        self.limb_names = list(self.solvers.keys())
        self.total_dof = sum([self.solvers[limb_name].get_dof() for limb_name in self.limb_names])
        self.joint_names, self.out_joint_names = [], []
        for limb_name in self.limb_names:
            self.joint_names += self.solvers[limb_name].config.joint_names
            self.out_joint_names += np.array(robot.joint_names)[robot.ctrl_arm_joint_idx_mapping[limb_name]].tolist()
        self.ctrl_joint_names = robot.joint_names
        self.out_idx_mapping = self.get_ik_idx_mappings()
        self.set_qpos(robot.init_qpos)

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
            qpos_idx += dof


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

class MujocoIKSolver(BaseIKSolver):
    def __init__(self, robot_name, ik_config, env=None, env_joint_names=None, env_eef_name=None):
        self.config = ik_config
        if env is None:
            urdf_path = os.path.abspath(ik_config.urdf_path)
            self.env = MuJoCoParserClass(robot_name, rel_xml_path=str(urdf_path), VERBOSE=False)
            self.joint_names = ik_config.joint_names
            self.eef_name = ik_config.ee_link
        else:
            # TODO: Consider this env can contain multiple robots
            self.env = env
            self.joint_names = env_joint_names
            self.eef_name = env_eef_name

        self.robot_name = robot_name
        self.max_iter = getattr(ik_config, 'max_iter', 30)
        self.dt = ik_config.dt
        self.ik_damping = ik_config.ik_damping
        self.ik_eps = ik_config.eps

        self.idx_mapping = [self.env.joint_names.index(name) for name in self.joint_names]


    def reset(self):
        zero_pos = np.zeros(len(self.joint_names))
        self.set_current_qpos(zero_pos)
        return

    def solve(self, Rt):
        pos, R = Rt[:3,3], Rt[:3,:3]
        q_rev = self.env.data.qpos[self.env.rev_joint_idxs]
        for i in range(self.max_iter):
            J_pri, ik_err_pri = self.env.get_ik_ingredients(
                body_name=self.eef_name,
                p_trgt=pos,
                R_trgt=R,
            )
            if np.linalg.norm(ik_err_pri) < self.ik_eps:
                break
            dq_pri = self.env.damped_ls(J_pri, ik_err_pri, stepsize=1, eps=self.ik_eps, th=np.deg2rad(1.0))
            q_rev = self.env.data.qpos[self.env.rev_joint_idxs]
            q_rev = q_rev + dq_pri[:self.env.n_rev_joint]
            q_rev = np.clip(q_rev, self.env.rev_joint_mins, self.env.rev_joint_maxs)  # clip
            self.env.forward(q=q_rev, joint_idxs=self.env.rev_joint_idxs)
        return q_rev[self.idx_mapping]

    def compute_ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        self.env.forward(q=qpos, joint_idxs=self.env.rev_joint_idxs)
        p_ee = self.env.get_p_body(self.eef_name)
        R_ee = self.env.get_R_body(self.eef_name)
        ee_pose = np.eye(4)
        ee_pose[:3, :3] = R_ee
        ee_pose[:3, 3] = p_ee
        return ee_pose

    def set_current_qpos(self, qpos):
        self.qpos = np.zeros(len(self.env.joint_names))
        self.qpos[self.idx_mapping] = qpos
        self.env.forward(q=qpos, joint_idxs=self.idx_mapping)
        return

    def get_current_qpos(self):
        return self.qpos

    def get_joint_names(self): return self.joint_names

    def get_dof(self): return len(self.idx_mapping)

    def get_timestep(self): return self.dt
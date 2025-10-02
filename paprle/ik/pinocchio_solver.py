import os, sys
from paprle.ik.base import BaseIKSolver
from paprle.utils.misc import import_pinocchio
pin = import_pinocchio()
from typing import List, Optional, Dict
import numpy as np

class PinocchioIKSolverMultiWrapper(BaseIKSolver):
    def __init__(self, robot):
        self.robot_name = robot.name
        self.config = robot.ik_config
        self.solvers = {}
        self.limb_names = robot.limb_names
        self.base2world = {}
        for limb_name, cfg in self.config.items():
            self.solvers[limb_name] = PinocchioIKSolver(self.robot_name, cfg)
            self.base2world[limb_name] = robot.urdf.get_transform(robot.urdf.base_link,cfg.attached_to)
        self.limb_names = list(self.solvers.keys())
        self.total_dof = sum([self.solvers[limb_name].get_dof() for limb_name in self.limb_names])
        self.joint_names, self.out_joint_names = [], []
        for limb_name in self.limb_names:
            self.joint_names += self.solvers[limb_name].config.joint_names
            self.out_joint_names += np.array(robot.joint_names)[robot.ctrl_arm_joint_idx_mapping[limb_name]].tolist()
        self.ctrl_joint_names = robot.joint_names
        self.out_idx_mapping = self.get_ik_idx_mappings()
        self.qpos = np.zeros(len(self.out_joint_names))

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
        self.qpos = qpos


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


class PinocchioIKSolver(BaseIKSolver):
    def __init__(self, robot_name, ik_config):
        self.robot_name = robot_name
        self.config = ik_config

        self.ik_damping = ik_config.ik_damping * np.eye(6)
        self.ik_eps = ik_config.eps
        self.dt = ik_config.dt
        self.ee_name = ik_config.ee_link
        self.attached_to = ik_config.attached_to
        self.repeat = getattr(ik_config, 'repeat', 1)
        self.max_iter = getattr(ik_config, 'max_iter', 30)
        self.base_name = getattr(ik_config, 'base_link', '')

        self.urdf_path = os.path.abspath(ik_config.urdf_path)
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(str(self.urdf_path),
                                                                                      package_dirs=[ik_config.asset_dir])
        joint_names = self.get_joint_names()
        joints_to_lock, lock_positions = [], []
        for joint_name in joint_names:
            if joint_name not in self.config.joint_names:
                joints_to_lock.append(self.model.getJointId(joint_name))
            lock_positions.append(0.0)
        lock_positions = np.array(lock_positions)

        self.model = pin.buildReducedModel(self.model, joints_to_lock, lock_positions)
        self.data: pin.Data = self.model.createData()
        self.collision_data = self.collision_model.createData()

        frame_mapping: Dict[str, int] = {}
        for i, frame in enumerate(self.model.frames):
            frame_mapping[frame.name] = i

        if self.ee_name not in frame_mapping:
            raise ValueError(
                f"End effector name {self.ee_name} not find in robot with path: {urdf_path}."
            )

        self.frame_mapping = frame_mapping
        self.ee_frame_id = frame_mapping[self.ee_name]
        self.base_frame_id = frame_mapping[self.base_name] if self.base_name in frame_mapping else 0

        # Current state
        self.qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.ee_pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )
        self.ik_err = 0.0

        joint_names = self.get_joint_names()
        self.idx_mapping = [joint_names.index(name) for name in self.config.joint_names]

        # joint_limits
        self.lower_limit = np.array(self.model.lowerPositionLimit)
        self.upper_limit = np.array(self.model.upperPositionLimit)

    def reset(self):
        self.qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.ee_pose = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )
        self.ik_err = 0.0

        return

    def solve(self, Rt):
        oMdes = pin.SE3(Rt)
        qpos = self.qpos.copy()
        for k in range(self.max_iter):
            pin.forwardKinematics(self.model, self.data, qpos)
            ee_pose = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            J = pin.computeFrameJacobian(self.model, self.data, qpos, self.ee_frame_id)
            iMd = ee_pose.actInv(oMdes)
            err = pin.log(iMd).vector
            if np.linalg.norm(err) < self.ik_eps:
                break
            v = J.T.dot(np.linalg.solve(J.dot(J.T) + self.ik_damping, err))
            qpos = pin.integrate(self.model, qpos, v * self.dt)
        self.set_current_qpos(qpos, oMdes=oMdes)
        #print(f"IK solved in {k+1} iterations with error: {np.linalg.norm(err):.6f}")
        return qpos[self.idx_mapping]

    def compute_ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, qpos)
        oMf: pin.SE3 = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return np.array(oMf)

    def get_current_qpos(self) -> np.ndarray:
        return self.qpos.copy()

    def set_current_qpos(self, qpos: np.ndarray, oMdes=None):
        if len(qpos) != len(self.qpos):
            new_qpos = self.qpos.copy()
            new_qpos[self.idx_mapping] = qpos
            qpos = new_qpos
        self.qpos = qpos
        pin.forwardKinematics(self.model, self.data, self.qpos)
        if oMdes is not None:
            iMd = self.ee_pose.actInv(oMdes)
            err = np.linalg.norm(pin.log(iMd).vector)
            self.ik_err = err

    def get_ee_name(self) -> str:
        return self.ee_name

    def get_dof(self) -> int:
        return pin.neutral(self.model).shape[0]

    def get_timestep(self) -> float:
        return self.dt

    def get_joint_names(self) -> List[str]:

        try:
            # Pinocchio by default add a dummy joint name called "universe"
            names = list(self.model.names)
            return names[1:]
        except:
            names = []
            for f in self.model.frames:
                if f.type == pin.JOINT:
                    names.append(f.name)
            return names


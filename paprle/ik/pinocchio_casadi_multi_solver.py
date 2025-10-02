import os, sys
from paprle.ik.base import BaseIKSolver
from paprle.utils.misc import import_pinocchio
pin = import_pinocchio()
from typing import List, Optional, Dict
import numpy as np
import casadi
from pinocchio import casadi as cpin
from paprle.ik.pinocchio_casadi_single_solver import WeightedMovingFilter

class PinocchioCasadiMultiIKSolver(BaseIKSolver):
    def __init__(self, robot):
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

        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(str(urdf_file),
                                                                                      package_dirs=[asset_dir])
        joint_names = self.get_joint_names()
        joints_to_lock, lock_positions = [], []
        for joint_name in joint_names:
            if joint_name not in actuated_joint_names:
                joints_to_lock.append(self.model.getJointId(joint_name))
            lock_positions.append(0.0)
        lock_positions = np.array(lock_positions)

        self.model = pin.buildReducedModel(self.model,joints_to_lock, lock_positions)
        self.data: pin.Data = self.model.createData()
        self.collision_data = self.collision_model.createData()

        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        frame_mapping: Dict[str, int] = {}
        for i, frame in enumerate(self.model.frames):
            frame_mapping[frame.name] = i
        self.frame_mapping = frame_mapping

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.var_q_last = self.opti.parameter(self.model.nq)

        self.ee_frame_ids = []
        self.cTFs = []
        self.translational_error_args, self.rotational_error_args = [], []
        self.param_tfs = []
        for ee_id, ee_name in enumerate(ee_links):
            cTf = casadi.SX.sym(f"tf_{ee_id}", 4, 4)
            self.cTFs.append(cTf)
            self.ee_frame_ids.append(frame_mapping[ee_name])
            trans_arg = self.cdata.oMf[frame_mapping[ee_name]].translation - cTf[:3,3]
            rot_arg = cpin.log3(self.cdata.oMf[frame_mapping[ee_name]].rotation @ cTf[:3,:3].T)
            self.translational_error_args.append(trans_arg)
            self.rotational_error_args.append(rot_arg)
            param_tf = self.opti.parameter(4,4)
            self.param_tfs.append(param_tf)


        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, *self.cTFs],
            [
                casadi.vertcat(
                    *self.translational_error_args
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, *self.cTFs],
            [
                casadi.vertcat(
                    *self.rotational_error_args
                )
            ],
        )

        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, *self.param_tfs))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, *self.param_tfs))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit)
        )

        if getattr(robot.robot_config, 'prefer_qpos', None):
            prefer_qpos = robot.robot_config.prefer_qpos
            joint_names = self.get_joint_names()
            idxs, values = [], []
            for joint_name, value in prefer_qpos.items():
                idxs.append(joint_names.index(joint_name))
                values.append(value)
            self.prefer_cost = casadi.sumsqr(self.var_q[idxs] - np.array(values))
            self.opti.minimize(30 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost + 0.1 * self.prefer_cost)
        else:
            self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), len(actuated_joint_names))

        # Current state
        self.qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.ee_poses = []
        for ee_frame_id in self.ee_frame_ids:
            ee_pose: pin.SE3 = pin.updateFramePlacement(
                self.model, self.data, ee_frame_id
            )
            self.ee_poses.append(ee_pose)
        self.ik_err = 0.0

        joint_names = self.get_joint_names()
        self.idx_mapping = [joint_names.index(name) for name in actuated_joint_names]

        # joint_limits
        self.lower_limit = np.array(self.model.lowerPositionLimit)
        self.upper_limit = np.array(self.model.upperPositionLimit)

    def reset(self):
        self.qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.ee_poses = []
        for ee_frame_id in self.ee_frame_ids:
            ee_pose: pin.SE3 = pin.updateFramePlacement(
                self.model, self.data, ee_frame_id
            )
            self.ee_poses.append(ee_pose)
        self.ik_err = 0.0
        self.smooth_filter._data_queue = []
        return

    def solve(self, Rts):

        for ee_id in range(len(self.ee_frame_ids)):
            oMdes = pin.SE3(Rts[ee_id])
            self.opti.set_value(self.param_tfs[ee_id], np.array(oMdes))

        self.opti.set_initial(self.var_q, self.qpos)
        self.opti.set_value(self.var_q_last, self.qpos) # for smooth

        sol = self.opti.solve()
        # sol = self.opti.solve_limited()

        sol_q = self.opti.value(self.var_q)
        sol_q[np.isnan(sol_q)] = self.qpos[np.isnan(sol_q)]
        self.smooth_filter.add_data(sol_q[self.idx_mapping])
        sol_q = self.smooth_filter.filtered_data
        #v = (sol_q - self.init_data) * 0.0
        self.qpos[self.idx_mapping] = sol_q

        #sol_tauff = pin.rnea(self.model, self.data, sol_q, v,np.zeros(self.model.nv))
        return sol_q


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


    def get_ik_idx_mappings(self) -> Dict[str, np.ndarray]:
        out_indices = []
        for name in self.out_joint_names:
            out_indices.append(self.ctrl_joint_names.index(name))
        self.out_idx_mapping = out_indices
        return out_indices

    def set_qpos(self, ctrl_qpos):
        qpos = ctrl_qpos[self.out_idx_mapping]

        self.qpos = pin.neutral(self.model)
        self.qpos[self.idx_mapping] = qpos
        pin.forwardKinematics(self.model, self.data, self.qpos)
        return

    def compute_ee_poses(self, ctrl_qpos):
        qpos = ctrl_qpos[self.out_idx_mapping]

        self.qpos = pin.neutral(self.model)
        self.qpos[self.idx_mapping] = qpos
        pin.forwardKinematics(self.model, self.data, self.qpos)

        self.ee_poses = {}
        for limb_name, ee_frame_id in zip(self.limb_names, self.ee_frame_ids):
            ee_pose: pin.SE3 = pin.updateFramePlacement(
                self.model, self.data, ee_frame_id
            )
            self.ee_poses[limb_name] = ee_pose
        return self.ee_poses


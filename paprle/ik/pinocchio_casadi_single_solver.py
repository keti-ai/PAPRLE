import os, sys
from paprle.ik.base import BaseIKSolver
from paprle.utils.misc import import_pinocchio
pin = import_pinocchio()
from typing import List, Optional, Dict
import numpy as np
import casadi
from pinocchio import casadi as cpin


class WeightedMovingFilter:
    def __init__(self, weights, data_size=14):
        self._window_size = len(weights)
        self._weights = np.array(weights)
        assert np.isclose(np.sum(self._weights), 1.0), "[WeightedMovingFilter] the sum of weights list must be 1.0!"
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        temp_filtered_data = np.zeros(self._data_size)
        for i in range(self._data_size):
            temp_filtered_data[i] = np.convolve(data_array[:, i], self._weights, mode='valid')[-1]

        return temp_filtered_data

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return  # skip duplicate data

        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self):
        return self._filtered_data

class PinocchioCasadiIKSolverMultiWrapper(BaseIKSolver):
    def __init__(self, robot):
        self.robot_name = robot.name
        self.config = robot.ik_config
        self.solvers = {}
        self.limb_names = robot.limb_names
        self.base2world = {}
        for limb_name, cfg in self.config.items():
            self.solvers[limb_name] = PinocchioCasadiIKSolver(self.robot_name, cfg)
            self.base2world[limb_name] = robot.urdf.get_transform(robot.urdf.base_link,cfg.attached_to)
        self.limb_names = list(self.solvers.keys())
        self.total_dof = sum([self.solvers[limb_name].get_dof() for limb_name in self.limb_names])
        self.joint_names, self.out_joint_names = [], []
        for limb_name in self.limb_names:
            self.joint_names += self.solvers[limb_name].config.joint_names
            self.out_joint_names += np.array(robot.joint_names)[robot.ctrl_arm_joint_idx_mapping[limb_name]].tolist()
        self.ctrl_joint_names = robot.joint_names
        self.out_idx_mapping = self.get_ik_idx_mappings()

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

class PinocchioCasadiIKSolver(BaseIKSolver):
    def __init__(self, robot_name, ik_config):
        self.robot_name = robot_name
        self.config = ik_config

        self.ik_damping = ik_config.ik_damping
        self.ik_eps = ik_config.eps
        self.dt = ik_config.dt
        self.ee_name = ik_config.ee_link
        self.attached_to = ik_config.attached_to
        self.repeat = getattr(ik_config, 'repeat', 1)
        self.max_iter = getattr(ik_config, 'max_iter', 30)
        self.base_name = getattr(ik_config, 'base_link', '')

        urdf_path = os.path.abspath(ik_config.urdf_path)
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(str(urdf_path),
                                                                                      package_dirs=[ik_config.asset_dir])
        joint_names = self.get_joint_names()
        joints_to_lock, lock_positions = [], []
        for joint_name in joint_names:
            if joint_name not in self.config.joint_names:
                joints_to_lock.append(self.model.getJointId(joint_name))
            lock_positions.append(0.0)
        lock_positions = np.array(lock_positions)

        self.model = pin.buildReducedModel(self.model,joints_to_lock, lock_positions)


        self.data: pin.Data = self.model.createData()
        self.collision_data = self.collision_model.createData()

        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

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

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.ee_frame_id].translation - self.cTf[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.ee_frame_id].rotation @ self.cTf[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.var_q_last = self.opti.parameter(self.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit)
        )
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
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), len(self.config.joint_names))

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
        self.smooth_filter._data_queue = []
        return

    def solve(self, Rt):
        oMdes = pin.SE3(Rt)

        self.opti.set_initial(self.var_q, self.qpos)
        self.opti.set_value(self.param_tf, np.array(oMdes))
        self.opti.set_value(self.var_q_last, self.qpos) # for smooth

        sol = self.opti.solve()
        # sol = self.opti.solve_limited()

        sol_q = self.opti.value(self.var_q)
        self.smooth_filter.add_data(sol_q[self.idx_mapping])
        sol_q = self.smooth_filter.filtered_data
        #v = (sol_q - self.init_data) * 0.0
        self.qpos[self.idx_mapping] = sol_q

        #sol_tauff = pin.rnea(self.model, self.data, sol_q, v,np.zeros(self.model.nv))
        return sol_q

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
        #self.base_pose: pin.SE3 = pin.updateFramePlacement(
        #    self.model, self.data, self.base_frame_id
        #)
        self.ee_pose = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )
        #self.ee_pose = self.base_pose.actInv(self.ee_pose)#self.ee_pose.act(self.base_pose.inverse())
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


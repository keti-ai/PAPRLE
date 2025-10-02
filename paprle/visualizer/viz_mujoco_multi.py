from paprle.visualizer.mujoco_vis import MujocoViz
from paprle.envs.mujoco_env_utils.util import MultiSliderClass, r2rpy, rpy2r, r2quat
import numpy as np
import tkinter as tk
from functools import partial
from pytransform3d.rotations import euler_from_matrix, quaternion_from_euler, matrix_from_euler
from pytransform3d import transformations as pt
from paprle.envs.mujoco_env_utils.mujoco_parser import MuJoCoParserClass, init_ik_info
class MujocoMultiIKViz(MujocoViz):
    def __init__(self, robot, robot_names, spacing=0.5, env=None, verbose=False, xml_string=None):
        #super().__init__(robot, env, verbose, xml_string=xml_string)
        self.robot = robot
        self.viewer_args = robot.robot_config.viewer_args
        self.robot_names = robot_names
        self.num_robots = len(robot_names)
        self.spacing = spacing

        if env is None:
            self.env = MuJoCoParserClass(self.robot.name, rel_xml_path=robot.xml_path, VERBOSE=verbose, xml_string=xml_string)
        else:
            # don't recreate the env if we already have one
            self.env = env

        if 'base_pose' in self.robot.robot_config:
            self.base_mats = {limb_name: pt.transform_from_pq(pq) for limb_name, pq in self.robot.robot_config.base_pose.items()}
        else:
            self.base_mats = None
        # add ik targets
        self.ik_target_sliders = MultiSliderClass(
            title='IK Target',
            n_slider=6,
            window_width=450,
            window_height=400,
            slider_width=300,
            label_width=150,
            label_texts=['X', 'Y', 'Z', "Roll", "Pitch", "Yaw"],
            slider_mins=[-2, -2, -2, -np.pi*2, -np.pi*2, -np.pi*2],
            slider_maxs=[2, 2, 2, np.pi*2, np.pi*2, np.pi*2],
            verbose=False
        )

        button_frame = tk.Frame(self.ik_target_sliders.canvas)
        button_frame.pack(side=tk.BOTTOM)

        self.sim_joint_names = self.env.joint_names
        self.ctrl_joint_idxs, self.mimic_joints_info = [], []
        for i in range(self.num_robots):
            single_robot_joint_names = []
            for j in self.sim_joint_names:
                if j.startswith(f'r{i}'):
                    single_robot_joint_names.append(j.replace(f'r{i}_', ''))
            ctrl_joint_idxs, mimic_joints_info = self.robot.set_joint_idx_mapping(single_robot_joint_names)
            self.ctrl_joint_idxs.append(np.array(ctrl_joint_idxs) + i * len(single_robot_joint_names))
            self.mimic_joints_info.append(mimic_joints_info)

        self.set_qpos(robot.init_qpos)
        self.limb_targets = {}
        self.buttons = {}
        for limb_name in self.robot.limb_names:
            change_button = tk.Button(button_frame, text=limb_name, command=partial(self.change_eef, limb_name))
            change_button.pack(side=tk.LEFT)
            self.buttons[limb_name] = change_button

            p, R = self.env.get_pR_body('r0_' + self.robot.eef_names[limb_name])
            self.limb_targets[limb_name] = np.concatenate([p, euler_from_matrix(R, 0, 1, 2, extrinsic=False, strict_check=False)])

        reset_button = tk.Button(button_frame, text="RESET", command=self.reset_target)
        reset_button.pack(side=tk.LEFT)

        reset_button = tk.Button(button_frame, text="RESET_Q", command=self.reset_q)
        reset_button.pack(side=tk.LEFT)

         # set current limb

        self.curr_limb = self.robot.limb_names[0]
        for bn, button in self.buttons.items():
            if bn == self.curr_limb:
                button.config(bg='lightgreen')
            else:
                button.config(bg='lightgrey')
        self.ik_target_sliders.set_slider_values(self.limb_targets[self.curr_limb])
        self.reset_needed = False


    def reset_target(self):
        for i in range(self.num_robots):
            p_curr_eef, R_curr_eef = self.env.get_pR_body(f'r{i}_' + self.robot.eef_names[self.curr_limb])
            if i == 0:
                target = np.concatenate((p_curr_eef, r2rpy(R_curr_eef)))
                self.ik_target_sliders.set_slider_values(target)
        print(f"[MujocoIKViz] Reset {self.curr_limb} target")


    def reset_q(self):
        self.set_qpos(self.robot.init_qpos)
        for i in range(self.num_robots):
            for limb_name in self.robot.limb_names:
                p, R = self.env.get_pR_body(f"r{i}_" + self.robot.eef_names[limb_name])
                if i == 0:
                    self.limb_targets[limb_name] = np.concatenate((p, r2rpy(R)))
        self.ik_target_sliders.set_slider_values(self.limb_targets[self.curr_limb])
        self.reset_needed = True
        print(f"[MujocoIKViz] Reset qpos")

    def change_eef(self, limb_name):
        self.curr_limb = limb_name
        target = self.limb_targets[limb_name]
        self.ik_target_sliders.set_slider_values(target)
        for bn, button in self.buttons.items():
            if bn == self.curr_limb:
                button.config(bg='lightgreen')
            else:
                button.config(bg='lightgrey')
        print(f"[MujocoIKViz] Changed to {limb_name}")

    def render(self):
        self.env.render()
        self.ik_target_sliders.update()
        self.limb_targets[self.curr_limb] = self.ik_target_sliders.get_values()

        for i in range(self.num_robots):
            offset = np.array([0, i * self.spacing, 0, 0, 0, 0])
            p_curr_eef, R_curr_eef = self.env.get_pR_body(f'r{i}_'+self.robot.eef_names[self.curr_limb])
            self.env.plot_T(p=p_curr_eef, R=R_curr_eef, PLOT_AXIS=True, axis_len=0.1, axis_width=0.01,
                            axis_rgba=[[1.0, 0.0, 0.0, 0.7],
                                       [0.0, 1.0, 0.0, 0.7],
                                       [0.0, 0.0, 1.0, 0.7]])


            target = self.limb_targets[self.curr_limb] + offset
            self.env.plot_T(p=target[:3], R=matrix_from_euler(target[3:], 0, 1, 2, extrinsic=False), PLOT_AXIS=True, axis_len=0.1, axis_width=0.01,
                            axis_rgba=[[0.94117647, 0.32941176, 0.57254902, 0.7],
                                       [0.68235294, 0.9254902 , 0.25490196, 0.7],
                                       [0.41960784, 0.56862745, 0.85098039, 0.7]])

            err = np.linalg.norm(p_curr_eef - target[:3])
            self.env.plot_line_fr2to(p_curr_eef, target[:3], rgba=[0.5,0.5,0.5,0.7], label=f"err {err:.3f}")
            self.env.plot_T(p=offset[:3] + [0, 0, 0.5],R=np.eye(3,3), PLOT_AXIS=False,label=self.robot_names[i])
        self.env.plot_T(p=np.zeros(3),R=np.eye(3,3), PLOT_AXIS=True,axis_len=0.5,axis_width=0.005)

    def get_ik_targets(self):
        out_targets = {}
        for limb_name in self.robot.limb_names:
            p = self.limb_targets[limb_name][:3]
            euler = self.limb_targets[limb_name][3:]
            quat = quaternion_from_euler(euler, 0, 1, 2, extrinsic=False)
            out_targets[limb_name] = np.concatenate([p, quat])
        return out_targets

    def set_qpos(self, qpos):
        if len(qpos) == len(self.ctrl_joint_idxs[0]):
            qpos = np.tile(qpos, self.num_robots)
        new_qpos = np.zeros_like(self.env.data.qpos)
        for i in range(len(self.ctrl_joint_idxs)):
            new_qpos[self.ctrl_joint_idxs[i]] = qpos[i * len(self.ctrl_joint_idxs[0]): (i + 1) * len(self.ctrl_joint_idxs[0])]
            if len(self.mimic_joints_info[i]) > 0:
                ind1, ind2 = self.mimic_joints_info[i][:, 0].astype(np.int32), self.mimic_joints_info[i][:, 1].astype(np.int32)
                new_qpos[ind1] = new_qpos[ind2] * self.mimic_joints_info[i][:, 2] + self.mimic_joints_info[i][:, 3]
        self.curr_qpos = new_qpos
        self.env.forward(q=new_qpos, INCREASE_TICK=True)
        return
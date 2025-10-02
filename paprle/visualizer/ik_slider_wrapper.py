import numpy as np
import tkinter as tk
from paprle.envs.mujoco_env_utils.util import MultiSliderClass, r2rpy, rpy2r, r2quat
from functools import partial
from pytransform3d.rotations import euler_from_matrix, quaternion_from_euler, euler_from_quaternion
from pytransform3d.transformations import transform_from_pq

class IKSliderVizWrapper:
    def __init__(self, robot, viz):
        self.viz = viz
        self.robot = robot

        # add ik targets
        self.ik_target_sliders = MultiSliderClass(
            title='IK Target',
            n_slider=6,
            window_width=450,
            window_height=400,
            slider_width=300,
            label_width=150,
            label_texts=['X', 'Y', 'Z', "Roll", "Pitch", "Yaw"],
            slider_mins=[-2, -2, -2, -2*np.pi, -2*np.pi, -2*np.pi],
            slider_maxs=[2, 2, 2, 2*np.pi, 2*np.pi, 2*np.pi],
            verbose=False
        )

        button_frame = tk.Frame(self.ik_target_sliders.canvas)
        button_frame.pack(side=tk.BOTTOM)

        self.set_qpos(robot.init_qpos)
        limb_targets = robot.get_ee_transform(robot.init_qpos)
        self.limb_targets = {}
        self.buttons = {}
        for limb_name in self.robot.limb_names:
            change_button = tk.Button(button_frame, text=limb_name, command=partial(self.change_eef, limb_name))
            change_button.pack(side=tk.LEFT)
            self.buttons[limb_name] = change_button

            Rt = limb_targets[limb_name]
            self.limb_targets[limb_name] = np.concatenate([Rt[:3,3], euler_from_matrix(Rt[:3,:3], 0, 1, 2, extrinsic=False, strict_check=False)])

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
        self.last_slider_values = self.get_slider_values()
        self.last_qpos = np.array(robot.init_qpos)

        self.reset_needed = False
        return

    def init_viewer(self):
        return self.viz.init_viewer()

    def set_qpos(self, qpos):
        self.last_qpos = qpos
        return self.viz.set_qpos(qpos)

    def reset(self):
        return self.viz.reset()

    def is_alive(self): return self.viz.is_alive()

    def set_ee_targets(self, ee_targets):
        for limb_name, Rt in zip(self.robot.limb_names, ee_targets):
            p = Rt[:3,3]
            rpy = euler_from_matrix(Rt[:3,:3], 0, 1, 2, extrinsic=False, strict_check=False)
            self.limb_targets[limb_name] = np.concatenate([p, rpy])
        self.ik_target_sliders.set_slider_values(self.limb_targets[self.curr_limb])
        self.viz.set_ee_targets(ee_targets)
        self.last_ik_targets = self.viz.get_ik_targets()
        return

    def reset_target(self):
        limb_targets = self.robot.get_ee_transform(self.last_qpos)
        Rt = limb_targets[self.curr_limb]
        target = np.concatenate((Rt[:3,3], euler_from_matrix(Rt[:3,:3], 0, 1, 2, extrinsic=False, strict_check=False)))
        self.ik_target_sliders.set_slider_values(target)
        self.last_slider_values = self.get_slider_values()
        self.viz.set_ee_target(Rt, self.robot.limb_names.index(self.curr_limb))
        print(f"[IKViz] Reset {self.curr_limb} target")

    def reset_q(self):
        self.set_qpos(self.robot.init_qpos)
        limb_targets = self.robot.get_ee_transform(self.last_qpos)
        for limb_name in self.robot.limb_names:
            Rt = limb_targets[limb_name]
            self.limb_targets[limb_name] = np.concatenate((Rt[:3,3], euler_from_matrix(Rt[:3,:3], 0, 1, 2, extrinsic=False, strict_check=False)))
        self.ik_target_sliders.set_slider_values(self.limb_targets[self.curr_limb])
        self.reset_needed = True
        print(f"[IKViz] Reset qpos")

    def change_eef(self, limb_name):
        self.curr_limb = limb_name
        target = self.limb_targets[limb_name]
        self.ik_target_sliders.set_slider_values(target)
        # change button color
        for bn, button in self.buttons.items():
            if bn == limb_name:
                button.config(bg='lightgreen')
            else:
                button.config(bg='lightgrey')
        print(f"[IKViz] Changed to {limb_name}")

    def get_ik_targets(self):
        ik_targets = self.viz.get_ik_targets()
        # if sliders are moved, override the current limb target
        curr_val = self.get_slider_values()
        if not np.allclose(curr_val, self.last_slider_values):
            ik_targets[self.curr_limb] = np.concatenate([curr_val[:3], quaternion_from_euler(self.last_slider_values[3:], 0, 1, 2, extrinsic=False)])
            self.limb_targets[self.curr_limb] = curr_val
            self.viz.set_ee_target(transform_from_pq(ik_targets[self.curr_limb]), self.robot.limb_names.index(self.curr_limb))
        else:
            for limb_name in self.robot.limb_names:
                if not np.allclose(ik_targets[limb_name], self.last_ik_targets[limb_name]):
                    overwrite_pos = ik_targets[limb_name][:3]
                    overwrite_euler = euler_from_quaternion(ik_targets[limb_name][3:], 0, 1, 2, extrinsic=False)
                    self.limb_targets[limb_name] = np.concatenate([overwrite_pos, overwrite_euler])
            self.ik_target_sliders.set_slider_values(self.limb_targets[self.curr_limb])
        self.last_ik_targets = ik_targets.copy()
        self.last_slider_values = curr_val.copy()
        out_targets = {}
        for limb_name in self.robot.limb_names:
            p = self.limb_targets[limb_name][:3]
            quat = quaternion_from_euler(self.limb_targets[limb_name][3:], 0, 1, 2, extrinsic=False)
            out_targets[limb_name] = np.concatenate([p, quat])
        return out_targets

    def render(self):
        self.viz.render()
        self.ik_target_sliders.update()
        return

    def get_slider_values(self):
        vals = self.ik_target_sliders.get_values()
        euler = vals[3:]
        euler[euler > np.pi] -= 2*np.pi
        euler[euler < -np.pi] += 2*np.pi
        vals[3:] = euler
        return vals


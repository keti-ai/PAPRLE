import glob
import os
from yourdfpy.urdf import URDF
import numpy as np
from threading import Thread
import os


class LAFAN_Dataset:
    def __init__(self, robot, leader_config, env_config, render_mode='human', verbose=False, *args, **kwargs):
        self.is_ready = False
        self.require_end = False
        self.shutdown = False
        self.robot = robot
        self.leader_config = leader_config

        self.motion_file_list = glob.glob(os.path.join(leader_config.lafan_data_dir, leader_config.dataset_robot, '*.csv'))
        if len(self.motion_file_list) == 0:
            raise ValueError(f'No motion files found in {os.path.join(leader_config.lafan_data_dir, leader_config.dataset_robot, "*.csv")}')
        self.leader_robot = URDF.load(leader_config.dataset_robot_urdf)
        WAIST_LOCK = os.environ['WAIST_LOCK'] == '1'
        if WAIST_LOCK:
            self.lock_joints = []
            for jname in self.leader_robot.actuated_joint_names:
                if 'waist' in jname:
                    self.lock_joints.append(self.leader_robot.actuated_joint_names.index(jname))
        self.leader_robot_eef_names = leader_config.eef_names
        self.episode_idx = -1
        self.timestep = 0
        self.motion_scale = leader_config.motion_scale

        self.follower_limb_names = robot.limb_names
        self.follower_joint_names = self.robot.joint_names
        self.joint_mapping_idxs = []
        for joint_name in self.follower_joint_names:
            if joint_name in self.leader_robot.actuated_joint_names:
                self.joint_mapping_idxs.append(self.leader_robot.actuated_joint_names.index(joint_name))
            elif 'elbow_joint' in joint_name:
                joint_name = joint_name.replace('elbow_joint', 'elbow_pitch_joint')
                self.joint_mapping_idxs.append(self.leader_robot.actuated_joint_names.index(joint_name))
            elif 'wrist_roll_joint' in joint_name:
                joint_name = joint_name.replace('wrist_roll_joint', 'elbow_roll_joint')
                self.joint_mapping_idxs.append(self.leader_robot.actuated_joint_names.index(joint_name))
            else:
                print("Joint name not found in leader robot: ", joint_name)
        self.curr_qpos = np.zeros(self.leader_robot.num_actuated_joints)
        if render_mode:
            self.render_thread = Thread(target=self.__render_trimesh, args=())
            self.render_thread.start()
        return

    def __render_trimesh(self):
        self.leader_model = URDF.load(self.leader_config.dataset_robot_urdf)
        def callback(scene,  **kwargs ):
            self.leader_model.update_cfg(self.curr_qpos)
            # # To get current camera transform
            # print(self.leader_model._scene.camera_transform)
            # print(pt.pq_from_transform(self.leader_model._scene.camera_transform))
        self.leader_model._scene.show(
            callback=callback,
            flags={'grid': True}
        )


    def reset(self, ):
        self.timestep = 0
        return

    def launch_init(self, init_env_qpos):
        self.episode_idx = (self.episode_idx + 1) % len(self.motion_file_list)
        csv_file = self.motion_file_list[self.episode_idx]
        self.current_motion_data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        self.timestep = 0

        self.is_ready = True
        self.require_end = False
        self.leader_robot.update_cfg(np.zeros(self.leader_robot.num_actuated_joints))
        self.leader_eef_transforms = {}
        for eef_name in self.leader_robot_eef_names:
            self.leader_eef_transforms[eef_name] = self.leader_robot.get_transform(eef_name).copy()
        return

    def close_init(self):
        return

    def get_status(self):
        self.curr_qpos = self.current_motion_data[self.timestep][7:]
        if WAIST_LOCK:
            self.curr_qpos = np.array(self.curr_qpos)
            self.curr_qpos[self.lock_joints] = 0.0
        self.leader_robot.update_cfg(self.curr_qpos)

        out_eef_poses = {}
        for eef_name, follower_limb_name in zip(self.leader_robot_eef_names, self.follower_limb_names):
            curr_eef_transform = self.leader_robot.get_transform(eef_name)
            delta_transform = np.linalg.inv(self.leader_eef_transforms[eef_name]) @ curr_eef_transform
            delta_transform[:3, 3] *= self.motion_scale
            out_eef_poses[follower_limb_name] = delta_transform


        self.timestep += 1
        if self.timestep >= self.current_motion_data.shape[0]:
            self.require_end = True
            self.is_ready = False

        if self.timestep == 1:
            self.leader_eef_transforms = {}
            for eef_name in self.leader_robot_eef_names:
                self.leader_eef_transforms[eef_name] = self.leader_robot.get_transform(eef_name).copy()
            return {'command': self.curr_qpos[self.joint_mapping_idxs], 'command_type': 'joint_pos'}
        else:
            return out_eef_poses, {}

    def update_vis_info(self, env_vis_info):
        return env_vis_info

    def close(self):
        return

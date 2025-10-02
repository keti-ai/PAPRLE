import numpy as np
from yourdfpy.urdf import URDF

class Robot:
    def __init__(self, follower_config):
        self.follower_config = follower_config
        self.robot_config = follower_config.robot_cfg
        self.ik_solver = getattr(follower_config, 'ik_solver', 'pincasadi')
        self.ik_config = getattr(follower_config, 'ik_cfg', None)
        self.ros2_config = getattr(follower_config, 'ros2_cfg', None)
        self.camera_config = getattr(follower_config, 'camera_cfg', None)
        if self.camera_config is not None:
            camera_type = getattr(follower_config, 'camera_type', 'realsense')
            self.camera_type = camera_type

        self.name = self.robot_config.name

        self.control_dt = getattr(self.robot_config, 'control_dt', 0.008)
        self.joint_names = self.robot_config.ctrl_joint_names
        self.num_joints = len(self.joint_names)
        self.num_limbs =  self.robot_config.num_limbs
        self.eef_names = self.robot_config.end_effector_link

        # limb information
        self.limb_names = list(self.robot_config.limb_joint_names.keys())
        self.ctrl_joint_idx_mapping = self.robot_config.ctrl_joint_idx_mapping
        self.ctrl_joint_type = self.robot_config.ctrl_joint_type
        self.ctrl_arm_joint_idx_mapping = self.robot_config.ctrl_arm_joint_idx_mapping
        self.ctrl_hand_joint_idx_mapping = self.robot_config.ctrl_hand_joint_idx_mapping

        self.asset_dir = self.robot_config.asset_cfg.asset_dir
        self.urdf_file= self.robot_config.asset_cfg.urdf_path
        self.xml_path = self.robot_config.asset_cfg.xml_path
        self.urdf = URDF.load(self.urdf_file)
        self.joint_limits = []
        for joint_name in self.joint_names:
            self.joint_limits.append([self.urdf.joint_map[joint_name].limit.lower, self.urdf.joint_map[joint_name].limit.upper])
        self.joint_limits = np.array(self.joint_limits)

        self.init_qpos = np.array(getattr(self.robot_config, 'init_qpos', np.zeros(self.num_joints)))

    def get_ee_transform(self, ctrl_qpos):
        urdf_joint_names = self.urdf.actuated_joint_names
        qpos = np.zeros(len(urdf_joint_names))
        for id, name in enumerate(self.joint_names):
            qpos[urdf_joint_names.index(name)] = ctrl_qpos[id]
        self.urdf.update_cfg(qpos)
        out_ee_transforms = {}
        for limb_name in self.limb_names:
            ee_transform = self.urdf.get_transform(self.eef_names[limb_name], self.urdf.base_link)
            out_ee_transforms[limb_name] = ee_transform
        return out_ee_transforms

    def random_qpos(self):
        return  np.random.uniform(low=self.joint_limits[:, 0], high=self.joint_limits[:, 1])

    def set_joint_idx_mapping(self, env_joint_names):
        ctrl_joint_idxs = [env_joint_names.index(joint) for joint in self.joint_names]
        mimic_joints_info = []
        for joint_name in env_joint_names:
            if joint_name not in self.urdf.joint_map: continue
            if self.urdf.joint_map[joint_name].mimic:
                this_idx = env_joint_names.index(joint_name)
                mimic_idx = env_joint_names.index(self.urdf.joint_map[joint_name].mimic.joint)
                mimic_info = [this_idx, mimic_idx, self.urdf.joint_map[joint_name].mimic.multiplier, self.urdf.joint_map[joint_name].mimic.offset]
                mimic_joints_info.append(mimic_info)
        return ctrl_joint_idxs, np.array(mimic_joints_info)


if __name__ == '__main__':
    from configs import BaseConfig

    follower_config, leader_config, env_config = BaseConfig().parse()
    robot = Robot(follower_config)

    print("")
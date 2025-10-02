from omegaconf import OmegaConf
from paprle.utils.config_utils import change_working_directory, add_info_robot_config
change_working_directory()

from paprle.follower import Robot
from paprle.visualizer import VISUALIZER_DICT
from pytransform3d import transformations as pt
import numpy as np
from paprle.utils.mujoco_xml_utils import load_multiple_robot
import os
import jax
import time
import glob

robot_name = 'g1'
config_file = f'configs/follower/{robot_name}.yaml'
config = OmegaConf.load(config_file)
config.robot_cfg = add_info_robot_config(config.robot_cfg)

urdf_file = config.robot_cfg.asset_cfg.urdf_path
robot = Robot(config)

viz = VISUALIZER_DICT['viser'](robot)
viz.init_viewer()
viz.set_qpos(robot.init_qpos)

all_joints = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]

idx_mapping = [all_joints.index(name) for name in robot.joint_names]
motion_csv_list = glob.glob('~/lerobot_dataset/LAFAN1_Retargeting_Dataset/g1/*')
for csv_file in motion_csv_list:
    print(f'Processing {csv_file}')
    motion_data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    start_time = time.time()
    for i in range(motion_data.shape[0]):
        qpos = motion_data[i, 7:][idx_mapping]
        viz.set_qpos(qpos)
        viz.render()
        time.sleep(0.03)
    print(f'Finished {csv_file} in {time.time()-start_time:.2f} seconds')
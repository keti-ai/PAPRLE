from omegaconf import OmegaConf
from paprle.utils.config_utils import change_working_directory, add_info_robot_config
change_working_directory()
import argparse
from paprle.follower import Robot
from pytransform3d import transformations as pt
import numpy as np
from paprle.ik import IK_SOLVER_DICT
from paprle.visualizer import VISUALIZER_DICT
from paprle.visualizer.ik_slider_wrapper import IKSliderVizWrapper
parser = argparse.ArgumentParser()
parser.add_argument('--robot', type=str, default='ffw_sg2_follower', help='robot name')
parser.add_argument('--solver', type=str, default='mujoco', help='ik solver type',
                    choices=['mujoco', 'oscbf', 'pinocchio', 'pyroki', 'pincasadi', 'pincasadi_single'])
parser.add_argument('--viz', type=str, default='viser', help='visualizer type',
                    choices=['mujoco', 'pybullet', 'viser'])
args = parser.parse_args()

robot_name = args.robot
config_file = f'configs/follower/{robot_name}.yaml'
config = OmegaConf.load(config_file)
config.robot_cfg = add_info_robot_config(config.robot_cfg)

urdf_file = config.robot_cfg.asset_cfg.urdf_path
robot = Robot(config)

viz = VISUALIZER_DICT[args.viz](robot)
viz = IKSliderVizWrapper(robot,viz)
viz.init_viewer()
viz.set_qpos(robot.init_qpos)

ee_targets = list(robot.get_ee_transform(robot.init_qpos).values())
viz.set_ee_targets(ee_targets)

solver = IK_SOLVER_DICT[args.solver](robot)
idx_mapping = solver.get_ik_idx_mappings()
while viz.is_alive():
    if viz.reset_needed:
        viz.reset_needed = False
        solver.reset()

    ik_targets = viz.get_ik_targets()

    input_targets = []
    for limb_name in robot.limb_names:
        ik_target_pq = ik_targets[limb_name] # world -> eef
        ik_target = pt.transform_from_pq(ik_target_pq)
        input_targets.append(ik_target)
    out_qpos = solver.solve(input_targets)
    total_qpos = np.zeros([robot.num_joints])
    total_qpos[idx_mapping] = out_qpos
    viz.set_qpos(total_qpos)
    viz.render()

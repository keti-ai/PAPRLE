import numpy as np
from paprle.ik.pinocchio_solver import PinocchioIKSolver
def get_robot_joint_states(data, robot_names=[], keyword='pos'):
    robot_states = []
    for robot_name in robot_names:
        robot_state = np.concatenate([data['obs'][robot_name]['arm_state'][keyword],
                                      data['obs'][robot_name]['hand_state'][keyword]])
        robot_states.append(robot_state)
    robot_poses = np.concatenate(robot_states)
    return robot_poses

def get_command_to_qpos(data, robot_names=[]):
    target_qpos = []
    for i, robot_name in enumerate(robot_names):
        limb_name = robot_name
        target_qpos_for_each = np.concatenate([data['command'][0][limb_name],
                                               data['command'][1][limb_name]])
        target_qpos.append(target_qpos_for_each)
    target_qpos = np.concatenate(target_qpos)
    return target_qpos


def setup_fk_model(episode_info):
    ik_solvers, idx_mappings = [], []
    config = episode_info['robot_config']
    for i, (eef_name, eef_ik_info) in enumerate(config['ik_cfg'].items()):
        ik_solver = PinocchioIKSolver(eef_name, eef_ik_info)
        ik_solvers.append(ik_solver)
        idx_mapping = [[i, eef_ik_info.joint_names.index(name)] for i, name in enumerate(ik_solver.get_joint_names()) if
                       name in eef_ik_info.joint_names]
        idx_mapping = np.array(idx_mapping)
        idx_mappings.append(idx_mapping)
    return ik_solvers, idx_mappings



def get_ee_poses(fk_models, idx_mappings, target_qpos, arm_dof, hand_dof, robot_names):
    target_ee_poses = []
    for i in range(len(fk_models)):
        limb_name = robot_names[i]
        target_qpos_for_each = target_qpos[(arm_dof[limb_name] + hand_dof[limb_name]) * i:(arm_dof[limb_name] + hand_dof[limb_name]) * (i + 1)]
        new_qpos = fk_models[i].qpos.copy()
        new_qpos[idx_mappings[i][:, 0]] = target_qpos_for_each[idx_mappings[i][:, 1]]
        ee_pose = fk_models[i].compute_ee_pose(new_qpos)
        target_ee_poses.append(ee_pose)
    return np.array(target_ee_poses)
"""Script to visualize the hand-designed collision model of the Franka Panda in Pybullet"""
import time

import pybullet
import numpy as np
from omegaconf import OmegaConf
from paprle.utils.config_utils import change_working_directory, add_info_robot_config
change_working_directory()

from paprle.follower import Robot
from oscbf.core.manipulator import create_transform_numpy
from oscbf.utils.visualization import visualize_3D_sphere
from paprle.ik.oscbf_solver import OSCBFIKSolver
from paprle.envs.mujoco_env_utils.util import MultiSliderClass

def main():
    np.random.seed(0)

    robot_name = 'papras_orthrus'
    config_file = f'configs/follower/{robot_name}.yaml'
    config = OmegaConf.load(config_file)
    config.robot_cfg = add_info_robot_config(config.robot_cfg)

    robot = Robot(config)

    sliders = MultiSliderClass(
        title='Joint Angles',
        n_slider=robot.num_joints,
        window_width=450,
        window_height=800,
        slider_width=300,
        label_width=150,
        label_texts=robot.joint_names,
        slider_mins=robot.joint_limits[:,0],
        slider_maxs=robot.joint_limits[:,1],
        slider_vals=robot.init_qpos)


    urdf_files, collision_models = {}, {}
    solvers = {}
    for limb_name in robot.limb_names:
        config.ik_cfg[limb_name].oscbf = OmegaConf.create({
            'ee_pos_min': [-0.7, -0.50, 0.1],
            'ee_pos_max': [0.2, 0.50, 0.6],
            'wb_pos_min': [-5.0, -5.0, -5.0],
            'wb_pos_max': [5.0, 5.0, 5.0],
            'collision_pos': [[-0.5, 0.0, 0.2]],
            'collision_radii': [0.1],
        })

        ee_pos_min, ee_pos_max = np.array(config.ik_cfg[limb_name].oscbf.ee_pos_min), np.array(config.ik_cfg[limb_name].oscbf.ee_pos_max)
        ee_pos_mid = 0.5 * (ee_pos_min + ee_pos_max)
        ee_pos_len = (ee_pos_max - ee_pos_min)/2.

        collision_pos = np.array(config.ik_cfg[limb_name].oscbf.collision_pos)
        collision_radii = np.array(config.ik_cfg[limb_name].oscbf.collision_radii)

        solvers[limb_name] = OSCBFIKSolver(limb_name, config.ik_cfg[limb_name])
        solvers[limb_name].set_current_qpos(robot.init_qpos[robot.ctrl_arm_joint_idx_mapping[limb_name]])

        urdf_files[limb_name] = solvers[limb_name].config.urdf_path
        collision_models[limb_name] = solvers[limb_name].robot_collision_data


    pybullet.connect(pybullet.GUI)
    pybullet_robot = pybullet.loadURDF(
        config.robot_cfg.asset_cfg.urdf_path,
        useFixedBase=True,
        flags=pybullet.URDF_USE_INERTIA_FROM_FILE | pybullet.URDF_MERGE_FIXED_LINKS,
    )
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    for link_idx in range(solvers[limb_name].robot.num_joints):
        pybullet.changeVisualShape(pybullet_robot, link_idx, rgbaColor=(0, 0, 0, 0.5))

    q = robot.init_qpos
    N = pybullet.getNumJoints(pybullet_robot)
    valid_joint_ids = []
    for n in range(N):
        joint_info = pybullet.getJointInfo(pybullet_robot, n)
        joint_name = joint_info[1].decode()
        if joint_name in robot.joint_names:
            valid_joint_ids.append((n, robot.joint_names.index(joint_name)))
    for n, i in valid_joint_ids:
        pybullet.resetJointState(pybullet_robot, n, q[i])

    total_sphere_ids = []
    for limb_name in robot.limb_names:
        pybullet.stepSimulation()
        limb_q = robot.init_qpos[robot.ctrl_arm_joint_idx_mapping[limb_name]]
        limb_transform = robot.urdf.get_transform(solvers[limb_name].config.attached_to, robot.urdf.base_link)
        joint_transforms = solvers[limb_name].robot.joint_to_world_transforms(limb_q)
        joint_transforms = limb_transform @ joint_transforms

        # Determine the world-frame positions of the collision geometry
        for i in range(len(collision_models[limb_name]['positions'])):
            parent_to_world_tf = joint_transforms[i]
            num_collision_spheres = len(collision_models[limb_name]['positions'][i])
            for j in range(num_collision_spheres):
                collision_to_parent_tf = create_transform_numpy(
                    np.eye(3), collision_models[limb_name]['positions'][i][j]
                )
                collision_to_world_tf = parent_to_world_tf @ collision_to_parent_tf
                total_sphere_ids.append(
                    visualize_3D_sphere(
                        collision_to_world_tf[:3, 3], collision_models[limb_name]['radii'][i][j]
                    )
                )

    while True:
        sliders.update()
        q = sliders.get_values()
        for n, i in valid_joint_ids:
            pybullet.resetJointState(pybullet_robot, n, q[i])
        pybullet.stepSimulation()
        curr_sphere_id = 0
        for limb_name in robot.limb_names:
            pybullet.stepSimulation()
            limb_q = q[robot.ctrl_arm_joint_idx_mapping[limb_name]]
            limb_transform = robot.urdf.get_transform(solvers[limb_name].config.attached_to, robot.urdf.base_link)
            joint_transforms = solvers[limb_name].robot.joint_to_world_transforms(limb_q)
            joint_transforms = limb_transform @ joint_transforms

            # Determine the world-frame positions of the collision geometry
            for i in range(len(collision_models[limb_name]['positions'])):
                parent_to_world_tf = joint_transforms[i]
                num_collision_spheres = len(collision_models[limb_name]['positions'][i])
                for j in range(num_collision_spheres):
                    collision_to_parent_tf = create_transform_numpy(
                        np.eye(3), collision_models[limb_name]['positions'][i][j]
                    )
                    collision_to_world_tf = parent_to_world_tf @ collision_to_parent_tf

                    sphere_id = total_sphere_ids[curr_sphere_id]
                    pybullet.resetBasePositionAndOrientation(sphere_id, collision_to_world_tf[:3, 3], [0,0,0,1])
                    pybullet.resetBaseVelocity(sphere_id, [0,0,0], [0,0,0])
                    curr_sphere_id += 1
        time.sleep(0.01)
    #input("Press Enter to exit")


if __name__ == "__main__":
    main()

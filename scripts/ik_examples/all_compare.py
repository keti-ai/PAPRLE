from omegaconf import OmegaConf
from paprle.utils.config_utils import change_working_directory, add_info_robot_config
change_working_directory()

from paprle.follower import Robot
from paprle.visualizer.viz_mujoco_multi import MujocoMultiIKViz
from pytransform3d import transformations as pt
import numpy as np
from paprle.utils.mujoco_xml_utils import load_multiple_robot
import os
import jax
import time

def orientation_error_R(Rc, Rd):
    Re = Rc.T @ Rd
    skew = 0.5 * (Re - Re.T)
    return np.array([skew[2,1], skew[0,2], skew[1,0]])

robot_name = 'g1'
config_file = f'configs/follower/{robot_name}.yaml'
config = OmegaConf.load(config_file)
config.robot_cfg = add_info_robot_config(config.robot_cfg)
if 'multi_ik_cfg' not in config:
    config.multi_ik_cfg = config.ik_cfg
    for limb_name in config.multi_ik_cfg.keys():
        config.multi_ik_cfg[limb_name].urdf_path = os.path.abspath(config.robot_cfg.asset_cfg.urdf_path)
        config.multi_ik_cfg[limb_name].ee_link = config.robot_cfg.end_effector_link[limb_name]
        config.multi_ik_cfg[limb_name].joint_names = config.robot_cfg.limb_joint_names[limb_name]
        config.multi_ik_cfg[limb_name].attached_to = ''

robot_names = ['casadi', 'pyroki']
num_robots = len(robot_names)
spacing = 1.5

robot = Robot(config)
from paprle.ik.oscbf_solver import OSCBFIKSolverMultiWrapper
solvers = []
for robot_name in robot_names:
    try:
        if robot_name == 'casadi':
            from paprle.ik.pinocchio_casadi_multi_solver import PinocchioCasadiMultiIKSolver
            solvers.append(PinocchioCasadiMultiIKSolver(robot))
        elif robot_name == 'pinocchio':
            from paprle.ik.pinocchio_solver import PinocchioIKSolverMultiWrapper
            solvers.append(PinocchioIKSolverMultiWrapper(robot))
        elif robot_name == 'pyroki':
            from paprle.ik.pyroki_solver import PyrokiIKSolver
            solvers.append(PyrokiIKSolver(robot))
        elif robot_name == 'oscbf':
            from paprle.ik.oscbf_solver import OSCBFIKSolverMultiWrapper
            solvers.append(OSCBFIKSolverMultiWrapper(robot))
        elif robot_name == 'mujoco':
            from paprle.ik.mujoco_solver import MujocoIKSolverMultiWrapper
            solvers.append(MujocoIKSolverMultiWrapper(robot))
    except Exception as e:
        print(f"Could not import solver for {robot_name}: {e}")
        raise e

xml_text = load_multiple_robot(config.robot_cfg.asset_cfg.xml_path, num_robots, spacing)


total_joint_names = robot.joint_names
ik_joint_names = []
for limb_name in robot.limb_names:
    ik_joint_names += robot.robot_config.limb_joint_names[limb_name]
idx_mapping = [total_joint_names.index(name) for name in ik_joint_names]

viz = MujocoMultiIKViz(robot, robot_names=robot_names, xml_string=xml_text, spacing=spacing)
viz.init_viewer(viewer_width=1822, viewer_height=626)
viz.update_viewer(
    azimuth=179.36502095337903,
    distance=2.5762459184854127,
    elevation=0.7593691536331677,
    lookat = [1.14688727, 3.15996207, 0.98863539],

)
iterations = 0
time_dict = {name: [] for name in robot_names}
err_dict = {name: [] for name in robot_names}
while viz.is_alive():
    if viz.reset_needed:
        viz.reset_needed = False
        for solver in solvers: solver.reset()

    ik_targets = viz.get_ik_targets()
    input_ik_targets = []
    for limb_name in robot.limb_names:
        ik_target_pq = ik_targets[limb_name]  # world -> eef
        ik_target = pt.transform_from_pq(ik_target_pq)
        input_ik_targets.append(ik_target)


    total_qpos = []
    for id, solver in enumerate(solvers):
        if isinstance(solver, PyrokiIKSolver):
            jax.config.update("jax_enable_x64", False)
        elif isinstance(solver, OSCBFIKSolverMultiWrapper):
            jax.config.update("jax_enable_x64", True)

        start_time = time.time()
        sol_qpos = solver.solve(input_ik_targets)
        loop_time = time.time() - start_time
        if iterations > 10:
            time_dict[robot_names[id]].append(loop_time)

        out_qpos = np.zeros(len(total_joint_names))
        out_qpos[idx_mapping] = sol_qpos
        total_qpos.append(out_qpos)

    total_qpos = np.concatenate(total_qpos)
    viz.set_qpos(total_qpos)

    if iterations > 10:
        for idd, qpos in enumerate(total_qpos.reshape([num_robots, -1])):
            actual_ee_mat = robot.get_ee_transform(qpos)
            all_errors = []
            for limb_name in robot.limb_names:
                ik_target = pt.transform_from_pq(ik_targets[limb_name])
                rot_err = orientation_error_R(actual_ee_mat[limb_name][:3,:3], ik_target[:3,:3])
                err = ik_target[:3, 3] - actual_ee_mat[limb_name][:3, 3]
                total_err = np.linalg.norm(np.concatenate([err, rot_err]))
                all_errors.append(total_err)
            total_err = np.linalg.norm(all_errors)
            err_dict[robot_names[idd]].append(total_err)

    viz.render()
    iterations += 1
    if iterations > 10 and iterations % 10 == 0:
        log_str = f"Iteration {iterations}: "
        for name, times in time_dict.items():
            times = np.array(times)
            log_str += f"{name} mean: {np.mean(times)*1000:.2f} ms, err: {np.mean(err_dict[name]):.4f} | "
        print(log_str, end='\r')
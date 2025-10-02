import sys
import signal
import time
import cv2
from threading import Thread
import warnings

from configs import BaseConfig
follower_config, leader_config, env_config = BaseConfig().parse()
if env_config.name == 'isaacgym':
    import isaacgym
from paprle.teleoperator import Teleoperator
from paprle.follower import Robot
from paprle.leaders import LEADERS_DICT
from paprle.envs import ENV_DICT
import numpy as np
from threading import Thread
import os
import h5py
import json
from omegaconf import OmegaConf
from pytransform3d import transformations as pt

def orientation_error_R(Rc, Rd):
    Re = Rc.T @ Rd
    skew = 0.5 * (Re - Re.T)
    return np.array([skew[2,1], skew[0,2], skew[1,0]])

SOLVER_NAME = os.environ['SOLVER_NAME']
if SOLVER_NAME in ['pinocchio', 'oscbf', 'mujoco']:
    os.environ['WAIST_LOCK'] = '1'  # Lock the waist joints for these solvers
WAIST_LOCK = os.environ['WAIST_LOCK'] == '1'
TIME_DEBUG = False
class Runner:
    def __init__(self, robot_config, leader_config, env_config):
        self.robot_config, self.leader_config, self.env_config = robot_config, leader_config, env_config
        self.robot_config.ik_solver = SOLVER_NAME
        self.TELEOP_DT = robot_config.robot_cfg.teleop_dt = leader_config.teleop_dt
        self.robot = Robot(robot_config)
        self.leader = LEADERS_DICT[leader_config.type](self.robot, leader_config, env_config, render_mode=env_config.render_leader) # Get signals from teleop devices, outputs joint positions or eef poses as teleop commands.
        self.teleop = Teleoperator(self.robot, leader_config, env_config, render_mode=env_config.render_teleop) # Solving IK for joint positions if not already given, check collision, and output proper joint positions.


    def shutdown_handler(self, sig, frame):
        print("Shutting down the system..")
        self.env.close()
        print("🚫🌏 Env closed")
        self.teleop.close()
        print("🚫🤖 Teleop closed")
        self.leader.close()
        print("🚫🎮 Leader closed")
        sys.exit()

    def log_time(self, msg=''):
        if self.last_log_time is not None and msg != '':
            print(msg, time.time() - self.last_log_time)
        self.last_log_time = time.time()
        return

    def render_thread_func(self):
        while True:
            if self.env.view_im is None or self.reset:
                time.sleep(0.01)
                continue
            cv2.imshow("View", self.env.view_im[:,:,::-1])
            key = cv2.waitKey(1)
            if key == ord("r"):
                self.reset = True
                print("Reset signal detected")

            if key == ord("q"):
                self.shutdown = True
                print("Shutting down signal detected")
                self.shutdown_handler(None, None)
        return
    def run(self):
        exp_name = f'{SOLVER_NAME}_{self.leader_config.name}_offcol{self.env_config.off_collision}_waistlock{WAIST_LOCK}'
        print(f"Experiment name: {exp_name}")
        from tqdm import tqdm
        with h5py.File(f'./ik_analysis_{exp_name}.hdf5', 'w') as root:
            root.attrs['robot_config'] = json.dumps(OmegaConf.to_container(self.robot_config))
            root.attrs['device_config'] = json.dumps(OmegaConf.to_container(self.leader_config))
            root.attrs['env_config'] = json.dumps(OmegaConf.to_container(self.env_config))
            err_dataset = root.create_dataset('errors', (len(self.leader.motion_file_list), 10000, len(self.robot.limb_names)), maxshape=(len(self.leader.motion_file_list), None, len(self.robot.limb_names)), dtype='float32', compression='gzip', compression_opts=4)
            time_dataset = root.create_dataset('times', (len(self.leader.motion_file_list), 10000), maxshape=(len(self.leader.motion_file_list), None), dtype='float32', compression='gzip', compression_opts=4)
            coll_dataset = root.create_dataset('collisions', (len(self.leader.motion_file_list), 10000), maxshape=(len(self.leader.motion_file_list), None), dtype='bool', compression='gzip', compression_opts=4)
            qpos_dataset = root.create_dataset('qpos', (len(self.leader.motion_file_list), 10000, self.robot.num_joints), maxshape=(len(self.leader.motion_file_list), None, self.robot.num_joints), dtype='float32', compression='gzip', compression_opts=4)
            ee_target_datset = root.create_dataset('ee_targets', (len(self.leader.motion_file_list), 10000, len(self.robot.limb_names), 7), maxshape=(len(self.leader.motion_file_list), None, len(self.robot.limb_names), 7), dtype='float32', compression='gzip', compression_opts=4)
            for ep in range(len(self.leader.motion_file_list)):
                init_env_qpos = self.robot.init_qpos  # Move the robot to the default position
                self.teleop.reset(init_env_qpos)
                shutdown = self.leader.launch_init(init_env_qpos) # Wait in the initialize function until the leader is ready (for visionpro and gello)

                initial_command = self.leader.get_status()
                initial_qpos = self.teleop.step(initial_command, initial=True) # process initial command

                timestep = 0
                pbar = tqdm(total=self.leader.current_motion_data.shape[0], desc=f"Episode {ep+1}/{len(self.leader.motion_file_list)}")
                errs, times = [], []
                while True:
                    pbar.update(1)
                    if TIME_DEBUG:
                        print("===========================")
                        self.log_time('Start Loop')

                    command = self.leader.get_status()

                    # If reset signal is detected, reset the environment
                    if self.leader.require_end:
                        self.leader.require_end = False
                        break

                    start_time = time.time()
                    qposes = self.teleop.step(command)
                    loop_time = time.time() - start_time

                    solved_ee_targets = self.robot.get_ee_transform(qposes)
                    all_limb_errs = []
                    for limb_id, limb_name in enumerate(self.robot.limb_names):
                        target_Rt = self.teleop.target_ee_poses[limb_id]
                        curr_Rt = solved_ee_targets[limb_name]
                        pos_err = curr_Rt[:3, 3] - target_Rt[:3, 3]
                        rot_err = orientation_error_R(curr_Rt[:3, :3], target_Rt[:3, :3])
                        err = np.linalg.norm(np.concatenate([pos_err, rot_err]))
                        all_limb_errs.append(err)

                    err_dataset[ep, timestep, :] = np.array(all_limb_errs)
                    time_dataset[ep, timestep] = loop_time
                    coll_dataset[ep, timestep] = bool(self.teleop.detected_collisions)
                    qpos_dataset[ep, timestep, :] = qposes
                    ee_target_datset[ep, timestep, :, :] = np.array([pt.pq_from_transform(self.teleop.target_ee_poses[i]) for i in range(len(self.robot.limb_names))])
                    timestep += 1





if __name__ == "__main__":
    runner = Runner(follower_config, leader_config, env_config)
    runner.run()

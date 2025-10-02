from paprle.utils.config_utils import change_working_directory
change_working_directory()
import sys
import glfw
import rclpy
import pinocchio as pin
from sensor_msgs.msg import JointState
from omegaconf import OmegaConf
import argparse
import numpy as np
from pytransform3d import transformations as pt
import time
from threading import Thread
import signal
from paprle.envs.mujoco_env_utils.mujoco_parser import MuJoCoParserClass

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', '-c', type=str, default='configs/leader/puppeteers/puppeteer_g1.yaml')
args = parser.parse_args()

print("Running with the following device: ", args.config)


cfg = OmegaConf.load(args.config)

class Device:
    def __init__(self, cfg):
        self.cfg = cfg
        self.shutdown = False
        self.curr_mode = 1 # 0: visualization follows the leader, 1: visualization is fixed
        self.leader_joint_names = None
        self.positions = None
        self.cube_half_size = [0.05, 0.05, 0.05] # just default value, will be overwritten in main loop
        self.specified_poses = np.array(cfg.reset_pose, dtype=np.float32) if hasattr(cfg, 'reset_pose') else None

        # Setup device model
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(self.cfg.asset_cfg.urdf_path,
                                                                                      package_dirs=[self.cfg.asset_cfg.asset_dir])
        self.data = self.model.createData()
        self.pin_model_joint_names = [name for name in self.model.names]
        if 'universe' in self.pin_model_joint_names:
            self.pin_model_joint_names.remove('universe')
        
        # For now only support single DOF gripper
        self.gripper_joint_idxs = [self.pin_model_joint_names.index(name[0]) for name in self.cfg.hand_joint_names.values() if len(name) > 0]
        self.gripper_ranges = np.array([self.cfg.hand_limits[key][0] for key in self.cfg.hand_joint_names.keys() if len(self.cfg.hand_joint_names[key]) > 0])
        frame_mapping = {}
        for i, frame in enumerate(self.model.frames):
            frame_mapping[frame.name] = i
        self.end_effector_frame_ids = [frame_mapping[name] for name in self.cfg.end_effector_link.values()]
        self.motion_scale = self.cfg.motion_scale
        self.neutral_pos = getattr(self.cfg, 'neutral_pos', pin.neutral(self.model))
        self.neutral_pos = np.array(self.neutral_pos, dtype=np.float32)
        # Setup joint states subscriber
        rclpy.init()
        self.node = rclpy.create_node('gello_listener')
        self.sub = self.node.create_subscription(JointState, self.cfg.leader_subscribe_topic, self.joint_state_callback, 10)
        def spin_thread():
            while not self.shutdown:
                rclpy.spin_once(self.node, timeout_sec=0.1)
        self.thread_spin = Thread(target=spin_thread, daemon=True)
        self.thread_spin.start()


        signal.signal(signal.SIGINT, self.shutdown_handler)
        self.thread = Thread(target=self.trigger)
        self.thread.start()

        self.viz = MuJoCoParserClass("Leader" , rel_xml_path=self.cfg.asset_cfg.urdf_path, VERBOSE=False)
        self.viz.init_viewer()
        self.viz_joint_idx_mappings = [self.viz.joint_names.index(name) for name in self.pin_model_joint_names]

        #self.specified_poses = [None] * len(self.end_effector_frame_ids)
        self.curr_ee_poses = [None] * len(self.end_effector_frame_ids)


    def shutdown_handler(self, sig, frame):
        print("Shutting down the system..")
        print("====================================")
        log_str= 'reset_pose: ['
        for i, pose in enumerate(self.specified_poses):
            if pose is not None:
                log_str += f"[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}], "
                if i == len(self.specified_poses) - 1:
                    log_str = log_str[:-2] + ']'
        log_str += f"\nreset_cube: [{self.cube_half_size[0]:.3f}, {self.cube_half_size[1]:.3f}, {self.cube_half_size[2]:.3f}]"
        print(log_str)
        self.shutdown = True
        self.viz.viewer.close()
        self.thread.join()
        sys.exit()


    def trigger(self):
        self.mode_change_progress = 0.0
        self.free_to_change = True
        while not self.shutdown:
            if self.positions is None: continue
            gripper_pose = self.positions[self.gripper_joint_idxs]
            norm_grasp = (gripper_pose - self.gripper_ranges[:, 0]) / (self.gripper_ranges[:, 1] - self.gripper_ranges[:, 0])
            triggered = (norm_grasp > 0.9).all()
            if not self.free_to_change and triggered:
                print("Release the gripper")
            elif not self.free_to_change and not triggered:
                self.free_to_change = True
            elif triggered:
                self.mode_change_progress += 0.003
            else:
                self.mode_change_progress = 0.0

            if self.mode_change_progress > 1.0:
                self.curr_mode = 1 - self.curr_mode
                self.mode_change_progress = 0.0
                print(f"Mode changed to {self.curr_mode}")
                self.free_to_change = False
            time.sleep(0.01)

        return

    def joint_state_callback(self, msg):
        if self.leader_joint_names is None:
            self.leader_joint_names = msg.name
            self.idx_mappings = []
            for name in self.leader_joint_names:
                if name not in self.pin_model_joint_names:
                    print(f"Warning: {name} not found in the model")
                    continue
                idx = self.pin_model_joint_names.index(name)
                self.idx_mappings.append(idx)

        new_positions = np.zeros(len(self.pin_model_joint_names))
        new_positions[self.idx_mappings] = msg.position
        self.positions = new_positions
        self.leader_states = self.positions

    def set_qpos(self, qpos):
        viz_qpos = np.zeros(len(self.viz.joint_names))
        viz_qpos[self.viz_joint_idx_mappings] = qpos
        self.viz.forward(viz_qpos)
        for idx in range(len(self.end_effector_frame_ids)):
            if self.specified_poses[idx] is not None and self.curr_ee_poses[idx] is not None:
                self.viz.plot_T(p=self.curr_ee_poses[idx][:3, 3], R=self.curr_ee_poses[idx][:3, :3], label=f"EEF {idx}", PLOT_AXIS=True, PLOT_SPHERE=True, sphere_r=0.02, axis_len=0.12, axis_width=0.005)
                inside_cube = np.all(np.abs(self.specified_poses[idx] - self.curr_ee_poses[idx][:3, 3]) < np.array(self.cube_half_size))
                color = [0, 1, 0, 0.3] if inside_cube else [1, 0, 0, 0.3]
                self.viz.plot_box(p=self.specified_poses[idx][:3], xlen=self.cube_half_size[0], ylen=self.cube_half_size[1], zlen=self.cube_half_size[2], rgba=color)
        self.viz.render()

    def get_eef_poses(self, pin_qpos):
        pin.forwardKinematics(self.model, self.data, pin_qpos)
        eef_poses = []
        for frame_id in self.end_effector_frame_ids:
            oMf: pin.SE3 = pin.updateFramePlacement(self.model, self.data, frame_id)
            xyzw_pose = pin.SE3ToXYZQUAT(oMf)
            pose = np.concatenate(
                [
                    xyzw_pose[:3],
                    np.array([xyzw_pose[6], xyzw_pose[3], xyzw_pose[4], xyzw_pose[5]]),
                ]
            )
            Rt = pt.transform_from_pq(pose)
            eef_poses.append(Rt)
        self.curr_ee_poses = eef_poses
        return eef_poses

if __name__ == '__main__':
    device = Device(cfg)
    device.cube_half_size = [0.2, 0.1, 0.1]

    # # TODO: The values are very specific to 7dof papras leaders, need to make it more general
    iteration, hz = 0, 100
    np.printoptions(precision=3, suppress=True)

    print("=====================================")
    print(f"Grasp the gripper for 3 secs to change the mode. Current mode: {device.curr_mode}")
    print("Mode: 0 - ending zone visualization follows the leader, 1 - visualization is fixed")
    print(f"If you want to change the size of the ending zone cubes, change line 165 of this file. device.cube_half_size = {device.cube_half_size}")
    print("=====================================")

    while True:
        iteration += 1
        start_time = time.time()
        if device.leader_joint_names is None:
            if iteration % hz == 0: # for every second
                print("Waiting for leader joint names")
            continue
        device.set_qpos(device.positions)
        new_eef_Rts = device.get_eef_poses(device.positions)
        log_str = f'Curr mode: {device.curr_mode} '
        for i, eef_Rt in enumerate(new_eef_Rts):
            eef_Rt = np.round(eef_Rt, decimals=3)
            log_str += f"EEF {i} pose: {eef_Rt[:3,3]} "
            if device.curr_mode == 0:
                device.specified_poses[i] = eef_Rt[:3,3]
        if device.mode_change_progress > 0.0:
            log_str = f"Mode change in progress - {device.mode_change_progress:.2f}  " + log_str

        if iteration % (hz * 5) == 0:
            print(log_str)
            print("Joint positions:")
            for i, name in enumerate(device.leader_joint_names):
                print(f"{name}: {device.positions[i]:.3f}")
            print("----------------------")
        else:
            print(log_str + '\r', end='')

        loop_time = time.time() - start_time
        time.sleep(max(1/hz - loop_time, 0.0))

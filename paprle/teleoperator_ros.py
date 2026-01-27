import warnings
warnings.filterwarnings('ignore')

import time
from threading import Thread, Lock
import numpy as np
from pytransform3d import transformations as pt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from paprle.ik import IK_SOLVER_DICT
from paprle.hands import BaseHand, ParallelGripper, PowerGripper
from paprle.collision import MujocoCollisionChecker
from paprle.visualizer.mujoco_vis import MujocoViz


class Teleoperator:
    def __init__(self, robot, leader_config, env_config, render_mode=False):
        self.shutdown, self.viz = False, None
        self.threads = []

        self.robot = robot
        self.collision_checker = MujocoCollisionChecker(robot)
        self.detected_collisions = False
        self.collision_list = [[],[]]


        self.joint_limits = self.robot.joint_limits
        self.max_joint_diff = robot.robot_config.max_joint_vel * robot.robot_config.teleop_dt

        self.leader_config = leader_config
        self.env_config = env_config
        self.check_collision = not env_config.off_collision
        self.command_type = leader_config.output_type  # joint_pos or eef_pose

        self.init_qpos = robot.init_qpos

        self.ik_idx_mappings, self.curr_ee_poses = {}, {}
        self.ik_solvers, self.init_world2ees, self.world2bases, self.base2ees = {}, {}, {}, {}
        if self.command_type == 'delta_eef_pose':
            self.ik_solver = IK_SOLVER_DICT[self.robot.ik_solver](self.robot)
            self.ik_idx_mappings = self.ik_solver.get_ik_idx_mappings()
            self.ik_solver.set_qpos(self.init_qpos)
            self.init_world2ees = self.ik_solver.compute_ee_poses(self.init_qpos)

        self.pos_lock = Lock()
        self.last_target_qpos = self.init_qpos
        self.target_ee_poses = None

        self.hand_solvers = {}
        self.eef_type = self.robot.robot_config.eef_type # hand or gripper
        for i, limb_name in enumerate(self.robot.limb_names):
            if self.eef_type == 'parallel_gripper':
                hand_solver = ParallelGripper(self.robot.robot_config, self.leader_config, self.env_config)
            elif self.eef_type == 'power_gripper':
                name = list(self.robot.robot_config.retargeting)[i]
                retargeting_config = self.robot.robot_config.retargeting[name]
                hand_solver = PowerGripper(self.robot.robot_config, self.leader_config, self.env_config, retargeting_config, joint_limits=self.joint_limits)
            elif self.eef_type == None:
                hand_solver = BaseHand(self.robot.robot_config, self.leader_config, self.env_config)
            else:
                raise ValueError('Unknown end effector type: %s' % self.eef_type)
            self.hand_solvers[limb_name] = hand_solver

        self.vis_info = {
            'collision_list': self.collision_list,
            'qpos': self.last_target_qpos,
        }
        self.render_mode = render_mode
        if self.render_mode:
            self.viz_thread = Thread(target=self.render_thread)
            self.viz_thread.start()
            self.threads.append(self.viz_thread)
        return

    def render_thread(self):
        while True:
            if self.shutdown:
                if self.viz is not None:
                    self.viz.env.close_viewer()
                return
            if self.render_mode:
                if self.viz is None:
                    self.viz = MujocoViz(self.robot)
                    viewer_args = self.robot.robot_config.viewer_args.mujoco
                    self.viz.init_viewer(viewer_title='Teleoperator',
                                         viewer_width=getattr(viewer_args, 'viewer_width', 1200),
                                         viewer_height=getattr(viewer_args, 'viewer_height', 800),
                                         viewer_hide_menus=True)
                    self.viz.update_viewer(**viewer_args)
                # print(
                #     f"azimuth: {self.viz.env.viewer.cam.azimuth}\n"
                #     f"distance: {self.viz.env.viewer.cam.distance}\n"
                #     f"elevation: {self.viz.env.viewer.cam.elevation}\n"
                #     f"lookat: {self.viz.env.viewer.cam.lookat.tolist()}")
                if self.viz is not None:
                    if self.target_ee_poses is not None:
                        target_ee_poses = self.target_ee_poses
                        self.viz.set_ee_targets(target_ee_poses)
                    if self.last_target_qpos is not None:
                        target_qpos = self.last_target_qpos.copy()
                        self.viz.set_qpos(target_qpos)
                    #self.viz.log = self.vis_log
                    self.viz.render()
                    self.last_image = self.viz.env.grab_image()
            time.sleep(0.03)
            
    def step(self, command, initial=False):
        target_qpos = self.last_target_qpos.copy()
        target_ee_poses = []
        command_type = self.command_type
        if isinstance(command, dict) and 'command_type' in command:
            command_type = command['command_type']
            command = command['command']
            
        if command_type == 'joint_pos':
            target_qpos = command
            
        elif command_type == 'delta_eef_pose':
            delta_ee_pose, hand_command = command
            for follower_limb_name in delta_ee_pose.keys():
                ΔRt = delta_ee_pose[follower_limb_name]
                new_world2ee_Rt = self.init_world2ees[follower_limb_name] @ ΔRt
                #new_ee_pose = pt.pq_from_transform(new_world2ee_Rt)
                target_ee_poses.append(new_world2ee_Rt)

                inds = self.robot.ctrl_hand_joint_idx_mapping[follower_limb_name]
                if len(hand_command) and len(self.robot.ctrl_hand_joint_idx_mapping[follower_limb_name]) > 0:
                    hand_qpos = self.hand_solvers[follower_limb_name].solve(hand_command[follower_limb_name])
                    target_qpos[inds] = hand_qpos
            target_ee_poses = np.stack(target_ee_poses)
            arm_qpos = self.ik_solver.solve(target_ee_poses)
            target_qpos[self.ik_idx_mappings] = arm_qpos

        target_qpos = self.process_joint_pos(target_qpos, initial=initial)
        with self.pos_lock:
            self.last_target_qpos = self.vis_info['qpos'] = target_qpos
            if len(target_ee_poses):
                self.target_ee_poses = target_ee_poses

        if initial and self.command_type == 'delta_eef_pose':
            # update ik solvers with initial poses
            self.init_world2ees = self.ik_solver.compute_ee_poses(target_qpos)
        return target_qpos

    def process_joint_pos(self, input_qpos, initial=False):
        qpos = np.clip(input_qpos, self.joint_limits[:,0], self.joint_limits[:,1])
        if self.check_collision:
            new_qpos, self.vis_info['collision_list'] = self.collision_checker.get_collision_free_pose(qpos, verbose=True)
            if new_qpos is None:
                qpos = self.last_target_qpos
            else:
                qpos = new_qpos

        # # if it deviates too much from the current joint angles, we need to move slowly
        if not initial:
            qpos_diff = qpos - self.last_target_qpos
            qpos_diff = np.clip(qpos_diff, -self.max_joint_diff, self.max_joint_diff)
            qpos = self.last_target_qpos + qpos_diff

        return qpos

    def reset(self, initial_qpos=None):
        if initial_qpos is None:
            self.init_qpos = initial_qpos
            self.robot.init_qpos = self.init_qpos

        if self.command_type == 'delta_eef_pose':
            for eef_idx, limb_name in enumerate(self.robot.limb_names):
                self.hand_solvers[limb_name].reset()
                self.ik_solver.reset()
                self.ik_solver.set_qpos(self.init_qpos)
                self.init_world2ees = self.ik_solver.compute_ee_poses(self.init_qpos)
        self.last_target_qpos = self.vis_info['qpos'] = self.init_qpos
        return

    def close(self):
        self.shutdown = True
        # If any threads are running, join them here
        for t in self.threads:
            t.join()
        return

    def update_vis_info(self, vis_info=None):
        if vis_info is not None:
            vis_info['teleop'] = self.vis_info
        return vis_info


if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.utils.config_utils import change_working_directory
    change_working_directory()
    from paprle.follower import Robot

    robot_config, device_config, env_config = BaseConfig().parse()
    robot = Robot(robot_config)

    teleoperator = Teleoperator(robot, device_config, env_config, render_mode=True)
    
    
    # ROS2 transform_matrix subscriber를 사용하여 transform_matrix 토픽에서 데이터 받기
    rclpy.init()
    transform_node = Node('transform_matrix_subscriber_node')
    
    # 좌표계 변환 행렬: tracker 좌표계 → 로봇 좌표계 (4x4 행렬)
    # tracker x+ → robot y-
    # tracker y+ → robot x-
    # tracker z+ → robot z-
    
    coord_transform_T = np.array([
    [0, -1,  0,  0],
    [-1, 0,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
    ])
    
    # Transform matrix 데이터를 저장할 딕셔너리
    transform_data = {'left': None, 'right': None}
    transform_lock = Lock()
    
    # 토픽 이름 (파라미터로 받을 수도 있음)
    transform_topic_base = '/leader/vive_tracker'
    
    def transform_left_callback(msg):
        """Left transform_matrix 토픽 콜백 함수"""
        with transform_lock:
            # Float64MultiArray의 16개 요소를 4x4 행렬로 reshape
            T_tracker = np.array(msg.data).reshape(4, 4)
            # 좌표계 변환 적용: tracker → robot
            T_robot = T_tracker @ coord_transform_T
            transform_data['left'] = T_robot
    
    def transform_right_callback(msg):
        """Right transform_matrix 토픽 콜백 함수"""
        with transform_lock:
            # Float64MultiArray의 16개 요소를 4x4 행렬로 reshape
            T_tracker = np.array(msg.data).reshape(4, 4)
            # 좌표계 변환 적용: tracker → robot
            T_robot = T_tracker @ coord_transform_T
            transform_data['right'] = T_robot
    
    # transform_matrix 토픽 구독
    transform_left_subscription = transform_node.create_subscription(
        Float64MultiArray,
        f'{transform_topic_base}/eef_l_pose',
        transform_left_callback,
        10
    )
    
    transform_right_subscription = transform_node.create_subscription(
        Float64MultiArray,
        f'{transform_topic_base}/eef_r_pose',
        transform_right_callback,
        10
    )
    
    # 초기 pose를 기준으로 delta 계산을 위한 초기값 저장
    state = {
        'init_poses': {'left': None, 'right': None},
        'first_update': True
    }
    
    def get_delta_from_transform():
        """Transform matrix 데이터를 delta 변환 행렬로 변환"""
        with transform_lock:
            if transform_data['left'] is None or transform_data['right'] is None:
                return None, None
            
            if state['first_update']:
                # 첫 업데이트 시 초기 pose 저장
                state['init_poses']['left'] = transform_data['left'].copy()
                state['init_poses']['right'] = transform_data['right'].copy()
                state['first_update'] = False
                # 초기 delta는 identity
                return {
                    'left': np.eye(4),
                    'right': np.eye(4)
                }, None
            
            # delta 계산: init_pose^(-1) @ current_pose
            delta_ee_pose = {}
            for limb in ['left', 'right']:
                if state['init_poses'][limb] is not None and transform_data[limb] is not None:
                    init_inv = np.linalg.inv(state['init_poses'][limb])
                    delta = init_inv @ transform_data[limb]
                    delta_ee_pose[limb] = delta
            
            return delta_ee_pose, None
    
    # 메인 루프
    try:
        while rclpy.ok():
            rclpy.spin_once(transform_node, timeout_sec=0.1)
            
            delta_ee_pose, _ = get_delta_from_transform()
            if delta_ee_pose is not None:
                hand_command = {
                    'left': np.array([0.0]),   # gripper 값 (필요시 수정)
                    'right': np.array([0.0])   # gripper 값 (필요시 수정)
                }
                
                command = (delta_ee_pose, hand_command)
                joint_poses = teleoperator.step(command)
                # print(f"Joint poses: {joint_poses}")
            
            time.sleep(0.01)  # 100Hz
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        transform_node.destroy_node()
        rclpy.shutdown()
        
import warnings
warnings.filterwarnings('ignore')

import time
from threading import Lock
import numpy as np
from pytransform3d import transformations as pt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray  

from paprle.utils.config_utils import change_working_directory
change_working_directory()
from configs import BaseConfig
from paprle.follower import Robot
from paprle.teleoperator import Teleoperator
from paprle.envs import ENV_DICT

robot_config, device_config, env_config = BaseConfig().parse()
robot = Robot(robot_config)


class TeleoperatorWrapperNode(Node):
    def __init__(self):
        super().__init__('teleoperator_wrapper_node')
        
        self.declare_parameter('eef_l_tracker_topic', '/leader/vive_tracker/eef_l_pose')
        self.declare_parameter('eef_r_tracker_topic', '/leader/vive_tracker/eef_r_pose')
        self.declare_parameter('hz', 100)
        self.declare_parameter('vis_render', True)
        self.declare_parameter('sim_only', False)
        self.declare_parameter('mirror_mode', False)
        self.declare_parameter('max_position_jump', 0.25)  # 5cm
        self.declare_parameter('apply_arm_ratio', False)
        self.declare_parameter('human_arm_length', 550)
        
        self.eef_l_tracker_topic = self.get_parameter('eef_l_tracker_topic').get_parameter_value().string_value
        self.eef_r_tracker_topic = self.get_parameter('eef_r_tracker_topic').get_parameter_value().string_value
        self.hz = self.get_parameter('hz').get_parameter_value().integer_value
        self.vis_render = self.get_parameter('vis_render').get_parameter_value().bool_value
        self.sim_only = self.get_parameter('sim_only').get_parameter_value().bool_value
        self.mirror_mode = self.get_parameter('mirror_mode').get_parameter_value().bool_value
        self.max_position_jump = self.get_parameter('max_position_jump').get_parameter_value().double_value
        self.apply_arm_len_ratio = self.get_parameter('apply_arm_ratio').get_parameter_value().bool_value
        self.human_arm_length = self.get_parameter('human_arm_length').get_parameter_value().integer_value
        
        # 좌표계 변환 행렬: tracker 좌표계 → 로봇 좌표계 (4x4 행렬)
        # tracker x+ → robot y-
        # tracker y+ → robot x-
        # tracker z+ → robot z-
        self.coord_transform_T = np.array([
        [0, -1,  0,  0],
        [-1, 0,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
        ])
        
        # Transform matrix 데이터를 저장할 딕셔너리
        self.transform_lock = Lock()
        self.transform_data = {'left': None, 'right': None}
        self.state = {
            'init_poses': {'left': None, 'right': None},
            'last_poses': {'left': None, 'right': None},
            'first_update': True
        }
        self.robot_arm_length = 710
        self.arm_ratio = self.robot_arm_length / self.human_arm_length

        self.teleop_initialized = False
        self.tracker_send = False
        self.last_tracker_pose = None
        
        self.eef_l_tracker_subscription = self.create_subscription(
            Float64MultiArray,
            self.eef_l_tracker_topic,
            self.eef_l_tracker_callback,
            10
        )
        
        self.eef_r_tracker_subscription = self.create_subscription(
            Float64MultiArray,
            self.eef_r_tracker_topic,
            self.eef_r_tracker_callback,
            10
        )
        
        self.start_subscriber = self.create_subscription(Bool, 
            '/leader/teleop_start', 
            self.start_callback, 
            10
        )

        self.set(sim_only=self.sim_only)
        
        self.timer = self.create_timer(1/self.hz, self.timer_callback)
        self.timer_stop = False
    
    def set(self, sim_only=False):
        self.teleoperator = Teleoperator(robot, device_config, env_config, render_mode=self.vis_render)
        
        if not sim_only:
            self.env = ENV_DICT[env_config.name](robot, device_config, env_config, render_mode=False)
            init_env_qpos = self.env.reset()
            self.teleoperator.reset(init_env_qpos)
        
        self.get_logger().info('Teleoperator wrapper node initialized')

    def timer_callback(self):
        if self.tracker_send:
            delta_ee_pose, _ = self.get_delta_from_transform()
            if delta_ee_pose is not None:
                hand_command = {
                    'left': np.array([0.0]),
                    'right': np.array([0.0])
                }
                command = (delta_ee_pose, hand_command)
                
                if not self.teleop_initialized:
                    # self.get_logger().info('Initializing teleoperator...')
                    initial_qpos = self.teleoperator.step(command, initial=True)
                    if not self.sim_only:
                        self.env.initialize(initial_qpos)
                    self.teleop_initialized = True
                    # self.get_logger().info('Teleoperator initialized.')

                else:
                    qposes = self.teleoperator.step(command)
                
                    if not self.sim_only:
                        self.env.step(qposes)
                    # self.get_logger().info(f'Updated qposes')
                # qposes는 numpy 배열로 모든 관절의 각도(rad)를 포함합니다
                # 각 관절의 의미를 출력
                # joint_info = ', '.join([f'{name}: {angle:.4f}' for name, angle in zip(robot.joint_names, joint_poses)])
                # self.get_logger().info(f'Joint angles ({len(joint_poses)}): {joint_info}')
                
                # self.publish_joint_poses(qposes)

    # def publish_joint_poses(self, joint_poses):
    #     self.env.step(joint_poses)
        # arm_l_joints = joint_poses[0:7]
        # arm_r_joints = joint_poses[8:15]
        # lift_joint = joint_poses[16]

    def eef_l_tracker_callback(self, msg):
        with self.transform_lock:
            T_tracker = np.array(msg.data).reshape(4, 4)
            # 좌표계 변환 적용: tracker → robot
            T_robot = T_tracker @ self.coord_transform_T

            # 팔 길이만큼 배율 x, y, z축 배율
            if self.apply_arm_len_ratio:
                T_robot[:3, 3] = T_robot[:3, 3] * self.arm_ratio

            # 이전 위치와 비교하여 튀는 값 필터링
            if self.state['last_poses']['left'] is not None:
                last_pos = self.state['last_poses']['left'][:3, 3]
                current_pos = T_robot[:3, 3]
                distance = np.linalg.norm(current_pos - last_pos)
                
                if distance > self.max_position_jump:
                    # 5cm 이상 차이나면 이전 값을 유지하고 현재 값을 무시
                    self.get_logger().warn(
                        f'Left tracker jump detected: {distance*100:.2f}cm, ignoring current pose'
                    )
                    return  # 현재 값을 무시하고 이전 값 유지
            
            # 정상적인 값이면 업데이트
            self.transform_data['left'] = T_robot
            self.state['last_poses']['left'] = T_robot.copy()

    def eef_r_tracker_callback(self, msg):
        with self.transform_lock:
            T_tracker = np.array(msg.data).reshape(4, 4)
            # 좌표계 변환 적용: tracker → robot
            T_robot = T_tracker @ self.coord_transform_T

            # 팔 길이만큼 배율 x, y, z축 배율
            if self.apply_arm_len_ratio:
                T_robot[:3, 3] = T_robot[:3, 3] * self.arm_ratio

            # 이전 위치와 비교하여 튀는 값 필터링
            if self.state['last_poses']['right'] is not None:
                last_pos = self.state['last_poses']['right'][:3, 3]
                current_pos = T_robot[:3, 3]
                distance = np.linalg.norm(current_pos - last_pos)
                
                if distance > self.max_position_jump:
                    # 5cm 이상 차이나면 이전 값을 유지하고 현재 값을 무시
                    self.get_logger().warn(
                        f'Right tracker jump detected: {distance*100:.2f}cm, ignoring current pose'
                    )
                    return  # 현재 값을 무시하고 이전 값 유지
            
            # 정상적인 값이면 업데이트
            self.transform_data['right'] = T_robot
            self.state['last_poses']['right'] = T_robot.copy()
    
    def apply_mirror_mode_transform(self, T):
        """mirror_mode 모드에서 x축과 z축 회전 및 y, z축 변위를 반전시키는 변환 적용
        회전: y축 중심 180도 회전 변환 (x축, z축 반전)
        변위: x축 유지, y축과 z축 반전
        """
        # y축 중심 180도 회전 변환 행렬 (x축, z축 반전)
        flip_rotation = np.array([
            [-1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0,  0,  1]
        ])
        # 회전 부분만 변환: flip_rotation @ R @ flip_rotation.T
        R_flipped = flip_rotation[:3, :3] @ T[:3, :3] @ flip_rotation[:3, :3].T
        
        # 변위 부분: x축 유지, y축과 z축 반전
        t_flipped = T[:3, 3].copy()
        # t_flipped[0] = -t_flipped[0]  # x축 반전
        t_flipped[1] = -t_flipped[1]  # y축 반전
        # t_flipped[2] = -t_flipped[2]  # z축 반전

        # 변환된 행렬 조립
        T_flipped = np.eye(4)
        T_flipped[:3, :3] = R_flipped
        T_flipped[:3, 3] = t_flipped    
        
        return T_flipped
            
    def get_delta_from_transform(self):
        """Transform matrix 데이터를 delta 변환 행렬로 변환"""
        with self.transform_lock:
            if self.transform_data['left'] is None or self.transform_data['right'] is None:
                return None, None
            
            if self.state['first_update']:
                # 첫 업데이트 시 초기 pose 저장
                self.state['init_poses']['left'] = self.transform_data['left'].copy()
                self.state['init_poses']['right'] = self.transform_data['right'].copy()
                # last_poses도 초기화 (필터링을 위해)
                self.state['last_poses']['left'] = self.transform_data['left'].copy()
                self.state['last_poses']['right'] = self.transform_data['right'].copy()
                self.state['first_update'] = False
                # 초기 delta는 identity
                return {
                    'left': np.eye(4),
                    'right': np.eye(4)
                }, None
            
            # delta 계산: init_pose^(-1) @ current_pose
            delta_ee_pose = {}
            for limb in ['left', 'right']:
                if self.state['init_poses'][limb] is not None and self.transform_data[limb] is not None:
                    init_inv = np.linalg.inv(self.state['init_poses'][limb])
                    delta = init_inv @ self.transform_data[limb]
                    delta_ee_pose[limb] = delta
            
            # mirror_mode 모드일 때 좌우 매핑 반전 및 x축, z축 회전/변위 반전
            # 사용자의 오른손(eef_r) → 로봇의 왼손, 사용자의 왼손(eef_l) → 로봇의 오른손
            if self.mirror_mode:
                # 좌우 매핑 반전
                delta_ee_pose = {
                    'left': delta_ee_pose['right'],   # 사용자 오른손 → 로봇 왼손
                    'right': delta_ee_pose['left']     # 사용자 왼손 → 로봇 오른손
                }
                # x축, z축 회전 및 변위 반전 적용
                delta_ee_pose['left'] = self.apply_mirror_mode_transform(delta_ee_pose['left'])
                delta_ee_pose['right'] = self.apply_mirror_mode_transform(delta_ee_pose['right'])
            
            return delta_ee_pose, None
        
    def start_callback(self, msg):
        self.tracker_send = msg.data
        
        if self.tracker_send == False:
            self.timer.cancel()
            self.timer_stop = True

            if not self.sim_only:
                init_env_qpos = self.env.reset()
                self.teleoperator.reset(init_env_qpos)
            else:
                self.teleoperator.reset()
                
            self.transform_data = {'left': None, 'right': None}
            self.state = {
                'init_poses': {'left': None, 'right': None},
                'last_poses': {'left': None, 'right': None},
                'first_update': True
            }
                
            self.teleop_initialized = False

        else:
            if self.timer_stop:
                self.timer.reset()
                self.timer_stop = False

def main(args=None):
    rclpy.init(args=args)
    node = TeleoperatorWrapperNode()
    
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f'Error: {e}')
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down node')
    finally:
        node.destroy_node()
        rclpy.shutdown()
    

if __name__ == '__main__':
    main()
    
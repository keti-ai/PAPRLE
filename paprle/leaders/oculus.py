from oculus_reader.reader import OculusReader
from threading import Thread, Lock
import numpy as np
import time
from pytransform3d import transformations as pt
def apply_transfer(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    # xyz can be 3dim or 4dim (homogeneous) or can be a rotation matrix
    if len(xyz) == 3:
        xyz = np.append(xyz, 1)
    return np.matmul(mat, xyz)[:3]
from pytransform3d import rotations, coordinates
from pytransform3d import transformations as pt
class LPFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False
class LPRotationFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.is_init = False

        self.y = None

    def next(self, x: np.ndarray):
        assert x.shape == (4,)

        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()

        self.y = rotations.quaternion_slerp(self.y, x, self.alpha, shortest_path=True)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False
# This code is partially from https://github.com/ToruOwO/hato/blob/main/agents/quest_agent.py
class Oculus:
    def __init__(self, robot, leader_config, env_config, render_mode='human', verbose=False, *args, **kwargs):
        self.robot = robot
        self.leader_config = leader_config
        self.oculus2paprle = np.array([[0, -1, 0, 0],
                                     [-1, 0, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 0, 0, 1]])
        self.oculus_reader = OculusReader()#(ip_address='10.1.10.225')

        self.r_ee_Rt = np.eye(4)
        self.l_ee_Rt = np.eye(4)
        self.past_rp, self.past_rR = None, None
        self.past_lp, self.past_lR = None, None

        self.is_ready = False
        self.require_end = False
        self.shutdown = False
        self.leader_viz_info = {'color': 'blue',  'log': "Oculus is ready!"}

        self.left_pos_filter = LPFilter(0.4)
        self.right_pos_filter = LPFilter(0.4)
        self.left_rot_filter = LPRotationFilter(0.4)
        self.right_rot_filter = LPRotationFilter(0.4)

        self.render_mode = render_mode
        if self.render_mode:
            self.render_thread = Thread(target=self.render)
            self.render_thread.start()
        return

    def render(self):
        from paprle.visualizer.viser_vis import ViserViz
        self.viz = ViserViz(None, server_label='oculus')
        print("Oculus rendering started. You can check each controller pose in the above link")

        batched_wxyz = np.tile(np.array([1.0,0.0,0.0,0.0]), [2,1])
        batched_positions = np.tile(np.array([0.0,0.0,0.0]), [2,1])
        wrist_handle = self.viz.server.add_batched_axes(name='wrist_axes', batched_wxyzs=batched_wxyz[:2], batched_positions=batched_positions[:2], axes_length=0.05, axes_radius=0.005)

        self.right_wrist_label = self.viz.server.scene.add_label(name='right_wrist_label', text='Right Wrist',
                                                             position=np.array([0, 0, 0]))
        self.left_wrist_label = self.viz.server.scene.add_label(name='left_wrist_label', text='Left Wrist',
                                                            position=np.array([0, 0, 0]))

        while not self.shutdown:
            start_time = time.time()
            transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
            current_right_pose = transformations['r'] @ self.oculus2paprle
            U, _, Vt = np.linalg.svd(current_right_pose[:3, :3])
            current_right_pose[:3,:3] = np.dot(U, Vt)
            right_pq = pt.pq_from_transform(current_right_pose, strict_check=False)
            current_left_pose = transformations['l'] @ self.oculus2paprle
            U, _, Vt = np.linalg.svd(current_left_pose[:3, :3])
            current_left_pose[:3,:3] = np.dot(U, Vt)
            left_pq = pt.pq_from_transform(current_left_pose, strict_check=False)
            batched_wxyz[0] = right_pq[3:]
            batched_positions[0] = right_pq[:3]
            batched_wxyz[1] = left_pq[3:]
            batched_positions[1] = left_pq[:3]
            wrist_handle.batched_wxyzs = batched_wxyz[:2]
            wrist_handle.batched_positions = batched_positions[:2]

            self.right_wrist_label.position = right_pq[:3] + np.array([0,0,0.05])
            self.left_wrist_label.position = left_pq[:3] + np.array([0,0,0.05])

            loop_time = time.time() - start_time
            time.sleep(max(0.00, 0.03 - loop_time))

        return

    def reset(self, ):
        return

    def launch_init(self, init_env_qpos):
        self.init_thread = Thread(target=self.initialize)
        self.init_thread.start()
        return

    def initialize(self):
        iteration, initialize_progress = 0.0, 0.0
        dt, threshold_time = 0.03, 3
        while not initialize_progress >= threshold_time:
            if self.shutdown: return
            transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
            if 'r' not in transformations: continue
            r_gripper_closed = buttons['rightGrip'][0] > 0.5 if 'right' in self.leader_config.limb_mapping else True
            l_gripper_closed = buttons['leftGrip'][0] > 0.5 if 'left' in self.leader_config.limb_mapping else True
            if r_gripper_closed and l_gripper_closed:
                initialize_progress += dt
            else:
                initialize_progress = 0.0
            print(f"Waiting for oculus initialize, Press gripper .... {initialize_progress}", end="\r")
            self.leader_viz_info['color'] = 'blue'
            self.leader_viz_info['log'] = f"Close the grippers to initialize the controller.... {initialize_progress:.2f}/{threshold_time}"

            time.sleep(dt)


        self.right_pos_filter.reset()
        self.right_rot_filter.reset()
        self.r_ee_Rt = np.eye(4)
        self.l_ee_Rt = np.eye(4)
        if 'r' in transformations:
            rRt = transformations['r'] @ self.oculus2paprle
            self.past_rp, self.past_rR = rRt[:3, 3], rRt[:3, :3]
        if 'l' in transformations:
            lRt = transformations['l'] @ self.oculus2paprle
            self.past_lp, self.past_lR = lRt[:3, 3], lRt[:3, :3]
        self.is_ready = True
        self.require_end = False
        self.leader_viz_info['color'] = 'green'
        self.leader_viz_info['log'] = 'Oculus initialized successfully!'
        return


    def close_init(self):
        self.init_thread.join()
        return

    def get_status(self):
        transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
        local_poses, hand_poses = {}, {}

        self.leader_viz_info['color'] = 'green'
        self.leader_viz_info['log'] = 'Oculus is running...'

        r_end_detect = buttons['A'] if 'right' in self.leader_config.limb_mapping else True
        l_end_detect = buttons['X'] if 'left' in self.leader_config.limb_mapping else True
        if r_end_detect and l_end_detect:
            self.leader_viz_info['color'] = 'red'
            self.leader_viz_info['log'] = "End signal detected! Resetting the leader and follower"
            self.require_end = True
            self.is_ready = False
            for leader_limb_name, follower_limb_name in self.leader_config.limb_mapping.items():
                local_poses[follower_limb_name] = np.eye(4)
                hand_poses[follower_limb_name] = 0.0
            return (local_poses, hand_poses)

        if 'r' in transformations:
            current_pose = transformations['r'] @ self.oculus2paprle
            U, _, Vt = np.linalg.svd(current_pose[:3, :3])
            current_pose[:3,:3] = np.dot(U, Vt)
            follower_limb_name = self.leader_config.limb_mapping['right']
            if buttons['rightGrip'][0] > 0.5:
                Rt = current_pose
                new_pos = self.past_rR.T @ Rt[:3,3] - self.past_rR.T @ self.past_rp
                new_rot = self.past_rR.T @ Rt[:3, :3]
                new_Rt = pt.transform_from(R=new_rot, p=new_pos)
                self.r_ee_Rt = self.r_ee_Rt @ new_Rt
                local_poses[follower_limb_name] = self.r_ee_Rt
            else:
                local_poses[follower_limb_name] = self.r_ee_Rt
            self.past_rp, self.past_rR = current_pose[:3, 3], current_pose[:3, :3]
            hand_poses[follower_limb_name] = buttons['rightTrig'][0]
        if 'l' in transformations and 'left' in self.leader_config.limb_mapping:
            current_pose = transformations['l'] @ self.oculus2paprle
            follower_limb_name = self.leader_config.limb_mapping['left']
            if buttons['leftGrip'][0] > 0.5:
                Rt = current_pose
                new_pos = self.past_lR.T @ Rt[:3,3] - self.past_lR.T @ self.past_lp
                new_rot = self.past_lR.T @ Rt[:3, :3]
                U, _, Vt = np.linalg.svd(new_rot)
                new_rot = np.dot(U, Vt)
                new_Rt = pt.transform_from(R=new_rot, p=new_pos)
                self.l_ee_Rt = self.l_ee_Rt @ new_Rt
                local_poses[follower_limb_name] = self.l_ee_Rt
            else:
                local_poses[follower_limb_name] = self.l_ee_Rt
            self.past_lp, self.past_lR = current_pose[:3, 3], current_pose[:3, :3]
            hand_poses[follower_limb_name] = buttons['leftTrig'][0]
        return (local_poses, hand_poses)


    def update_vis_info(self, env_vis_info):
        if env_vis_info is not None:
            env_vis_info['leader'] = self.leader_viz_info
        return env_vis_info

    def close(self):
        self.shutdown = True
        if self.init_thread is not None:
            self.init_thread.join()
        if self.render_mode:
            self.render_thread.join()
        return

from paprle.utils.config_utils import change_working_directory
change_working_directory()

from tqdm import tqdm
import json
import h5py
import glob
import argparse
import os
import pickle
import numpy as np
import cv2
import pytransform3d.transformations as pt
from omegaconf import OmegaConf
from paprle.utils.data_process_utils import setup_fk_model, get_ee_poses
from multiprocessing import Pool
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/media/obin/02E6-928E/teleop_data/raw_data/fruits/success/')
parser.add_argument('--save_command', action='store_true', help='Save command data', default=True)
parser.add_argument('--ep-start-idx', type=int, default=0, help='The index of the first episode')
parser.add_argument('--no-depth', action='store_true', help='Do not save depth data', default=False)
parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel workers') # Be careful with memory usage
args = parser.parse_args()

save_dir = os.path.join(os.path.dirname(args.data_dir), '../processed')
os.makedirs(save_dir, exist_ok=True)
print("Save dir: ", save_dir)

episode_list = sorted(glob.glob(glob.escape(args.data_dir) + '*'))
print(f"Episode nums: {len(episode_list)}")

fk_models = None
ep_start_idx = args.ep_start_idx
episode_id_list = list(range(ep_start_idx, len(episode_list)+ep_start_idx))

def process_episode(idx):
    if idx < 300: return
    episode_dir = episode_list[idx]
    original_step_list = sorted(glob.glob(glob.escape(episode_dir) + '/data_*.pkl'))
    episode_info_file = os.path.join(episode_dir, 'episode_info.pkl')
    if not os.path.exists(episode_info_file):
        print(f"Episode: {episode_dir} is invalid")
        return
    original_episode_info = pickle.load(open(episode_info_file, 'rb'))
    trim_info = original_episode_info['trim_info']
    start_idx, end_idx = trim_info[0], trim_info[1]

    step_list = original_step_list[start_idx:end_idx + 1]
    T = len(step_list)
    print("Episode length: ", T, "Start idx: ", start_idx, "End idx: ", end_idx, 'Original length: ', len(step_list))
    episode_info = pickle.load(open(os.path.join(episode_dir, 'episode_info.pkl'), 'rb'))
    arm_dof, hand_dof = episode_info['robot_config'].robot_cfg.arm_dof, episode_info['robot_config'].robot_cfg.hand_dof
    command_type = episode_info['device_config']['output_type']
    robot_names = list(episode_info['robot_config']['robot_cfg']['limb_joint_names'].keys())

    fk_models, idx_mappings = setup_fk_model(episode_info)

    sample_data = pickle.load(open(step_list[0], 'rb'))
    if episode_info['env_config']['name'] == 'mujoco':
        qpos, qvel = None, None
    else:
        qpos = sample_data['obs']['qpos']
        qvel = sample_data['obs']['qvel']
        qeff = sample_data['obs']['qeff']

    target_qpos = sample_data['target_qpos']
    target_ee_poses = get_ee_poses(fk_models, idx_mappings, target_qpos, arm_dof, hand_dof, robot_names)

    episode_file = os.path.join(save_dir, f'episode_{ep_start_idx + idx:03d}_{episode_info["robot_config"].robot_cfg.name}_{episode_info["device_config"].name}.hdf5')
    with h5py.File(episode_file, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['episode_name'] = episode_info['episode_name']
        root.attrs['robot_name'] = str(episode_info['robot_config'].robot_cfg.name)
        root.attrs['device_name'] = str(episode_info['device_config'].name)
        root.attrs['robot_config'] = json.dumps(OmegaConf.to_container(episode_info['robot_config']))
        root.attrs['device_config'] = json.dumps(OmegaConf.to_container(episode_info['device_config']))
        root.attrs['env_config'] = json.dumps(OmegaConf.to_container(episode_info['env_config']))
        # ep info
        root.attrs['collector'] = episode_info['collector']
        root.attrs['task'] = episode_info['task']
        obs = root.create_group('obs')

        have_camera = ('camera' in sample_data['obs']) and len(sample_data['obs']['camera']) > 0
        if have_camera:
            cam_names = list(sample_data['obs']['camera'].keys())
            im_h, im_w = 240, 424
            _ = obs.create_dataset('rgb', (len(cam_names), T, im_h, im_w, 3), dtype='uint8', chunks=(len(cam_names), 1, im_h, im_w, 3), compression='gzip', compression_opts=4)
            if not args.no_depth:
                _ = obs.create_dataset('depth', (len(cam_names), T, im_h, im_w), dtype='uint16', chunks=(len(cam_names), 1, im_h, im_w), compression='gzip', compression_opts=4)
            depth_unit_info = {}
            for cam_idx, cam_name in enumerate(cam_names):
                depth_unit_info[cam_name] = sample_data['obs']['camera'][cam_name]['depth_units']
            root.attrs['camera_info'] = json.dumps(depth_unit_info)

        if qpos is not None:
            qpos = obs.create_dataset('qpos', (T, len(qpos)))
            qvel = obs.create_dataset('qvel', (T, len(qvel)))
            qeff = obs.create_dataset('qeff', (T, len(qeff)))
        timestamp = obs.create_dataset('timestamp', (T, 1), dtype='float64')

        action = root.create_group('action')
        qpos_action = action.create_dataset('qpos_action', (T, len(target_qpos)))
        eepose_action = action.create_dataset('eepose_action', (T, *target_ee_poses.shape))

        if args.save_command:
            command = root.create_group('command')
            if episode_info['device_config']['output_type'] == 'joint_pos':
                command_data = command.create_dataset('joint_pos', (T, len(target_qpos)))
                #command_hand_data = command.create_dataset('hand_pose', (T, *hand_pose_size.shape))
            else:
                limb_pose_dict, hand_pose_dict = sample_data['command']
                command_data = command.create_dataset('delta_ee_pos', (T, *target_ee_poses.shape))
                hand_pose_size = np.stack([v for v in hand_pose_dict.values()])
                command_hand_data = command.create_dataset('hand_pose', (T, *hand_pose_size.shape))


        for idx, step_file in enumerate(tqdm(step_list, desc=f"Episode {idx} processing...")):
            if 'episode_info' in step_file: continue
            data = pickle.load(open(step_file, 'rb'))
            if have_camera:
                rgbs, depths = [], []
                for cam_name in cam_names:
                    rgb = cv2.resize(data['obs']['camera'][cam_name]['color'], (im_w, im_h))
                    rgbs.append(rgb)
                    if not args.no_depth:
                        depth = cv2.resize(data['obs']['camera'][cam_name]['depth'], (im_w, im_h), interpolation=cv2.INTER_NEAREST)
                        depths.append(depth)
                obs['rgb'][:, idx] = rgbs
                if not args.no_depth:
                    obs['depth'][:, idx] = depths
            timestamp[idx] = data['timestamp']
            # joint states
            if qpos is not None:
                qpos[idx] = data['obs']['qpos']
                qvel[idx] = data['obs']['qvel']
                qeff[idx] = data['obs']['qeff']
            # actions
            qpos_action[idx] = data['target_qpos']
            eepose_action[idx] = get_ee_poses(fk_models, idx_mappings, data['target_qpos'], arm_dof, hand_dof, robot_names)

            if args.save_command:
                if command_type == 'joint_pos':
                    command_data[idx] = data['command']
                else:
                    limb_pose_dict, hand_pose_dict = data['command']
                    try:
                        command_data[idx] = np.stack([pt.pq_from_transform(data['command'][0][limb_name], strict_check=False) for limb_name in limb_pose_dict.keys()])
                        command_hand_data[idx] = np.stack([data['command'][1][hand_name] for hand_name in hand_pose_dict.keys()])
                    except:
                        print(f"Error processing command data at step {idx} in episode {episode_dir}. Skipping this step.")
                        continue


if __name__ == '__main__':
    with Pool(processes=args.num_workers) as pool:
        pool.map(process_episode, episode_id_list)
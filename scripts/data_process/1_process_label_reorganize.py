import glob
import os
import json
from tqdm import tqdm
import pickle
from paprle.utils.config_utils import change_working_directory
change_working_directory()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/media/obin/02E6-928E/teleop_bag_data/demo_data_0911',
                    help='Path to the directory containing episode data')
args = parser.parse_args()

TASK_DICT = {}

def make_new_task_dir(task_name, task_dir):
    os.makedirs(task_dir, exist_ok=True)
    TASK_DICT[task_name] = {'name': task_name, 'task_dir': task_dir}
    with open(os.path.join(task_dir, 'TASK_INFO.json'), 'w') as f:
        json.dump(TASK_DICT[task_name], f, indent=4)
    os.makedirs(os.path.join(task_dir, 'success'), exist_ok=True)
    os.makedirs(os.path.join(task_dir, 'failure'), exist_ok=True)
    os.makedirs(os.path.join(task_dir, 'invalid'), exist_ok=True)
    return

def process_dir(data_dir):
    ep_list = sorted(glob.glob(os.path.join(data_dir, '*')))
    for ep_dir in ep_list:
        if os.path.exists(os.path.join(ep_dir, 'TASK_INFO.json')):
            with open(os.path.join(ep_dir, 'TASK_INFO.json'), 'r') as f:
                task_info = json.load(f)
            TASK_DICT[task_info['name']] = task_info
           

    episode_name_to_dir = {}
    episode_name_to_data_list = {}
    for ep_dir in tqdm(ep_list, desc="Loading episodes"):
        episode_info_file = os.path.join(ep_dir, 'episode_info.pkl')
        if not os.path.exists(episode_info_file):
            print(f"Warning: {episode_info_file} does not exist. Skipping this episode.")
            continue
        data_list = glob.glob(os.path.join(ep_dir, 'data*.pkl'))
        if len(data_list) > 0:
            ep_name = os.path.basename(ep_dir)
            episode_name_to_dir[ep_name] = ep_dir
            episode_name_to_data_list[ep_name] = data_list

            try:
                with open(episode_info_file, 'rb') as f:
                    episode_info = pickle.load(f)
            except:
                print(f"Error loading episode info from {episode_info_file}. Skipping this episode.")
                continue
            if 'task' not in episode_info: task_name = None
            else: task_name = episode_info['task']
            if task_name is None:
                print(f"Warning: Episode {ep_name} does not have a task name. Skipping.")
                continue
            if task_name not in TASK_DICT:
                task_dir = os.path.join(data_dir, task_name)
                make_new_task_dir(task_name, task_dir)
            
            if(TASK_DICT[task_name]['task_dir'] != os.path.join(data_dir, task_name)):
                TASK_DICT[task_name]['task_dir'] = os.path.join(data_dir, task_name)

           
            if episode_info['success'] == 'invalid':
                target_dir = os.path.join(TASK_DICT[task_name]['task_dir'], 'invalid')
            elif episode_info['success']:
                target_dir = os.path.join(TASK_DICT[task_name]['task_dir'], 'success')
            elif not episode_info['success']:
                target_dir = os.path.join(TASK_DICT[task_name]['task_dir'], 'failure')
            else:
                print(f"Warning: Episode {ep_name} has an unknown success status. Skipping.")
                continue
            dir_name = os.path.basename(ep_dir)
            new_dir = os.path.join(data_dir, target_dir, dir_name)
            
            os.rename(ep_dir, new_dir)


if __name__ == '__main__':
    process_dir(args.data_dir)

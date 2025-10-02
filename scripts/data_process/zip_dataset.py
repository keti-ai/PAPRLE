import os
import glob
import zipfile
from zipfile import ZipFile
from tqdm import tqdm

from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/media/obin/02E6-928E/teleop_data/',
                    help='Directory containing raw episode data files (.pkl)')
parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for zipping')
parser.add_argument('--out_dir', type=str, default='/media/obin/Processed/teleop_data_zip/',
                    help='Output directory for zipped files. If not set, zips are saved in the same directory as raw files.')
parser.add_argument('--ignore_existing', action='store_true', default=False)
args = parser.parse_args()
ep_list = sorted(glob.glob(args.data_dir + '/*'))

def process_all_episodes(data_dir, out_dir, num_workers=4):
    episode_dirs = []
    recursive_search_ep_dirs(data_dir, episode_dirs=episode_dirs, out_dir=out_dir)
    print(f"Found {len(episode_dirs)} episode directories to process.")
    with Pool(num_workers) as p:
        p.map(zip_dataset, episode_dirs)

def recursive_search_ep_dirs(data_dir, episode_dirs=[], out_dir=None):
    folders = glob.glob(os.path.join(data_dir, '*'))
    for folder in folders:
        if os.path.isdir(folder):
            episode_info_file = os.path.join(folder, 'episode_info.pkl')
            if os.path.exists(episode_info_file):
                if out_dir is not None:
                    zip_file_dst = os.path.join(out_dir, os.path.basename(folder) + '.zip')
                    episode_dirs.append([folder, zip_file_dst])
                else:
                    episode_dirs.append([folder, None])
            else:
                relative_path = os.path.relpath(folder, data_dir)
                target_dir = os.path.join(out_dir, relative_path)
                os.makedirs(target_dir, exist_ok=True)
                recursive_search_ep_dirs(folder, episode_dirs=episode_dirs, out_dir=target_dir)
    return

def zip_dataset(dir_info):
    raw_ep_dir, out_zip_file = dir_info
    episode_info_file = os.path.join(raw_ep_dir, 'episode_info.pkl')
    if not os.path.exists(episode_info_file):
        print(f"Skipping {raw_ep_dir}, no episode_info.pkl found.")
        return
    datafile_list = sorted(glob.glob(os.path.join(raw_ep_dir, '*.pkl')))
    if out_zip_file is not None:
        zip_file_path = out_zip_file
    else:
        zip_file_path = os.path.join(raw_ep_dir + '.zip')
    if os.path.exists(zip_file_path) and args.ignore_existing:
        print(f"Skipping {raw_ep_dir}, zip file already exists.")
        return
    zip_file_path = zip_file_path.replace('.zip', '_tmp.zip')
    with ZipFile(zip_file_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for file in tqdm(datafile_list, desc=f"Zipping {os.path.basename(raw_ep_dir)}"):
            zipf.write(file, os.path.basename(file))
    new_zip_file_path = zip_file_path.replace('_tmp.zip', '.zip')
    os.rename(zip_file_path, new_zip_file_path)
    return

if __name__ == "__main__":
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
    process_all_episodes(args.data_dir, args.out_dir, num_workers=args.num_workers)

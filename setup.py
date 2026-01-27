import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'paprle'


def get_data_files_recursive(directory):
    """디렉토리와 모든 하위 디렉토리의 파일을 재귀적으로 수집합니다."""
    data_files = []
    if not os.path.exists(directory):
        return data_files
    for root, dirs, files in os.walk(directory):
        # __pycache__ 디렉토리는 제외
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            file_path = os.path.join(root, file)
            # 상대 경로를 유지하면서 설치 경로 생성
            install_dir = os.path.join('share', package_name, root)
            data_files.append((install_dir, [file_path]))
    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, f'launch'), glob(f'launch/*.py')),
        # configs, models, scripts 디렉토리의 모든 파일을 재귀적으로 포함
        *get_data_files_recursive('configs'),
        *get_data_files_recursive('models'),
        *get_data_files_recursive('scripts'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='js',
    maintainer_email='moonjongsul@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'paprle_teleop = paprle.teleoperator_wrapper_node:main',
        ],
    },
)

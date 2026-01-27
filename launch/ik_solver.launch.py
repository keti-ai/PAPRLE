#!/usr/bin/env python3
#
# Copyright 2026 Korea Electronics Technology Institute (KETI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Jongsul Moon

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declared_arguments =[
        # PAPRLE arguments
        DeclareLaunchArgument('eef_l_tracker_topic', default_value='/leader/vive_tracker/eef_l_pose',
                              description='The topic to publish the left eef tracker state'),
        DeclareLaunchArgument('eef_r_tracker_topic', default_value='/leader/vive_tracker/eef_r_pose',
                              description='The topic to publish the right eef tracker state'),
        DeclareLaunchArgument('hz', default_value='100',
                              description='The frequency of the teleoperator'),
        DeclareLaunchArgument('vis_render', default_value='True',
                              description='Whether to render the visualizer'),
    ]

    eef_l_tracker_topic = LaunchConfiguration('eef_l_tracker_topic')
    eef_r_tracker_topic = LaunchConfiguration('eef_r_tracker_topic')
    hz = LaunchConfiguration('hz')
    vis_render = LaunchConfiguration('vis_render')
    
    paprle_teleop_node = Node(
        package='paprle',
        executable='paprle_teleop',
        name='paprle_teleop_node',
        parameters=[{'eef_l_tracker_topic': eef_l_tracker_topic, 
                     'eef_r_tracker_topic': eef_r_tracker_topic, 
                     'hz': hz,
                     'vis_render': vis_render,
                     }],
    )

    return LaunchDescription(
        declared_arguments + [
            paprle_teleop_node,
        ]
    )
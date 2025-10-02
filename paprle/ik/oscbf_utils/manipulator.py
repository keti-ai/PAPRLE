import io
import os
import numpy as np
import six
from lxml import etree as ET
from typing import Tuple, Optional
import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from oscbf.utils import urdfpy
from oscbf.core.manipulator import Manipulator, parse_urdf, tuplify
from oscbf.utils.urdf_parser import  genesis_to_mine, merge_fixed_links, _order_links
from oscbf.core.papras_collision_model import papras_collision_data


def gs_parse_urdf(filename, asset_path, merge_fixed=True, links_to_keep=[], fixed_base=True, joint_names=None):
    robot = CustomURDF.load(filename, asset_path, joint_names=joint_names)

    # merge links connected by fixed joints
    if merge_fixed:
        robot = merge_fixed_links(robot, links_to_keep)

    link_name_to_idx = dict()
    for idx, link in enumerate(robot.links):
        link_name_to_idx[link.name] = idx

    # Note that each link corresponds to one joint
    n_links = len(robot.links)
    assert n_links == len(robot.joints) + 1
    l_infos = [dict() for _ in range(n_links)]
    j_infos = [dict() for _ in range(n_links)]

    for i in range(n_links):
        link = robot.links[i]
        l_info = l_infos[i]
        l_info["name"] = link.name

        if link.inertial is None:
            l_info["inertial_pos"] = np.zeros(3)
            l_info["inertial_rot"] = np.eye(3)
            l_info["inertial_i"] = None
            l_info["inertial_mass"] = None

        else:
            l_info["inertial_pos"] = link.inertial.origin[:3, 3]
            l_info["inertial_rot"] = link.inertial.origin[:3, :3]
            l_info["inertial_i"] = link.inertial.inertia
            l_info["inertial_mass"] = link.inertial.mass

    #########################  non-base joints and links #########################
    for joint in robot.joints:
        idx = link_name_to_idx[joint.child]
        l_info = l_infos[idx]
        j_info = j_infos[idx]

        j_info["name"] = joint.name
        j_info["pos"] = np.zeros(3)
        j_info["rot"] = np.eye(3)

        j_info["axis"] = joint.axis

        l_info["parent_idx"] = link_name_to_idx[joint.parent]
        l_info["pos"] = joint.origin[:3, 3]
        l_info["rot"] = joint.origin[:3, :3]

        if joint.joint_type == "fixed":
            j_info["dofs_motion_ang"] = np.zeros((0, 3))
            j_info["dofs_motion_vel"] = np.zeros((0, 3))
            j_info["dofs_limit"] = np.zeros((0, 2))
            j_info["dofs_stiffness"] = np.zeros((0))

            j_info["type"] = "fixed"
            j_info["n_qs"] = 0
            j_info["n_dofs"] = 0
            j_info["init_qpos"] = np.zeros(0)

        elif joint.joint_type == "revolute":
            j_info["dofs_motion_ang"] = np.array([joint.axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            j_info["dofs_limit"] = np.array(
                [
                    [
                        joint.limit.lower if joint.limit.lower is not None else -np.inf,
                        joint.limit.upper if joint.limit.upper is not None else np.inf,
                    ]
                ]
            )
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = "revolute"
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)

        elif joint.joint_type == "continuous":
            j_info["dofs_motion_ang"] = np.array([joint.axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            j_info["dofs_limit"] = np.array([[-np.inf, np.inf]])
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = "revolute"
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)

        elif joint.joint_type == "prismatic":
            j_info["dofs_motion_ang"] = np.zeros((1, 3))
            j_info["dofs_motion_vel"] = np.array([joint.axis])
            j_info["dofs_limit"] = np.array(
                [
                    [
                        joint.limit.lower if joint.limit.lower is not None else -np.inf,
                        joint.limit.upper if joint.limit.upper is not None else np.inf,
                    ]
                ]
            )
            j_info["dofs_stiffness"] = np.array([0.0])

            j_info["type"] = "prismatic"
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)

        elif joint.joint_type == "floating":
            raise NotImplementedError("Floating joint not supported yet")
        else:
            raise Exception(f"Unsupported URDF joint type: {joint.joint_type}")

        # NOTE: This is the stand-in default value from genesis
        j_info["dofs_force_range"] = np.tile([[-100, 100]], [j_info["n_dofs"], 1])

        # NEW: Velocity limit handling
        # Matching the way Genesis handled it (though, I'm not a fan)
        j_info["dofs_velocity_range"] = np.tile([[-100, 100]], [j_info["n_dofs"], 1])

        if joint.limit is not None:
            if joint.limit.effort is not None:
                j_info["dofs_force_range"] = (
                    j_info["dofs_force_range"]
                    / np.abs(j_info["dofs_force_range"])
                    * joint.limit.effort
                )
            # NEW: Velocity limit handling
            if joint.limit.velocity is not None:
                j_info["dofs_velocity_range"] = (
                    j_info["dofs_velocity_range"]
                    / np.abs(j_info["dofs_velocity_range"])
                    * joint.limit.velocity
                )

    l_infos, j_infos, _ = _order_links(l_infos, j_infos)
    ######################### first joint and base link #########################
    j_info = j_infos[0]
    l_info = l_infos[0]

    j_info["pos"] = np.zeros(3)
    j_info["rot"] = np.eye(3)
    j_info["name"] = f'joint_{l_info["name"]}'

    # Genesis parses the base link as having an associated joint even though this is fixed
    j_info["axis"] = np.zeros(3)

    l_info["pos"] = np.zeros(3)
    l_info["rot"] = np.eye(3)

    if not fixed_base:
        raise NotImplementedError("Base link must be fixed for now")
    else:
        j_info["dofs_motion_ang"] = np.zeros((0, 3))
        j_info["dofs_motion_vel"] = np.zeros((0, 3))
        j_info["dofs_limit"] = np.zeros((0, 2))
        j_info["dofs_stiffness"] = np.zeros((0))

        j_info["type"] = "fixed"
        j_info["n_qs"] = 0
        j_info["n_dofs"] = 0
        j_info["init_qpos"] = np.zeros(0)

    return l_infos, j_infos

class CustomURDF(urdfpy.URDF):
    @staticmethod
    def load(file_obj, asset_path, joint_names=None):
        if isinstance(file_obj, six.string_types):
            if os.path.isfile(file_obj):
                parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
                with open(file_obj, "r") as f:
                    file_str = f.read()
                # version 0.0 cannot be parsed by lxml
                file_str = file_str.replace('<?xml version="0.0" ?>', "")
                rel_path = os.path.relpath(asset_path, start=os.path.dirname(file_obj))
                file_str = file_str.replace('filename="', 'filename="{}/'.format(rel_path))
                file_str = file_str.encode()
                tree = ET.parse(io.BytesIO(file_str), parser=parser)
                path, _ = os.path.split(file_obj)
            else:
                raise ValueError("{} is not a file".format(file_obj))
        else:
            parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
            tree = ET.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)
        node = tree.getroot()
        if joint_names is not None:
            for child in node.getchildren():
                if child.tag == "joint" and child.attrib['name'] not in joint_names:
                    child.attrib['type'] = 'fixed'
        return CustomURDF._from_xml(node, path)

@jax.tree_util.register_static
class CustomManipulator(Manipulator):

    def __init__(
        self,
        num_joints: int,
        joint_types: tuple,
        joint_lower_limits: tuple,
        joint_upper_limits: tuple,
        joint_max_forces: tuple,
        joint_max_velocities: tuple,
        joint_axes: tuple,
        joint_parent_frame_positions: tuple,
        joint_parent_frame_rotations: tuple,
        link_masses: tuple,
        link_local_inertias: tuple,
        link_local_inertia_positions: tuple,
        link_local_inertia_rotations: tuple,
        base_pos: tuple,
        base_orn: tuple,
        ee_offset: tuple,
        collision_positions: tuple,
        collision_radii: tuple,
        joint_names=None,
    ):
        super().__init__(
            num_joints=num_joints,
            joint_types=joint_types,
            joint_lower_limits=joint_lower_limits,
            joint_upper_limits=joint_upper_limits,
            joint_max_forces=joint_max_forces,
            joint_max_velocities=joint_max_velocities,
            joint_axes=joint_axes,
            joint_parent_frame_positions=joint_parent_frame_positions,
            joint_parent_frame_rotations=joint_parent_frame_rotations,
            link_masses=link_masses,
            link_local_inertias=link_local_inertias,
            link_local_inertia_positions=link_local_inertia_positions,
            link_local_inertia_rotations=link_local_inertia_rotations,
            base_pos=base_pos,
            base_orn=base_orn,
            ee_offset=ee_offset,
            collision_positions=collision_positions,
            collision_radii=collision_radii,
        )
        self.joint_names = joint_names

    @classmethod
    def from_urdf(cls, urdf_filename, asset_path, ee_offset, collision_data=None, joint_names=None):
        l_infos, j_infos = gs_parse_urdf(urdf_filename, asset_path=asset_path, merge_fixed=True, joint_names=joint_names)
        data = genesis_to_mine(l_infos, j_infos)
        data = {k: tuplify(v) for k, v in data.items()}

        assert isinstance(collision_data, dict) or collision_data is None
        if isinstance(collision_data, dict):
            collision_positions = collision_data["positions"]
            collision_radii = collision_data["radii"]
        else:
            collision_positions = ()
            collision_radii = ()

        if ee_offset is None:
            ee_offset = tuplify(np.eye(4))
        else:
            ee_offset = np.asarray(ee_offset)
            assert ee_offset.shape == (4, 4)
            ee_offset = tuplify(ee_offset)

        return cls(
            num_joints=data["num_joints"],
            joint_types=data["joint_types"],
            joint_lower_limits=data["joint_lower_limits"],
            joint_upper_limits=data["joint_upper_limits"],
            joint_max_forces=data["joint_max_forces"],
            joint_max_velocities=data["joint_max_velocities"],
            joint_axes=data["joint_axes"],
            joint_parent_frame_positions=data["joint_parent_frame_positions"],
            joint_parent_frame_rotations=data["joint_parent_frame_rotations"],
            link_masses=data["link_masses"],
            link_local_inertias=data["link_local_inertias"],
            link_local_inertia_positions=data["link_local_inertia_positions"],
            link_local_inertia_rotations=data["link_local_inertia_rotations"],
            # TEMP: ignore base data
            base_pos=None,  # tuple(data["base_pos"]),
            base_orn=None,  # tuple(data["base_orn"]),
            ee_offset=ee_offset,
            collision_positions=collision_positions,
            collision_radii=collision_radii,
            joint_names=data['joint_names']
        )

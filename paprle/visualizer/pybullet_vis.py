import numpy as np
import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from paprle.visualizer.pybullet_vis_utils import stdout_redirected
from pytransform3d import transformations as pt
# Large parts of this code are adapted from the oscbf library.
# https://github.com/StanfordASL/oscbf/tree/main
class PyBulletViz:
    def __init__(self, robot, timestep=1/240, load_floor=True, bg_color=None, **kwargs):
        #self.robot = robot # URDF object from yourdfpy
        with stdout_redirected():
            self.client: pybullet = BulletClient(pybullet.GUI)

        assert isinstance(timestep, float) and timestep > 0
        self.client.setTimeStep(timestep)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = self.client.loadURDF(
            robot.urdf_file,
            useFixedBase=True,
            flags=self.client.URDF_USE_INERTIA_FROM_FILE
            | self.client.URDF_MERGE_FIXED_LINKS,
        )

        if load_floor:
            self.floor = self.client.loadURDF("plane.urdf")
            # Get loweset point of the robot
            N = pybullet.getNumJoints(self.robot)
            min_z = 20.0
            for i in range(-1, N):
                aabb = pybullet.getAABB(self.robot, i)
                min_z = min(min_z, aabb[0][2])
            self.client.resetBasePositionAndOrientation(self.floor, [0, 0, min_z - 0.1], [0, 0, 0, 1])
        self.client.configureDebugVisualizer(self.client.COV_ENABLE_GUI, 0)
        if bg_color is not None:
            assert len(bg_color) == 3
            self.client.configureDebugVisualizer(rgbBackground=bg_color)
        self.num_joints = self.client.getNumJoints(self.robot)
        self.sim_joint_names = []
        for i in range(self.num_joints):
            joint = pybullet.getJointInfo(self.robot, i)
            self.sim_joint_names.append(joint[1].decode())

        self.idx_mappings, self.mimic_joint_infos = robot.set_joint_idx_mapping(self.sim_joint_names)

        self.client.setJointMotorControlArray(
            self.robot,
            list(range(self.num_joints)),
            pybullet.VELOCITY_CONTROL,
            forces=[0.1] * self.num_joints,
        )

        self.client.setGravity(0, 0, -9.81)
        self.dt = self.client.getPhysicsEngineParameters()["fixedTimeStep"]
        self.t = 0
        self.position_gains = [0.1] * self.num_joints
        self.limb_names = robot.limb_names
        self.ee_target_handles = None
        self.reset_needed = False
        return

    def init_viewer(self, *args, **kwargs):

        return

    def reset(self, *args, **kwargs):
        return

    def set_ee_targets(self,ee_targets):
        if self.ee_target_handles is None:
            self.ee_target_handles = []
            for limb_name, ee_target in zip(self.limb_names, ee_targets):
                pq = pt.pq_from_transform(ee_target)
                handle = self.client.loadURDF(
                    "models/assets/point_robot.urdf",
                    basePosition=pq[:3],
                    baseOrientation=pq[3:],
                    globalScaling=0.2,
                )
                N = pybullet.getNumJoints(self.robot)
                for i in range(-1, N + 1):
                    self.client.setCollisionFilterPair(self.robot, handle, i, -1, 0)

                self.target_mass = 1.0
                self.client.changeDynamics(handle, -1, linearDamping=10, angularDamping=30)
                self.ee_target_handles.append(handle)
        for handle, ee_target in zip(self.ee_target_handles, ee_targets):
            pq = pt.pq_from_transform(ee_target)
            self.client.resetBasePositionAndOrientation(handle,pq[:3],pq[3:])
            self.client.resetBaseVelocity(handle,[0,0,0],[0,0,0])
        return

    def set_ee_target(self,ee_target, index):
        assert self.ee_target_handles is not None
        handle = self.ee_target_handles[index]
        pq = pt.pq_from_transform(ee_target)
        self.client.resetBasePositionAndOrientation(handle,pq[:3],pq[3:])
        self.client.resetBaseVelocity(handle,[0,0,0],[0,0,0])
        return

    def get_ik_targets(self):
        ik_targets = {}
        for limb_name, handle in zip(self.limb_names, self.ee_target_handles):
            pos, orn = self.client.getBasePositionAndOrientation(handle)
            ik_targets[limb_name] = pt.pq_from_transform(pt.transform_from_pq(np.concatenate([pos, orn])))
            vel, omega = self.client.getBaseVelocity(handle)
            self.client.resetBaseVelocity(handle, vel, [0, 0, 0])
        return ik_targets


    def set_qpos(self, qpos):
        total_qpos = np.zeros(self.num_joints)
        total_qpos[self.idx_mappings] = qpos
        if len(self.mimic_joint_infos):
            total_qpos[self.mimic_joint_infos[:, 0].astype(np.int32)] = total_qpos[self.mimic_joint_infos[:, 1].astype(np.int32)] * self.mimic_joint_infos[:, 2] + self.mimic_joint_infos[:, 3]
        self.client.setJointMotorControlArray(
            self.robot,
            range(self.num_joints),
            pybullet.POSITION_CONTROL,
            targetPositions=total_qpos,
            positionGains=self.position_gains,
        )
        return

    def render(self):
        self.client.stepSimulation()
        self.t += self.dt
        if self.ee_target_handles is not None:
            for handle in self.ee_target_handles:
                # Apply a force to counteract gravity
                self.client.applyExternalForce(
                    handle,
                    -1,
                    [0, 0, 9.81 * self.target_mass],
                    self.client.getBasePositionAndOrientation(handle)[0],
                    self.client.WORLD_FRAME,
                )
        return

    def is_alive(self):
        """Check if the PyBullet client is still alive."""
        try:
            self.client.getConnectionInfo()
            return True
        except pybullet.error:
            return False


if __name__ == "__main__":
    import time
    from paprle.follower import Robot
    from omegaconf import OmegaConf
    from paprle.envs.mujoco_env_utils.util import MultiSliderClass
    from paprle.utils.config_utils import change_working_directory, add_info_robot_config
    change_working_directory()
    robot_name = 'papras_7dof_2arm_table'
    config_file = f'configs/follower/{robot_name}.yaml'
    config = OmegaConf.load(config_file)
    config.robot_cfg = add_info_robot_config(config.robot_cfg)

    urdf_file = config.robot_cfg.asset_cfg.urdf_path
    asset_path = config.robot_cfg.asset_cfg.asset_dir

    robot = Robot(config)

    env = PybulletViz(robot, timestep=1/240)

    slider = MultiSliderClass(
        n_slider      = robot.num_joints,
        title         = 'Sliders for Joint Control',
        window_width  = 600,
        window_height = 800,
        x_offset      = 50,
        y_offset      = 100,
        slider_width  = 350,
        label_texts   = robot.joint_names,
        slider_mins   = robot.joint_limits[:,0],
        slider_maxs   = robot.joint_limits[:,1],
        slider_vals   = robot.init_qpos,
        resolution    = 0.01,
        verbose       = False,
    )

    slider.set_slider_values(robot.init_qpos)
    while env.is_alive():
        slider.update()
        env.render()
        qpos = slider.get_slider_values()
        env.set_qpos(qpos)
        time.sleep(0.01)
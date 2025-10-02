import cv2
import numpy as np
import viser
from viser.extras import ViserUrdf
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr

class ViserViz:
    def __init__(self, robot, *args, server_label='',**kwargs):
        self.server = viser.ViserServer(label=server_label)
        self.server.scene.add_grid("/ground", width=2, height=2)
        if robot is None:
            pass
        else:
            self.urdf_vis = ViserUrdf(self.server, robot.urdf, root_node_name="/base")
            self.link_name_to_frame = {}
            for joint_frame in self.urdf_vis._joint_frames:
                name = joint_frame.name
                elements = name.split('/')
                link_name = '/'.join(elements[-2:]) if len(elements) >= 2 else elements[-1]
                self.link_name_to_frame[link_name] = joint_frame
            self.joint_names = self.urdf_vis.get_actuated_joint_names()
            self.ctrl_joint_idxs = [self.joint_names.index(name) for name in robot.joint_names]
            self.set_qpos(robot.init_qpos)
            self.limb_names = robot.limb_names

        self.ee_target_handles = None
        self.reset_needed = False

        self.image_handles = {}
        self.text_handles = {}
        return

    def init_viewer(self): return

    def set_ee_targets(self, ee_targets, ik_errs=None):
        if self.ee_target_handles is None:
            self.ee_target_handles = []
            for limb_name, ee_target in zip(self.limb_names, ee_targets):
                pq = pt.pq_from_transform(ee_target)
                handle = self.server.scene.add_transform_controls(
                    f"/{limb_name}_ee_target", scale=0.2, position=pq[:3], wxyz=pq[3:]
                )
                self.ee_target_handles.append(handle)
        for handle, ee_target in zip(self.ee_target_handles, ee_targets):
            pq = pt.pq_from_transform(ee_target)
            handle.position = pq[:3]
            handle.wxyz = pq[3:]
        return

    def set_ee_target(self, ee_target, index):
        assert self.ee_target_handles is not None
        handle = self.ee_target_handles[index]
        pq = pt.pq_from_transform(ee_target)
        handle.position = pq[:3]
        handle.wxyz = pq[3:]
        return

    def set_qpos(self, qpos):
        qpos_full = np.zeros(self.urdf_vis._urdf.num_actuated_joints)
        qpos_full[self.ctrl_joint_idxs] = qpos
        self.urdf_vis.update_cfg(qpos_full)
        return

    def reset(self):
        for limb_name, handle in zip(self.limb_names, self.ee_target_handles):
            self.server.scene.remove_by_name(f'/{limb_name}_ee_target')
        self.ee_target_handles = None
        return

    def render(self): return

    def is_alive(self):
        return True

    def get_ik_targets(self):
        ik_targets = {}
        for limb_name, handle in zip(self.limb_names, self.ee_target_handles):
            pq = np.zeros(7)
            pq[:3] = handle.position
            pq[3:] = handle.wxyz
            ik_targets[limb_name] = pq
        return ik_targets

    def create_image_handle(self, img, render_width=None, render_height=None, name="image", position=np.array([0,0,0.75])):
        render_width = img.shape[1]*0.003 if render_width is None else render_width
        render_height = img.shape[0]*0.003 if render_height is None else render_height
        wxyz = np.array([0.0, 0.0, -0.707, 0.707])
        self.image_handles[name] = self.server.scene.add_image(
            f"/{name}", img,
            render_width=render_width, render_height=render_height, position=position, wxyz=wxyz
        )
        return

    def update_image_handle(self, img, name="image"):
        assert name in self.image_handles, f"Image handle {name} does not exist."
        self.image_handles[name].image = img
        return

    def create_text_handle(self, text, name="text", position=np.array([0,0,0.5])):
        self.text_handles[name] = self.server.scene.add_label(f"/{name}", text, position=position)
        return
    def update_text_handle(self, text, name="text"):
        assert name in self.text_handles, f"Text handle {name} does not exist."
        self.text_handles[name].text = text
        return

if __name__ == "__main__":
    import time
    from paprle.follower import Robot
    from omegaconf import OmegaConf
    from paprle.envs.mujoco_env_utils.util import MultiSliderClass
    from paprle.utils.config_utils import change_working_directory, add_info_robot_config
    change_working_directory()
    robot_name = 'papras_teleop_bag'
    config_file = f'configs/follower/{robot_name}.yaml'
    config = OmegaConf.load(config_file)
    config.robot_cfg = add_info_robot_config(config.robot_cfg)

    urdf_file = config.robot_cfg.asset_cfg.urdf_path
    asset_path = config.robot_cfg.asset_cfg.asset_dir

    robot = Robot(config)

    env = ViserViz(robot)

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

    im = np.zeros((480, 640, 3), dtype=np.uint8)
    im = cv2.putText(im, 'Press "q" to quit', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    env.create_image_handle(im, name="info", position=np.array([0,0,1.0]))
    t = 0
    while env.is_alive():
        slider.update()
        env.render()
        qpos = slider.get_slider_values()
        env.set_qpos(qpos)

        time.sleep(0.01)
        t += 1
        if t % 10 == 0:
            im = np.zeros((480, 640, 3), dtype=np.uint8)
            im = cv2.putText(im, f'Press "q" to quit. Time: {t*0.01:.2f}s', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            env.update_image_handle(im, name="info")

import os
import pyrealsense2 as rs
import numpy as np
from threading import Thread
import copy
from multiprocessing import RawArray, Value, Process
import cv2
import time
class RealSenseReader:
    def __init__(self, camera_config, render_depth=False):
        self.cameras, self.threads = {}, []
        self.images = {}
        self.syncer = rs.syncer()
        self.shutdown = False
        self.cam_infos = {}
        fps = 30
        print("Initializing RealSense cameras...")
        for idx, (name, cam_info) in enumerate(camera_config.items()):
            try:
                print("Waiting for camera to initialize: ", name, cam_info.serial_number)
                self.cam_infos[name] = cam_info
                cam = rs.pipeline()
                config = rs.config()
                config.enable_device(str(cam_info.serial_number))
                width, height = cam_info.depth_resolution
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                width, height = cam_info.rgb_resolution
                config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

                profile = cam.start(config)

                #sensor_dep = profile.get_device().first_color_sensor()
                #sensor_dep.set_option(rs.option.exposure, 320.000)

                self.cameras[name] = profile
                self.images[name] = {}
                cam_thread = Thread(target=self.update_camera, args=(name, cam, getattr(cam_info, 'get_aligned', False)))
                cam_thread.start()
                self.threads.append(cam_thread)
                iterations = 0
                while self.images[name] == {}:
                    iterations += 1
                    if iterations % 100 == 0:
                        print("Waiting for camera to initialize: ", name, cam_info.serial_number)
                    time.sleep(0.01)
                    pass
                print("Initialized camera: ", name, cam_info.serial_number)
            except:
                raise
                print("Failed to initialize camera: ", name, cam_info.serial_number)
        self.render_depth = render_depth
    def update_camera(self, name, cam, get_aligned=False):
        while not self.shutdown:
            try:
                frames = cam.wait_for_frames()
            except:
                return
            if get_aligned:
                align = rs.align(rs.stream.color)
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            self.images[name]["color"] = np.array(color_frame.get_data())
            self.images[name]["depth"] = np.array(depth_frame.get_data())
            self.images[name]["timestamp"] = time.time()
            self.images[name]['depth_units'] = depth_frame.get_units()

    def get_status(self):
        return copy.deepcopy(self.images)

    def render(self):
        view_ims = []
        H = None
        for name in self.images.keys():
            im = self.images[name]['color'].copy()
            im = cv2.putText(im, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            image_timestamp = self.images[name]['timestamp']
            time_diff = time.time() - image_timestamp
            if time_diff > 0.1:
                im[30:60, :, 0] = 255  # Red channel for warning
                im = cv2.putText(im, f"Time Diff {time_diff:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if self.render_depth:
                dw, dh = self.cam_infos[name].depth_resolution if not self.cam_infos[name].get_aligned else self.cam_infos[name].rgb_resolution
                depth = np.ascontiguousarray(np.array(self.images[name]['depth']).reshape(dh, dw).astype(np.uint16))
                depth = np.clip(depth * self.images[name]['depth_units'].value, 0, 3.0)/3.0 * 255
                depth = cv2.applyColorMap(depth.astype(np.uint8),  cv2.COLORMAP_JET)
                depth = cv2.resize(depth, (im.shape[1], im.shape[0]))
                im = np.concatenate([im, depth], 0)
            if H is None:
                H = im.shape[0]
            else:
                im = cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H))
            view_ims.append(im)
        view_im = np.concatenate(view_ims, 1)
        view_im = cv2.resize(view_im, dsize=None, fx=0.6, fy=0.6)

        return view_im


if __name__ == "__main__":
    import cv2
    import time
    from omegaconf import OmegaConf
    camera_config = OmegaConf.load("configs/follower/papras_teleop_bag.yaml")
    reader = RealSenseReader(camera_config.camera_cfg)

    # while True:
    #     time.sleep(0.1)
    import imageio
    #imgs = []
    SAVE = False
    if SAVE:
        save_dir = 'realsense_images_{}'.format(time.time())
        os.makedirs(save_dir, exist_ok=True)
    while True:
        ims, timestamps = [], []
        H, W = 0, 0
        for name in reader.images.keys():
            color_im = reader.images[name]['color']
            depth_im = reader.images[name]['depth'] * reader.images[name]['depth_units']
            depth_real_val = np.clip(depth_im, 0.0, 5.0) / 5.0
            depth_color_image = cv2.applyColorMap((depth_real_val * 255).astype(np.uint8), cv2.COLORMAP_JET)
            #depth_color_image = cv2.resize(depth_color_image, (color_im.shape[1], color_im.shape[0]))
            #verlappped = cv2.addWeighted(color_im, 0.5, depth_color_image, 0.5, 0)
            im = color_im
            if H == 0:
                H, W = im.shape[:2]
            else:
                im = cv2.resize(im, (W, H))

            ims.append(im)
            timestamps.append(f"{reader.images[name]['timestamp']:10.04f}")
        view_im = np.concatenate(ims, 1)
        timestamps = ' '.join(timestamps)
        cv2.putText(view_im, timestamps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        view_im = cv2.resize(view_im, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow("view", view_im[:,:,::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.close()

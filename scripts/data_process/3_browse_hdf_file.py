import sys
import os
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
import glob
import argparse
import json
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from paprle.utils.config_utils import change_working_directory
change_working_directory()
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QToolButton,
    QMessageBox, QComboBox, QSlider, QInputDialog, QTextEdit
)
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QColor
from PySide6.QtCore import Qt, Signal
from paprle.utils.gui_utils import RangeSlider
import h5py

task_color_palette = [
    '#ff3573',
    '#985cdc',
    '#FED144',
    '#185D7A',
    '#C8DB2A',
    '#0F830FFF',
]


collector_color_palette = {
    'obin': '#f6bd60',
    'sankalp': '#f7ede2',
    'noboru': '#84a59d',
    'random': '#f28482',
    'unknown': '#2a9d8f',
}

parser = argparse.ArgumentParser(description="Data Processing GUI")
parser.add_argument('--data_dir', type=str, default='/media/obin/02E6-928E/teleop_data/raw_data/fruits/processed/', help='Directory containing episode data')
args = parser.parse_args()


class DraggableHandle(QToolButton):
    moved = Signal(int)  # Emit value position when moved

    def __init__(self, parent, value_callback):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self.setCursor(Qt.OpenHandCursor)
        self.dragging = False
        self.value_callback = value_callback  # maps x -> value

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            global_x = self.mapToParent(event.position().toPoint()).x()
            value = self.value_callback(global_x)
            self.moved.emit(value)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.dragging = False
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

class DataProcessGUI(QWidget):
    def __init__(self, data_dir='demo_data/'):
        super().__init__()
        self.setWindowTitle("Data Browsing GUI")
        self.setGeometry(100, 100, 800, 600)
        self.setFocusPolicy(Qt.StrongFocus)

        self.load_episodes(data_dir)
        self.initUI()
        self.playing = False
        self.on_episode_change(0)

    def load_episodes(self, data_dir='demo_data/'):
        """
        Load episode data from the specified directory.
        """
        if not os.path.exists(data_dir):
            QMessageBox.critical(self, "Error", f"Directory {data_dir} does not exist.")
            return

        ep_list = sorted(glob.glob(os.path.join(data_dir, '*')))

        self.task_dict = {}
        for ep_dir in ep_list:
            if os.path.exists(os.path.join(ep_dir, 'TASK_INFO.json')):
                with open(os.path.join(ep_dir, 'TASK_INFO.json'), 'r') as f:
                    task_info = json.load(f)
                self.task_dict[task_info['name']] = task_info

        self.collector_dict = {}

        self.episode_name_to_dir = {}
        self.episode_name_to_data_list = {}
        self.episode_info_dict = {}
        for ep_dir in tqdm(ep_list, desc="Loading episodes"):
            ep_name = os.path.basename(ep_dir)
            self.episode_name_to_dir[ep_name] = ep_dir
            self.episode_name_to_data_list[ep_name] = ep_dir

            # if len(self.episode_name_to_dir.keys()) > 10: break

        self.episode_list = sorted(self.episode_name_to_dir.keys())

        self.current_episode = None
        self.current_timestamp = 0
        self.start_timestamp, self.end_timestamp = None, None
        self.curr_opened_data = None


    def initUI(self):
        self.main_layout = QVBoxLayout()

        ## First Row
        self.ep_info_layout = QHBoxLayout()

        self.ep_loading_layout = QHBoxLayout()

        self.ep_load_combo = QComboBox()
        self.ep_load_combo.addItems(self.episode_list)
        self.ep_load_combo.currentIndexChanged.connect(self.on_episode_change)
        self.ep_load_combo.setFocusPolicy(Qt.StrongFocus)
        self.ep_loading_layout.addWidget(self.ep_load_combo, alignment=Qt.AlignLeft)

        self.prev_ep_button = QPushButton("Previous Ep")
        self.prev_ep_button.clicked.connect(lambda: self.ep_load_combo.setCurrentIndex(max(0, self.ep_load_combo.currentIndex() - 1)))
        self.ep_loading_layout.addWidget(self.prev_ep_button, alignment=Qt.AlignLeft)

        self.next_ep_button = QPushButton("Next Ep")
        self.next_ep_button.clicked.connect(lambda: self.ep_load_combo.setCurrentIndex(min(len(self.episode_list) - 1, self.ep_load_combo.currentIndex() + 1)))
        self.ep_loading_layout.addWidget(self.next_ep_button, alignment=Qt.AlignLeft)
        self.ep_loading_layout.addStretch()
        self.ep_loading_layout.addSpacing(30)
        self.ep_info_layout.addLayout(self.ep_loading_layout)


        self.main_layout.addLayout(self.ep_info_layout)

        # Second Row
        self.task_info_layout = QHBoxLayout()
        self.task_label = QLabel("Task Label:")
        self.task_name_label = QLabel("N/A")

        self.task_info_layout.addWidget(self.task_label, alignment=Qt.AlignLeft)
        self.task_info_layout.addWidget(self.task_name_label, alignment=Qt.AlignLeft)

        # collector name
        self.human_label = QLabel("Collector:")
        self.collect_label = QLabel("N/A")
        self.task_info_layout.addWidget(self.human_label, alignment=Qt.AlignLeft)
        self.task_info_layout.addWidget(self.collect_label, alignment=Qt.AlignLeft)

        self.task_info_layout.addStretch()
        self.main_layout.addLayout(self.task_info_layout)

        # Third Row
        self.log_layout = QVBoxLayout()
        self.log_label = QLabel(f"Task: {'N/A'} Episode Name: {'N/A'} Success {'N/A'} Start {'N/A'} End {'N/A'}")
        self.log_layout.addWidget(self.log_label, alignment=Qt.AlignCenter)
        self.main_layout.addLayout(self.log_layout)

        # Fourth Row
        self.image_layout = QVBoxLayout()
        self.image_box = QLabel()
        self.image_box.setAlignment(Qt.AlignCenter)
        self.image_layout.addWidget(self.image_box)
        self.main_layout.addLayout(self.image_layout)

        # Fifth Row
        self.slider_layout = QVBoxLayout()

        self.play_slider_layout = QHBoxLayout()
        self.slider_label = QLabel("Timestamp:")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)  # Will be updated based on episode data
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_image)
        self.play_slider_layout.addWidget(self.slider_label, alignment=Qt.AlignLeft)
        self.play_slider_layout.addWidget(self.slider)
        self.slider_layout.addLayout(self.play_slider_layout)

        self.main_layout.addLayout(self.slider_layout)

        # eighth Row
        self.log_box_layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(100)
        self.log_box.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.log_box_layout.addWidget(self.log_box)
        self.main_layout.addLayout(self.log_box_layout)


        self.setLayout(self.main_layout)

    def on_episode_change(self, index):
        if self.curr_opened_data:
            self.curr_opened_data.close()
        self.current_episode = self.episode_list[index]
        self.ep_load_combo.setCurrentIndex(index)

        self.curr_opened_data = h5py.File(self.episode_name_to_data_list[self.current_episode],'r')
        self.current_timestamp = 0
        self.start_timestamp = 0
        self.end_timestamp = self.curr_opened_data['obs']['qpos'].shape[0]-1

        self.collect_label.setText(self.curr_opened_data.attrs['collector'])
        task_name = self.curr_opened_data.attrs['task'] if 'task' in self.curr_opened_data.attrs else 'unknown'
        self.task_name_label.setText(task_name)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.end_timestamp)
        self.slider.setValue(0)
        self.slider_label.setText(f"Timestamp: 0/{self.slider.maximum()}")

        self.update_image(self.current_timestamp)

        self.ep_load_combo.clearFocus()
        self.curr_play_idx = 0
        self.playing = False
        self.update_log_label()
        self.log_box.append(f"[INFO] Loaded episode: {self.current_episode}")

    def update_log_label(self):
        self.log_label.setText(
            f"Episode Name: {self.current_episode} "
            f"Start: {self.start_timestamp} "
            f"End: {self.end_timestamp}"
        )

    def update_image(self, timestamp):
        if self.current_episode:
            rgb = self.curr_opened_data['obs']['rgb'][:, timestamp]
            if 'depth' in self.curr_opened_data['obs']:
                depth = self.curr_opened_data['obs']['depth'][:, timestamp]
            cam_list = list(eval(self.curr_opened_data.attrs['camera_info']).keys())

            for id, cam_name in enumerate(cam_list):
                color_im = rgb[id]
                if 'depth' in self.curr_opened_data['obs']:
                    depth_im = self.make_depth_color(depth[id]*eval(self.curr_opened_data.attrs['camera_info'])[cam_name], max_depth=3.0)
                    im = np.concatenate([color_im, depth_im], axis=0)
                else:
                    im = color_im
                if id == 0:
                    all_camera_ims = [im]
                    H = im.shape[0]
                else:
                    if im.shape[0] != all_camera_ims[0].shape[0]:
                        total_H = all_camera_ims[0].shape[0]
                        im = cv2.resize(im, (int(total_H * im.shape[1] / im.shape[0]), total_H))
                    all_camera_ims.append(im)

            self.view_im = self.resize_image(np.concatenate(all_camera_ims, axis=1))#[...,[2,1,0]]
            self.image_box.setPixmap(self.np_to_qpixmap(self.view_im))
            # change the slider label
            self.slider_label.setText(f"Timestamp: {timestamp}/{self.slider.maximum()}")

    @staticmethod
    def make_depth_color(depth_im, max_depth=3.0):
        depth_im = np.clip(depth_im / max_depth, 0.0, 1.0)
        depth_im = cv2.applyColorMap((depth_im * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., [2, 1, 0]]
        return depth_im

    @staticmethod
    def np_to_qpixmap(np_array):
        height, width, channel = np_array.shape
        bytes_per_line = 3 * width
        if not np_array.flags['C_CONTIGUOUS']:
            np_array = np.ascontiguousarray(np_array)
        q_image = QImage(np_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    @property
    def window_height(self):
        return self.height()

    @property
    def window_width(self):
        return self.width()

    def resize_image(self, orig_color_im):
        # TODO: Lazy implementation, resize every time
        color_im = orig_color_im
        if color_im.shape[0] > self.window_height:
            target_height = self.window_height - 40
            color_im = cv2.resize(color_im, (int(color_im.shape[1] / color_im.shape[0] * target_height), target_height))
        if color_im.shape[1] > self.window_width:
            target_width = self.window_width - 40
            color_im = cv2.resize(color_im, (target_width, int(color_im.shape[0] / color_im.shape[1] * target_width)))
        return color_im

    def play_a_step(self):
        if self.current_episode:
            if self.playing and self.curr_play_idx < self.end_timestamp:
                self.slider.setValue(self.curr_play_idx)
                self.update_image(self.curr_play_idx)
                self.curr_play_idx = min(self.curr_play_idx + 10, self.end_timestamp)
                QApplication.processEvents()
                self.play_a_step()
            elif self.playing and self.curr_play_idx == self.end_timestamp:
                self.playing = False
                self.slider.setValue(self.curr_play_idx)
                self.update_image(self.curr_play_idx)
                QApplication.processEvents()
                self.curr_play_idx = 0
            elif self.playing and self.curr_play_idx > self.end_timestamp:
                self.curr_play_idx = self.start_timestamp
                self.playing = False
            else:
                self.playing = False


    def play_episode(self, idx=None):
        # play episode from start_timestep to end_timestep
        if self.current_episode:
            self.log_box.append(f"[INFO] Playing episode: {self.current_episode} from {self.start_timestamp} to {self.end_timestamp}")
            self.playing = True
            self.play_a_step()


    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            if self.playing:
                self.playing = False
            else:
                self.play_episode()
        elif key == Qt.Key_Q:
            curr_index = self.ep_load_combo.currentIndex()
            prev_index = max(0, curr_index - 1)
            self.on_episode_change(prev_index)
            self.log_box.append("Moved to previous episode.")
            if prev_index == 0:
                self.log_box.append("[INFO] This is the first episode.")
        elif key == Qt.Key_W:
            curr_index = self.ep_load_combo.currentIndex()
            next_index = min(len(self.episode_list) - 1, curr_index + 1)
            self.on_episode_change(next_index)
            self.log_box.append("Moved to next episode.")
            if next_index == len(self.episode_list) - 1:
                self.log_box.append("[INFO] This is the last episode.")






if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataProcessGUI(data_dir=args.data_dir)
    window.show()
    sys.exit(app.exec())
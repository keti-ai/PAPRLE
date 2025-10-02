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
    'Sankalp': '#f7ede2',
    'NobuChan': '#84a59d',
    'random': '#f28482',
    'unknown': '#2a9d8f',
    'sankalp': '#f7ede2',
    'noboru': '#84a59d',
    'random': '#f28482',
    'unknown': '#2a9d8f',
    'pam': '#ff99c8',
    'george': '#fcf6bd',
    'emily': '#d0f4de',
    'allen': '#a9def9',
    'jessica': '#e4c1f9',
}


parser = argparse.ArgumentParser(description="Data Processing GUI")
parser.add_argument('--data_dir', type=str, default='/media/kimlab/02E6-928E/teleop_bag_data', help='Directory containing episode data')
#parser.add_argument('--data_dir', type=str, default='/media/obin/02E6-928E/teleop_data/raw_data/eggs/success/', help='Directory containing episode data')
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
        self.setWindowTitle("Data Processing GUI")
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

        self.collector_dict, self.env_dict = {}, {}

        self.episode_name_to_dir = {}
        self.episode_name_to_data_list = {}
        self.episode_info_dict = {}
        for ep_dir in tqdm(ep_list, desc="Loading episodes"):
            episode_info_file = os.path.join(ep_dir, 'episode_info.pkl')
            if not os.path.exists(episode_info_file):
                print(f"Warning: {episode_info_file} does not exist. Skipping this episode.")
                continue
            data_list = sorted(glob.glob(os.path.join(glob.escape(ep_dir), 'data*.pkl')))
            if len(data_list) > 0:
                ep_name = os.path.basename(ep_dir)
                self.episode_name_to_dir[ep_name] = ep_dir
                self.episode_name_to_data_list[ep_name] = data_list

            # if len(self.episode_name_to_dir.keys()) > 10: break

        self.episode_list = sorted(self.episode_name_to_dir.keys())

        self.current_episode = None
        self.current_timestamp = 0
        self.start_timestamp, self.end_timestamp = None, None


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

        self.ep_labeling_layout = QHBoxLayout()
        self.btn_success = QPushButton("Success")
        self.btn_fail = QPushButton("Fail")
        self.btn_invalid = QPushButton("Invalid")
        self.btn_reset = QPushButton("Reset")
        self.btn_success.clicked.connect(lambda: self.label_ep('success'))
        self.btn_fail.clicked.connect(lambda: self.label_ep('fail'))
        self.btn_invalid.clicked.connect(lambda: self.label_ep('invalid'))
        self.btn_reset.clicked.connect(lambda: self.label_ep(None))
        self.ep_labeling_layout.addWidget(self.btn_success, alignment=Qt.AlignCenter)
        self.ep_labeling_layout.addWidget(self.btn_fail, alignment=Qt.AlignCenter)
        self.ep_labeling_layout.addWidget(self.btn_invalid, alignment=Qt.AlignCenter)
        self.ep_labeling_layout.addWidget(self.btn_reset, alignment=Qt.AlignCenter)
        self.ep_labeling_layout.addStretch()
        self.ep_labeling_layout.addSpacing(30)
        self.ep_info_layout.addLayout(self.ep_labeling_layout)


        self.ep_save_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save Episode")
        self.btn_save.clicked.connect(lambda: self.save_ep_info())
        self.ep_save_layout.addWidget(self.btn_save, alignment=Qt.AlignRight)
        self.ep_info_layout.addLayout(self.ep_save_layout)

        self.main_layout.addLayout(self.ep_info_layout)

        # Second Row
        self.task_info_layout = QHBoxLayout()
        self.task_label = QLabel("Task Label:")
        self.task_combo = QComboBox()
        self.task_combo.setMinimumWidth(200)
        self.task_combo.addItems(list(self.task_dict.keys()))
        self.task_combo.currentIndexChanged.connect(self.change_task)
        self.add_new_task_button = QPushButton("Add New Task")
        self.add_new_task_button.clicked.connect(self.on_add_new_task)

        self.task_info_layout.addWidget(self.task_label, alignment=Qt.AlignLeft)
        self.task_info_layout.addWidget(self.task_combo, alignment=Qt.AlignLeft)
        self.task_info_layout.addWidget(self.add_new_task_button, alignment=Qt.AlignLeft)

        # collector name
        self.human_label = QLabel("Collector:")
        self.human_combo = QComboBox()
        self.human_combo.setMinimumWidth(200)
        self.human_combo.addItems(list(self.collector_dict.keys()))
        self.human_combo.currentIndexChanged.connect(self.change_human)
        self.task_info_layout.addWidget(self.human_label, alignment=Qt.AlignLeft)
        self.task_info_layout.addWidget(self.human_combo, alignment=Qt.AlignLeft)
        self.add_new_human_button = QPushButton("Add New Human")
        self.add_new_human_button.clicked.connect(self.on_add_new_human)
        self.task_info_layout.addWidget(self.add_new_human_button, alignment=Qt.AlignLeft)

        # env name
        self.env_label = QLabel("Env:")
        self.env_combo = QComboBox()
        self.env_combo.setMinimumWidth(200)
        self.env_combo.addItems(list(self.env_dict.keys()))
        self.env_combo.currentIndexChanged.connect(self.change_env)
        self.task_info_layout.addWidget(self.env_label, alignment=Qt.AlignLeft)
        self.task_info_layout.addWidget(self.env_combo, alignment=Qt.AlignLeft)
        self.add_new_env_button = QPushButton("Add New Env")
        self.add_new_env_button.clicked.connect(self.on_add_new_env)
        self.task_info_layout.addWidget(self.add_new_env_button, alignment=Qt.AlignLeft)

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

        self.trim_slider_layout = QHBoxLayout()
        self.trim_slider_label_spacer = QLabel()
        self.trim_slider_label_spacer.setFixedWidth(self.slider_label.sizeHint().width())
        self.trim_slider_layout.addWidget(self.trim_slider_label_spacer)
        self.trim_slider = RangeSlider()
        self.trim_slider.setMinimum(0)
        self.trim_slider.setMaximum(10)  # Will be updated based on episode data
        self.trim_slider.start_btn.moved.connect(self.on_start_handle_moved)
        self.trim_slider.end_btn.moved.connect(self.on_end_handle_moved)
        self.trim_slider_layout.addWidget(self.trim_slider)
        self.slider_layout.addLayout(self.trim_slider_layout)

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

    def on_start_handle_moved(self, value):
        if self.current_episode:
            self.start_timestamp = self.trim_slider.start = value
            self.curr_episode_info['trim_info'] = [self.start_timestamp, self.end_timestamp]
            self.trim_slider.update_start_end()
            self.log_label.setText("Start timestamp set to: {}".format(self.start_timestamp))
            self.trim_slider_label_spacer.setText(f"start: {self.start_timestamp} end: {self.end_timestamp if self.end_timestamp is not None else 'N/A'}")
            self.update_log_label()

    def on_end_handle_moved(self, value):
        if self.current_episode:
            self.end_timestamp = self.trim_slider.end = value
            self.curr_episode_info['trim_info'] = [self.start_timestamp, self.end_timestamp]
            self.trim_slider.update_start_end()
            self.log_box.append("End timestamp set to: {}".format(self.end_timestamp))
            self.trim_slider_label_spacer.setText(f"start: {self.start_timestamp if self.start_timestamp is not None else 'N/A'} end: {self.end_timestamp}")
            self.update_log_label()

    def label_ep(self, label):
        if self.current_episode:
            if label == 'success':
                if self.curr_episode_info['success'] == True:
                    self.curr_episode_info['success'] = None
                else:
                    self.curr_episode_info['success'] = True
            elif label == 'fail':
                if self.curr_episode_info['success'] == False:
                    self.curr_episode_info['success'] = None
                else:
                    self.curr_episode_info['success'] = False
            elif label == 'invalid':
                if self.curr_episode_info['success'] == 'invalid':
                    self.curr_episode_info['success'] = None
                else:
                    self.curr_episode_info['success'] = 'invalid'
            self.update_label_btns()


    def update_label_btns(self):

        self.btn_success.setStyleSheet("")
        self.btn_fail.setStyleSheet("")
        self.btn_invalid.setStyleSheet("")
        if self.curr_episode_info['success'] == True:
            self.btn_success.setStyleSheet("""
                QPushButton {
                    background-color: #2f7d82;
                    color: white;
                }
            """)
        elif self.curr_episode_info['success'] == False:
            self.btn_fail.setStyleSheet("""
                QPushButton {
                    background-color: #b32237;
                    color: white;
                }
            """)
        elif self.curr_episode_info['success'] == 'invalid':
            self.btn_invalid.setStyleSheet("""
                QPushButton {
                    background-color: #716396;
                    color: white;
                }
            """)
        self.update_log_label()
        return

    def on_episode_change(self, index):
        self.current_episode = self.episode_list[index]
        self.ep_load_combo.setCurrentIndex(index)
        episode_info_file = os.path.join(self.episode_name_to_dir[self.current_episode], 'episode_info.pkl')
        try:
            with open(episode_info_file, 'rb') as f:
                self.curr_episode_info = pickle.load(f)
        except:
            return self.on_episode_change(index+1)

        self.current_timestamp = 0
        self.start_timestamp = 0
        self.end_timestamp = len(self.episode_name_to_data_list[self.current_episode]) - 1

        self.curr_episode_info['success'] = self.curr_episode_info.get('success', None)
        self.curr_episode_info['trim_info'] = self.curr_episode_info.get('trim_info', [self.start_timestamp, self.end_timestamp])
        if 'task' in self.curr_episode_info and self.curr_episode_info['task'] is not None:
            if self.curr_episode_info['task'] not in self.task_list:
                self.add_new_task(self.curr_episode_info['task'])
        self.curr_episode_info['task'] = self.curr_episode_info.get('task', None)

        if self.curr_episode_info['task'] is not None:
            task_idx = self.task_combo.findText(self.curr_episode_info['task'])
            self.task_combo.setCurrentIndex(task_idx)
            self.task_combo.setStyleSheet(f"QComboBox {{ background-color: {task_color_palette[task_idx]}; }}")
        elif len(self.task_list) > 0:
            curr_index = self.task_combo.currentIndex() # keep current index
            self.curr_episode_info['task'] = self.task_list[curr_index]
            self.task_combo.setStyleSheet(f"QComboBox {{ background-color: {task_color_palette[curr_index]}; }}")

        if 'collector' in self.curr_episode_info and self.curr_episode_info['collector'] is not None:
            if self.curr_episode_info['collector'] not in self.collector_dict:
                self.add_new_human(self.curr_episode_info['collector'])
        self.curr_episode_info['collector'] = self.curr_episode_info.get('collector', None)
        if self.curr_episode_info['collector'] is not None:
            human_idx = self.human_combo.findText(self.curr_episode_info['collector'])
            self.human_combo.setCurrentIndex(human_idx)
        elif len(self.collector_list) > 0:
            curr_index = self.human_combo.currentIndex()
            self.curr_episode_info['collector'] = self.collector_list[curr_index]

        if 'env' in self.curr_episode_info and self.curr_episode_info['env'] is not None:
            if self.curr_episode_info['env'] not in self.env_dict:
                self.add_new_env(self.curr_episode_info['env'])
        self.curr_episode_info['env'] = self.curr_episode_info.get('env', None)
        if self.curr_episode_info['env'] is not None:
            env_idx = self.env_combo.findText(self.curr_episode_info['env'])
            self.env_combo.setCurrentIndex(env_idx)
        elif len(self.env_list) > 0:
            curr_index = self.env_combo.currentIndex()
            self.curr_episode_info['env'] = self.env_list[curr_index]

        self.episode_info_dict[self.current_episode] = self.curr_episode_info
        self.update_label_btns()

        #print(self.curr_episode_info['episode_name'], self.current_episode)
        self.start_timestamp, self.end_timestamp = self.curr_episode_info['trim_info']
        self.on_start_handle_moved(self.start_timestamp)
        self.on_end_handle_moved(self.end_timestamp)

        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.episode_name_to_data_list[self.current_episode]) - 1)
        self.slider.setValue(0)
        self.slider_label.setText(f"Timestamp: 0/{self.slider.maximum()}")

        self.trim_slider.setMinimum(self.start_timestamp)
        self.trim_slider.setMaximum(len(self.episode_name_to_data_list[self.current_episode]) - 1)
        self.trim_slider.start = self.start_timestamp
        self.trim_slider.end = self.end_timestamp
        self.trim_slider.update_start_end()

        self.update_image(self.current_timestamp)

        self.ep_load_combo.clearFocus()
        self.curr_play_idx = 0
        self.playing = False
        self.update_log_label()
        self.log_box.append(f"[INFO] Loaded episode: {self.current_episode}")

    def update_log_label(self):
        self.log_label.setText(
            f"Task: {self.curr_episode_info['task']} "
            f"Episode Name: {self.current_episode} "
            f"Success: {self.curr_episode_info['success']} " 
            f"Start: {self.start_timestamp} "
            f"End: {self.end_timestamp}"
        )

    def update_image(self, timestamp):
        if self.current_episode:
            data_file = self.episode_name_to_data_list[self.current_episode][timestamp]
            with open(data_file, 'rb') as f:
                try:
                    data = pickle.load(f)
                except:
                    self.log_box.append(f"[ERROR] Failed to load data from {data_file}. Skipping this timestamp.")
                    return
                H = None
                if 'camera' in data['obs']:
                    all_camera_ims = []
                    for camera_name, image_dict in data['obs']['camera'].items():
                        ims = []
                        if 'color' in image_dict:
                            color_im = image_dict['color']
                            if H is None:
                                H = color_im.shape[0]
                            else:
                                color_im = cv2.resize(color_im, (int(H * color_im.shape[1] / color_im.shape[0]), H))
                            ims.append(color_im)
                        if 'depth' in image_dict:
                            depth_im = self.make_depth_color(image_dict['depth']*image_dict['depth_units'], max_depth=3.0)
                            if ims:
                                width = ims[0].shape[1]
                                depth_im = cv2.resize(depth_im, (width, int(width * depth_im.shape[0] / depth_im.shape[1]))) # Resize to match color image width
                            elif H is None:
                                H = depth_im.shape[0] # First depth image sets the height
                            else:
                                depth_im = cv2.resize(depth_im, (int(H * depth_im.shape[1] / depth_im.shape[0]), H)) # Resize to match height
                            ims.append(depth_im)
                        im = np.concatenate(ims, axis=0)
                        if all_camera_ims:
                            if im.shape[0] != all_camera_ims[0].shape[0]:
                                total_H = all_camera_ims[0].shape[0]
                                im = cv2.resize(im, (int(total_H * im.shape[1] / im.shape[0]), total_H))
                        all_camera_ims.append(im)
                        self.view_im = self.resize_image(np.concatenate(all_camera_ims, axis=1))#[...,[2,1,0]]
                else:
                    view_im = np.zeros([480, 640, 3], dtype=np.uint8)
                    view_im = cv2.putText(view_im, "No camera data available", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    self.view_im = view_im
                self.image_box.setPixmap(self.np_to_qpixmap(self.view_im))
            # change the slider label
            self.slider_label.setText(f"Timestamp: {timestamp}/{self.slider.maximum()}")
            self.trim_slider_label_spacer.setFixedWidth(self.slider_label.sizeHint().width())


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

    def save_ep_info(self):
        if self.current_episode:
            episode_info_file = os.path.join(self.episode_name_to_dir[self.current_episode], 'episode_info.pkl')
            with open(episode_info_file, 'wb') as f:
                pickle.dump(self.curr_episode_info, f)
            self.log_box.append(f"[INFO] Episode info saved for {self.current_episode}.")
        return

    def on_add_new_task(self):
        text, ok = QInputDialog.getText(self, "Add New Task", "Enter task name:")
        if ok and text.strip():
            task_name = text.strip()
            if task_name in self.task_dict:
                QMessageBox.warning(self, "Duplicate Task", f"'{task_name}' already exists.")
            else:
                self.add_new_task(task_name)

    def on_add_new_human(self):
        text, ok = QInputDialog.getText(self, "Add New Collector", "Enter collector name:")
        if ok and text.strip():
            human_name = text.strip()
            if human_name in self.collector_dict:
                QMessageBox.warning(self, "Duplicate Collector", f"'{human_name}' already exists.")
            else:
                self.add_new_human(human_name)

    def on_add_new_env(self):
        text, ok = QInputDialog.getText(self, "Add New Env", "Enter env name:")
        if ok and text.strip():
            env_name = text.strip()
            if env_name in self.env_dict:
                QMessageBox.warning(self, "Duplicate Env", f"'{env_name}' already exists.")
            else:
                self.add_new_env(env_name)

    def add_new_task(self, task_name):
        self.task_dict[task_name] = {}
        self.task_combo.addItem(task_name)
        self.task_combo.setCurrentText(task_name)
        idx = self.task_combo.findText(task_name)
        item = self.task_combo.model().item(idx)
        item.setBackground(QColor(task_color_palette[idx]))
        self.task_combo.setStyleSheet(f"QComboBox {{ background-color: {task_color_palette[idx]}; }}")
        self.log_box.append(f"[INFO] Task '{task_name}' added.")

    def add_new_human(self, human_name):
        self.collector_dict[human_name] = {}
        self.human_combo.addItem(human_name)
        self.human_combo.setCurrentText(human_name)
        idx = self.human_combo.findText(human_name)
        self.log_box.append(f"[INFO] Collector '{human_name}' added.")

    def add_new_env(self, env_name):
        self.env_dict[env_name] = {}
        self.env_combo.addItem(env_name)
        self.env_combo.setCurrentText(env_name)
        idx = self.env_combo.findText(env_name)
        self.log_box.append(f"[INFO] Env '{env_name}' added.")


    def change_task(self):
        curr_index = self.task_combo.currentIndex()
        self.task_combo.setCurrentIndex(curr_index)
        self.task_combo.setStyleSheet(f"QComboBox {{ background-color: {task_color_palette[curr_index]}; }}")
        self.curr_episode_info['task'] = self.task_list[self.task_combo.currentIndex()]
        self.log_box.append("Task changed to: {}".format(self.curr_episode_info['task']))
        self.update_log_label()

    def change_human(self):
        curr_index = self.human_combo.currentIndex()
        self.human_combo.setCurrentIndex(curr_index)
        human_name = self.human_combo.currentText()
        self.human_combo.setStyleSheet(f"QComboBox {{ background-color: {collector_color_palette.get(human_name, '#FFFFFF')}; }}")
        self.curr_episode_info['collector'] = self.human_combo.currentText()
        self.log_box.append("Collector changed to: {}".format(self.curr_episode_info['collector']))
        self.update_log_label()

    def change_env(self):
        curr_index = self.env_combo.currentIndex()
        self.env_combo.setCurrentIndex(curr_index)
        self.curr_episode_info['env'] = self.env_combo.currentText()
        self.log_box.append("Env changed to: {}".format(self.curr_episode_info['env']))
        self.update_log_label()

    @property
    def task_list(self):
        return list(self.task_dict.keys())

    @property
    def collector_list(self):
        return list(self.collector_dict.keys())

    @property
    def env_list(self):
        return list(self.env_dict.keys())

    def play_a_step(self):
        if self.current_episode:
            if self.playing and self.curr_play_idx < self.curr_episode_info['trim_info'][1]:
                self.slider.setValue(self.curr_play_idx)
                self.update_image(self.curr_play_idx)
                GG = 10 if self.curr_episode_info['trim_info'][1] < 2000 else 50
                self.curr_play_idx = min(self.curr_play_idx + GG, self.curr_episode_info['trim_info'][1])
                QApplication.processEvents()
                self.play_a_step()
            elif self.playing and self.curr_play_idx == self.curr_episode_info['trim_info'][1]:
                self.playing = False
                self.slider.setValue(self.curr_play_idx)
                self.update_image(self.curr_play_idx)
                QApplication.processEvents()
                start_idx, end_idx = self.curr_episode_info['trim_info']
                self.curr_play_idx = start_idx
            elif self.playing and self.curr_play_idx > self.curr_episode_info['trim_info'][1]:
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
        elif key == Qt.Key_Up:
            # change task
            curr_index = self.task_combo.currentIndex()
            prev_index = (curr_index - 1) % len(self.task_list)
            self.task_combo.setCurrentIndex(prev_index)
            self.task_combo.setStyleSheet(f"QComboBox {{ background-color: {task_color_palette[prev_index]}; }}")
            self.curr_episode_info['task'] = self.task_list[self.task_combo.currentIndex()]
            self.log_box.append("Task changed to: {}".format(self.curr_episode_info['task']))
        elif key == Qt.Key_Down:
            curr_index = self.task_combo.currentIndex()
            if len(self.task_list) == 0: return
            next_index = (curr_index + 1) % len(self.task_list)
            self.task_combo.setCurrentIndex(next_index)
            self.task_combo.setStyleSheet(f"QComboBox {{ background-color: {task_color_palette[next_index]}; }}")
            self.curr_episode_info['task'] = self.task_list[self.task_combo.currentIndex()]
            self.log_box.append("Task changed to: {}".format(self.curr_episode_info['task']))
        elif key == Qt.Key_Q:
            self.save_ep_info()
            curr_index = self.ep_load_combo.currentIndex()
            prev_index = max(0, curr_index - 1)
            self.on_episode_change(prev_index)
            self.log_box.append("Moved to previous episode.")
            if prev_index == 0:
                self.log_box.append("[INFO] This is the first episode.")
        elif key == Qt.Key_W:
            self.save_ep_info()
            curr_index = self.ep_load_combo.currentIndex()
            next_index = min(len(self.episode_list) - 1, curr_index + 1)
            self.on_episode_change(next_index)
            self.log_box.append("Moved to next episode.")
            if next_index == len(self.episode_list) - 1:
                self.log_box.append("[INFO] This is the last episode.")
        elif key == Qt.Key_1:
            # Label as success
            self.label_ep('success')
        elif key == Qt.Key_2:
            # Label as fail
            self.label_ep('fail')
        elif key == Qt.Key_3:
            # Label as invalid
            self.label_ep('invalid')
        # elif key == Qt.Key_7:
        #     self.on_start_handle_moved(self.start_timestamp)
        # elif key == Qt.Key_9:
        #     self.on_end_handle_moved(self.end_timestamp)
        elif key == Qt.Key_S:
            self.save_ep_info()
        elif key == Qt.Key_7:
            # change human name
            curr_index = self.human_combo.currentIndex()
            prev_index = (curr_index - 1) % len(self.collector_dict)
            self.human_combo.setCurrentIndex(prev_index)
            self.curr_episode_info['collector'] = self.human_combo.currentText()
            self.log_box.append("Collector changed to: {}".format(self.curr_episode_info['collector']))
        elif key == Qt.Key_9:
            curr_index = self.human_combo.currentIndex()
            if len(self.collector_dict) == 0: return
            next_index = (curr_index + 1) % len(self.collector_dict)
            self.human_combo.setCurrentIndex(next_index)
            self.curr_episode_info['collector'] = self.human_combo.currentText()
            self.log_box.append("Collector changed to: {}".format(self.curr_episode_info['collector']))
        elif key == Qt.Key_4:
            curr_index = self.env_combo.currentIndex()
            prev_index = (curr_index - 1) % len(self.env_dict)
            self.env_combo.setCurrentIndex(prev_index)
            self.curr_episode_info['env'] = self.env_combo.currentText()
            self.log_box.append("Env changed to: {}".format(self.curr_episode_info['env']))
        elif key == Qt.Key_6:
            curr_index = self.env_combo.currentIndex()
            if len(self.env_dict) == 0: return
            next_index = (curr_index + 1) % len(self.env_dict)
            self.env_combo.setCurrentIndex(next_index)
            self.curr_episode_info['env'] = self.env_combo.currentText()
            self.log_box.append("Env changed to: {}".format(self.curr_episode_info['env']))




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DataProcessGUI(data_dir=args.data_dir)
    window.show()
    sys.exit(app.exec())
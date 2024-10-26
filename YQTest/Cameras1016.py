import sys
import cv2
import threading
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QComboBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
import os

class CameraRecorder(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.setWindowTitle("Select Camera Port for Recording")
        self.setGeometry(100, 100, 800, 600)

        # 垂直布局
        self.layout = QVBoxLayout()

        # 摄像头选择器
        self.camera_selector = QComboBox(self)
        self.layout.addWidget(self.camera_selector)

        # 摄像头预览显示标签
        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(640, 480)  # 设置预览区域大小
        self.layout.addWidget(self.camera_label)

        # 开始和停止按钮
        self.start_button = QPushButton("Start Recording", self)
        self.start_button.clicked.connect(self.start_recording)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Recording", self)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)  # 初始禁用
        self.layout.addWidget(self.stop_button)

        self.setLayout(self.layout)

        # 摄像头和定时器
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_recording = False
        self.frames = {}
        self.timestamps = {}
        self.output_dir = "output"

        # 添加可用摄像头端口选项（假设有6个端口）
        self.available_ports = [0, 1, 2, 3, 4, 5]
        for port in self.available_ports:
            self.camera_selector.addItem(f"Camera Port {port}", port)

    def start_recording(self):
        # 获取用户选择的摄像头端口
        selected_port = self.camera_selector.currentData()

        # 打开选择的摄像头
        self.cap = cv2.VideoCapture(selected_port)

        if not self.cap.isOpened():
            print(f"Cannot open camera at port {selected_port}")
            return

        self.is_recording = True

        # 创建保存文件夹
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 启动定时器，每30毫秒更新一次帧
        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_recording(self):
        self.is_recording = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.camera_label.clear()  # 清空预览窗口
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        # 从摄像头读取帧
        ret, frame = self.cap.read()
        if ret:
            # 将BGR图像转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 显示帧
            self.camera_label.setPixmap(QPixmap.fromImage(q_image).scaled(self.camera_label.size(), Qt.KeepAspectRatio))

            # 记录时间戳和帧
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            frame_count = len(self.frames)
            self.frames[f"frame_{frame_count}"] = frame
            self.timestamps[f"frame_{frame_count}"] = timestamp

            # 保存当前帧为图像文件
            frame_filename = os.path.join(self.output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraRecorder()
    window.show()
    sys.exit(app.exec_())

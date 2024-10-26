import sys
import cv2
import threading
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QComboBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class MultiCameraPreview(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口
        self.setWindowTitle("Camera Selection and Recording")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()

        # 标签显示摄像头预览画面
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.preview_label)

        # 摄像头选择框
        self.camera_selector = QComboBox(self)
        self.layout.addWidget(self.camera_selector)

        # 刷新摄像头预览按钮
        self.refresh_button = QPushButton("Refresh Camera Previews", self)
        self.refresh_button.clicked.connect(self.refresh_cameras)
        self.layout.addWidget(self.refresh_button)

        # 开始录像按钮
        self.record_button = QPushButton("Start Recording", self)
        self.record_button.clicked.connect(self.start_recording)
        self.record_button.setEnabled(False)  # 默认情况下禁用
        self.layout.addWidget(self.record_button)

        # 停止录像按钮
        self.stop_button = QPushButton("Stop Recording", self)
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        self.setLayout(self.layout)

        # 摄像头预览和录像相关变量
        self.cameras = []
        self.current_camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.is_recording = False
        self.out = None
        self.frame_size = (640, 480)  # 默认大小

    def refresh_cameras(self):
        # 刷新摄像头列表
        self.cameras = self.detect_cameras()
        self.camera_selector.clear()

        # 如果有摄像头，添加到选择框并预览第一个
        if self.cameras:
            for i in range(len(self.cameras)):
                self.camera_selector.addItem(f"Camera {i}")
            self.preview_camera(0)
            self.record_button.setEnabled(True)  # 如果有摄像头，可以启用录像按钮

        else:
            self.preview_label.clear()
            self.record_button.setEnabled(False)

    def detect_cameras(self):
        # 检测可用摄像头，返回摄像头索引列表
        available_cameras = []
        for index in range(6):  # 假设最多6个摄像头
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()
        return available_cameras

    def preview_camera(self, index):
        # 预览选择的摄像头
        if self.current_camera is not None:
            self.current_camera.release()

        self.current_camera = cv2.VideoCapture(self.cameras[index])
        self.timer.start(30)  # 每30ms更新一次

    def update_preview(self):
        # 更新摄像头预览画面
        ret, frame = self.current_camera.read()
        if ret:
            # 获取帧的宽度和高度
            self.frame_size = (frame.shape[1], frame.shape[0])

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(q_image).scaled(self.preview_label.size(), Qt.KeepAspectRatio))

    def start_recording(self):
        # 获取选择的摄像头索引
        selected_camera_index = self.camera_selector.currentIndex()
        if selected_camera_index >= 0 and self.cameras:
            self.preview_camera(selected_camera_index)
            self.record_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            # 设置视频输出格式，使用检测到的帧大小
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter('output.avi', fourcc, 20.0, self.frame_size)

            # 开始录像
            self.is_recording = True
            threading.Thread(target=self.record_video, daemon=True).start()

    def record_video(self):
        # 在单独的线程中进行录像
        while self.is_recording:
            ret, frame = self.current_camera.read()
            if ret:
                self.out.write(frame)

    def stop_recording(self):
        # 停止录像
        self.is_recording = False
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.out.release()

    def closeEvent(self, event):
        # 关闭窗口时确保释放资源
        self.timer.stop()
        if self.current_camera is not None:
            self.current_camera.release()
        if self.out is not None:
            self.out.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiCameraPreview()
    window.show()
    sys.exit(app.exec_())

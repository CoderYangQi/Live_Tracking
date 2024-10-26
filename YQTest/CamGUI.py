import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class CameraViewer(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        # 初始化摄像头列表
        self.cameras = []
        self.cap = [None] * 6
        self.timer = [None] * 6
        self.running = [False] * 6

        self.list_cameras()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)

        self.labels = []
        self.buttons = []

        for i in range(6):
            label = QLabel(self)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(320, 240)
            grid.addWidget(label, i // 2, (i % 2) * 2)
            self.labels.append(label)

            button = QPushButton(f'Close Camera {i+1}', self)
            button.clicked.connect(lambda checked, i=i: self.close_camera(i))
            grid.addWidget(button, i // 2, (i % 2) * 2 + 1)
            self.buttons.append(button)

        self.setWindowTitle('Camera Viewer')
        self.show()

    def list_cameras(self):
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                break
            self.cameras.append(index)
            self.cap[index] = cap
            self.timer[index] = QTimer(self)
            self.timer[index].timeout.connect(lambda i=index: self.update_frame(i))
            self.timer[index].start(30)
            self.running[index] = True
            index += 1

    def update_frame(self, cam_index):
        if self.cap[cam_index] and self.running[cam_index]:
            ret, frame = self.cap[cam_index].read()
            if ret:
                frame = cv2.resize(frame, (320, 240))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.labels[cam_index].setPixmap(QPixmap.fromImage(q_img))
            else:
                self.close_camera(cam_index)

    def close_camera(self, cam_index):
        if self.running[cam_index]:
            self.timer[cam_index].stop()
            self.cap[cam_index].release()
            self.labels[cam_index].clear()
            self.running[cam_index] = False

    def closeEvent(self, event):
        for i in range(len(self.cap)):
            if self.running[i]:
                self.close_camera(i)
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = CameraViewer()
    sys.exit(app.exec_())

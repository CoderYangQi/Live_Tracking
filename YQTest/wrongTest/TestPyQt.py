import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, \
    QFileDialog, QComboBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from yq_dlclive import YQDLCLive
from pathlib import Path
class RealTimeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Face Detection")

        # Initialize components
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.video_layout = QHBoxLayout()
        self.layout.addLayout(self.video_layout)

        self.video_label = QLabel(self)
        self.video_layout.addWidget(self.video_label)

        self.result_label = QLabel(self)
        self.video_layout.addWidget(self.result_label)

        self.control_layout = QVBoxLayout()
        self.layout.addLayout(self.control_layout)

        self.model_selector = QComboBox(self)
        self.model_selector.addItem("Default Face Detector")
        self.control_layout.addWidget(self.model_selector)

        self.load_video_button = QPushButton("Load Video", self)
        self.load_video_button.clicked.connect(self.load_video)
        self.control_layout.addWidget(self.load_video_button)

        self.start_camera_button = QPushButton("Start Camera", self)
        self.start_camera_button.clicked.connect(self.start_camera)
        self.control_layout.addWidget(self.start_camera_button)

        self.stop_camera_button = QPushButton("Stop Camera", self)
        self.stop_camera_button.clicked.connect(self.stop_camera)
        self.stop_camera_button.setEnabled(False)
        self.control_layout.addWidget(self.stop_camera_button)
        # 定时器 触发 不断刷新GUI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.capture = None
        self.video_path = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.pcutoff = 0.25
        self.initModel()
    def load_video(self):
        self.video_path = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.avi *.mp4 *.mov)")[0]
        if self.video_path:
            self.capture = cv2.VideoCapture(self.video_path)
            self.start_camera_button.setEnabled(False)
            self.stop_camera_button.setEnabled(True)
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            n_frames = 1000
            n_frames = (
                n_frames
                if (n_frames > 0) and (n_frames < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
                else (cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
            )
            n_frames = int(n_frames)
            if ret:
                self.live.init_inference(frame)

            self.timer.start(20)

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.video_path = None
        self.start_camera_button.setEnabled(False)
        self.stop_camera_button.setEnabled(True)
        self.timer.start(20)

    def stop_camera(self):
        if self.capture:
            self.timer.stop()
            self.capture.release()
        self.video_label.clear()
        self.result_label.clear()
        self.start_camera_button.setEnabled(True)
        self.stop_camera_button.setEnabled(False)

    def update_frame(self):
        ret, frame = self.capture.read()
        flag = True
        if ret and flag:
            pose,frame = self.live.get_pose(frame)
            self.display_frame(frame)
            self.display_pose_results(pose)
        elif ret and not flag:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            self.display_frame(frame)
            self.display_results(faces)
        else:
            self.stop_camera()

    def display_frame(self, frame):
        # Convert the image to a QImage for display
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def display_results(self, faces):
        results_text = f"Detected Faces: {len(faces)}"
        for i, (x, y, w, h) in enumerate(faces, start=1):
            results_text += f"\nFace {i}: X={x}, Y={y}, W={w}, H={h}"
        self.result_label.setText(results_text)
    def display_pose_results(self, pose):
        results_text = f"Detected Pose: {len(pose)}"
        for i in range(pose.shape[0]):
            color = (0, 255, 0)  # 绿色
            if pose[i, 2] > self.pcutoff:
                results_text += f"\npose {i}: X={pose[i,0]}, Y={pose[i,1]}, conf={pose[2]}"
        self.result_label.setText(results_text)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

    def initModel(self):
        model_dir = r"D:\USERS\yq\code\MotionTracking\DLC_Live\DLC_Dog_resnet_50_iteration-0_shuffle-0"
        model_dir = Path(model_dir)
        video_file = r'D:\USERS\yq\code\MotionTracking\DLC_Live\check_install_dog_clip.avi'
        # benchmark_videos(str(model_dir), video_file, display=display, resize=0.5, pcutoff=0.25)
        model_path = str(model_dir)
        video_path = video_file
        tf_config = None;display = False
        dynamic = (False, 0.5, 10)
        n_frames = 1000
        display_radius = 3
        tf_config = tf_config
        resize = 0.5
        pixels = None
        cropping = None
        dynamic = dynamic
        n_frames = n_frames
        print_rate = False
        display = display
        pcutoff = 0.25
        display_radius = display_radius
        cmap = 'bmy'
        save_poses = False
        save_video = False
        output = None
        self.live = YQDLCLive(
            model_path,
            tf_config=tf_config,
            resize=resize,
            cropping=cropping,
            dynamic=dynamic,
            display=display,
            pcutoff=pcutoff,
            display_radius=display_radius,
            display_cmap=cmap,
        )
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealTimeDetectionApp()
    window.show()
    sys.exit(app.exec_())

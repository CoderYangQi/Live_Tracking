import sys
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, \
    QFileDialog, QComboBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
# from yq_dlclive import YQDLCLive
from pathlib import Path
from Utils import *
import time
from MoveUtils import Move
from calibration import calib
from ctypes import *
import ctypes
#import numpy as np
import struct 
import time
import math
class RealTimeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Pose Detection")
        self.index = 1

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

        self.usedTime_label = QLabel(self)
        self.video_layout.addWidget(self.usedTime_label)

        self.backCenter_label = QLabel(self)
        self.video_layout.addWidget(self.backCenter_label)


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
        self.counter = 0
        self.initModel()
        self.poseList = []
        self.radius = 3

        # self.moveDll = Move()
        # self.poseinit = [827, 314]
        self.poseinit = [729,  463]

        self.mycalib = calib()
        self.mycalib.Test()
        # self.init_threshold()
    def moveX(self,xValue):
        dll = CDLL(r'E:\yq\code\ControlModel\GAS.dll')
        print(dll)

        dll.GA_StartDebugLog(1)

        print('测试开始')
        a=dll.GA_OpenByIP(b'192.168.0.2',b'192.168.0.1',0,0)
        #a=dll.GA_Open(1,"COM1")
        print('打开板卡GA_Open返回值:',a)

        a=dll. GA_Reset()
        print('复位板卡GA_Reset返回值:',a)

        # axis = 1
        # value = 80 # - 右边
        axis = 1
        value = xValue # - 右边



        #  -  后面
                        # + 
        a=dll.GA_EncOff(axis)
        print(f'关闭轴{axis}编码器:',a)

        a=dll.GA_ZeroPos(1,3)
        print(f'清零轴{axis}零位，返回值:',a)

        a=dll.GA_AxisOn(axis)
        print(f'使能轴{axis}返回值:',a)

        a=dll.GA_PrfTrap(axis)
        print(f'设置轴{axis}进入点位模式，返回值:',a)

        a=dll.GA_SetTrapPrmSingle(axis,c_double(1.0),c_double(1.0),c_double(0.0),0)
        print(f'设置轴{axis}点位运动参数，返回值:',a)

        # while 1:
        dPrfPos = c_double(0.0)
        print('打开输出口Y5')
        a=dll.GA_SetExtDoBit(0, 5, 1)
        a=dll.GA_SetPos(axis,value)
        print(f'设置轴{axis}运动目标位置为20000脉冲的位置，返回值:',a)
        a=dll.GA_SetVel(axis,c_double(7.5))
        print(f'设置轴{axis}运动速度为7.5脉冲/毫秒，返回值:',a)
        a=dll.GA_Update(axis)
        print(f'启动轴{axis}运动')
        print('延时5秒钟')
        time.sleep(5)

        a = dll.GA_GetPrfPos(axis, byref(dPrfPos),1,0)
        dValue = dPrfPos
        print(f'获取轴{axis}脉冲位置，返回值：',a,'获取值：',dValue)

        lSts = c_long(0)
        a = dll.GA_GetSts(axis, byref(lSts),1,0)
        print(f'获取轴{axis}状态，返回值：',a,'获取值：',lSts)



        print('关闭输出口Y5')
        a = dll.GA_SetExtDoBit(0, 5, 0)
            
        a=dll.GA_Close()
        print('测试结束')
    def moveY(self,Yvalue):
        dll = CDLL(r'E:\yq\code\ControlModel\GAS.dll')
        print(dll)

        dll.GA_StartDebugLog(1)

        print('测试开始')
        a=dll.GA_OpenByIP(b'192.168.0.2',b'192.168.0.1',0,0)
        #a=dll.GA_Open(1,"COM1")
        print('打开板卡GA_Open返回值:',a)

        a=dll. GA_Reset()
        print('复位板卡GA_Reset返回值:',a)

        # axis = 1
        # value = 80 # - 右边
        axis = 2
        value = Yvalue # - 右边



        #  -  后面
                        # + 
        a=dll.GA_EncOff(axis)
        print(f'关闭轴{axis}编码器:',a)

        a=dll.GA_ZeroPos(1,3)
        print(f'清零轴{axis}零位，返回值:',a)

        a=dll.GA_AxisOn(axis)
        print(f'使能轴{axis}返回值:',a)

        a=dll.GA_PrfTrap(axis)
        print(f'设置轴{axis}进入点位模式，返回值:',a)

        a=dll.GA_SetTrapPrmSingle(axis,c_double(1.0),c_double(1.0),c_double(0.0),0)
        print(f'设置轴{axis}点位运动参数，返回值:',a)

        # while 1:
        dPrfPos = c_double(0.0)
        print('打开输出口Y5')
        a=dll.GA_SetExtDoBit(0, 5, 1)
        a=dll.GA_SetPos(axis,value)
        print(f'设置轴{axis}运动目标位置为20000脉冲的位置，返回值:',a)
        a=dll.GA_SetVel(axis,c_double(7.5))
        print(f'设置轴{axis}运动速度为7.5脉冲/毫秒，返回值:',a)
        a=dll.GA_Update(axis)
        print(f'启动轴{axis}运动')
        print('延时5秒钟')
        time.sleep(5)

        a = dll.GA_GetPrfPos(axis, byref(dPrfPos),1,0)
        dValue = dPrfPos
        print(f'获取轴{axis}脉冲位置，返回值：',a,'获取值：',dValue)

        lSts = c_long(0)
        a = dll.GA_GetSts(axis, byref(lSts),1,0)
        print(f'获取轴{axis}状态，返回值：',a,'获取值：',lSts)



        print('关闭输出口Y5')
        a = dll.GA_SetExtDoBit(0, 5, 0)
        # a=dll.GA_SetPos(1,0)
        # print('设置轴1运动目标位置为0脉冲的位置，返回值:',a)
        # a=dll.GA_Update(1)
        # print('启动轴1运动')
        # print('延时5秒钟')
        # time.sleep(5)
        # dPrfPos = c_double(0)
        # a = dll.GA_GetPrfPos(1, byref(dPrfPos),1,0)
        # dValue = dPrfPos
        # print('获取轴1脉冲位置，返回值：',a,'获取值：',dValue)
            
        a=dll.GA_Close()
        print('测试结束')
    def init_threshold(self):
        image_path = r'E:\yq\code\DLC_Project\12.jpg'  # 替换为你的图像路径
        image = cv2.imread(image_path)
        roi = [[420,385,490,455], [1600, 20, 1720, 120], [830, 830, 900, 940]]
        # 将ROI区域设为黑色 (0, 0, 0) 对应黑色
        image[roi[0][1]:roi[0][3], roi[0][0]:roi[0][2]] = (0, 0, 0)
        image[roi[1][1]:roi[1][3], roi[1][0]:roi[1][2]] = (0, 0, 0)
        image[roi[2][1]:roi[2][3], roi[2][0]:roi[2][2]] = (0, 0, 0)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # point_roi = [407,983]
        # radius = 5
        # point = image[point_roi[0] - radius:point_roi[0] + radius, point_roi[1] - radius:point_roi[1] + radius]
        # 提取指定的point区域

        point_roi = [983,407]
        radius = 5
        point = hsv_image[point_roi[1] - radius:point_roi[1] + radius, point_roi[0] - radius:point_roi[0] + radius]
        mean_hsv = np.mean(point, axis=(0, 1))  # 计算point区域的HSV均值
        threshold = np.array([10, 10, 10])  # 可以根据需要调整H, S, V的偏差范围
        lower_bound = np.clip(mean_hsv - threshold, [0, 0, 0], [180, 255, 255])  # HSV范围的下界
        upper_bound = np.clip(mean_hsv + threshold, [0, 0, 0], [180, 255, 255])  # HSV范围的上界
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def detect_red_dots(self,image,lower_bound, upper_bound, region_top_left, region_bottom_right):
        # 转换图像到HSV色彩空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # # 定义红色的HSV范围 (这里选择两个范围，因为红色在HSV中跨越180度)
        # lower_red1 = np.array([0, 50, 50])  # 红色的下界
        # upper_red1 = np.array([10, 255, 255])  # 红色的上界
        # lower_red2 = np.array([170, 50, 50])
        # upper_red2 = np.array([180, 255, 255])
        #
        # # 创建掩码，过滤出红色部分
        # mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        # mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        # red_mask = cv2.bitwise_or(mask1, mask2)

        red_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # 指定的检测区域，裁剪图像和掩码
        region = image[region_top_left[1]:region_bottom_right[1], region_top_left[0]:region_bottom_right[0]]
        region_mask = red_mask[region_top_left[1]:region_bottom_right[1], region_top_left[0]:region_bottom_right[0]]
        # cv2.imshow("region", region)

        # 寻找红色点的轮廓
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        red_dots_positions = []

        for contour in contours:
            # 计算每个红色区域的中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"]) + region_top_left[0]
                cY = int(M["m01"] / M["m00"]) + region_top_left[1]
                red_dots_positions.append((cX, cY))
                # 在原始图像上标记红点
                cv2.circle(image, (cX, cY), 5, (0, 255, 255), -1)  # 绿色圆圈表示检测到的红点

        return red_dots_positions, image

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
            # if ret:
            #     self.live.init_inference(frame)

            self.timer.start(20)

    def start_camera(self):
        self.capture = cv2.VideoCapture(self.index)
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置为最大宽度（你可以修改为具体最大值）
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置为最大高度
        self.capture.set(cv2.CAP_PROP_FPS, 30)  # 设置为最大帧率
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
        self.usedTime_label.clear()
        self.backCenter_label.clear()

        self.start_camera_button.setEnabled(True)
        self.stop_camera_button.setEnabled(False)
        # a=self.moveDll.GA_Close()
        # print('测试结束')

    def update_frame(self):
        ret, frame = self.capture.read()
        flag = True
        TFGPUinference = True
        cfg = self.cfg
        usedTime = 0
        if ret and flag:
            # if cfg["cropping"]:
            #     ny, nx = checkcropping(cfg, cap)

            pose_tensor = predict.extract_GPUprediction(
                self.outputs, self.dlc_cfg
            )  # extract_output_tensor(outputs, dlc_cfg)
            # PredictedData = np.zeros((nframes, 3 * len(self.dlc_cfg["all_joints_names"])))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg["cropping"]:
                frame = img_as_ubyte(
                    frame[cfg["y1"]: cfg["y2"], cfg["x1"]: cfg["x2"]]
                )
            else:
                frame = img_as_ubyte(frame)
            start = time.time()
            pose = self.sess.run(
                pose_tensor,
                feed_dict={self.inputs: np.expand_dims(frame, axis=0).astype(float)},
            )
            usedTime = time.time() - start
            pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
            # pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
            # 定义每个点的颜色 (BGR格式)
            colors = [(255, 0, 0),  # 蓝色
                      (0, 255, 0),  # 绿色
                      (0, 0, 255),  # 红色
                      (255, 255, 0)]  # 黄色
            for i in range(pose.shape[0]):
                # color = (0, 255, 0)  # 绿色
                # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                # if pose[i, 2] > self.pcutoff:
                frame = cv2.circle(frame, (pose[i, 0:2]).astype(np.int32), self.radius, colors[i % 4], -1)
            print(f'pose is {pose[4, 0:2]}')
            meanPose = np.mean(pose,axis=1)
            # center
            vector = [frame.shape[0]//2 - meanPose[0],frame.shape[1]//2 - meanPose[1]]
            print(f"used time is {usedTime}")
            print(f"vector is {vector}")
            print(f"center is {meanPose}")
            self.counter += 1
            self.display_frame(frame)
            self.display_pose_results(pose)
            self.poseList.append(pose)
            self.display_usedTime(usedTime)
            self.display_back2center(vector)
            midPose = pose[1, 0:2]
            if abs((midPose[0] - self.poseinit[0])) > 50 or abs(midPose[1] - self.poseinit[1]) > 50 :
                
                world_coords1 = self.mycalib.projectPointNoUnistort(self.poseinit[0], self.poseinit[1])
                world_coords2 = self.mycalib.projectPointNoUnistort(midPose[0], midPose[1])
                # diff = wor
                vector = world_coords1 - world_coords2
                print(f"vector is {vector}")
                block = 12800 / 72
                distX = vector[0] * block
                distY = vector[1] * block
                print(f"distX is {distX} distY is {distY} ")
                # self.moveDll.moveXY(int(distX), int(distY))
                self.moveX(int(distX))
                self.moveY(int(distY))
                time.sleep(10)
                self.poseinit = midPose



            # PredictedData[
            # self.counter, :
            # ] = (
            #     pose.flatten()
            # )  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
            # elif counter >= nframes:
            #     break
            # counter += 1

            # pbar.close()
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
        rgb_image = frame
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def display_results(self, faces):
        results_text = f"Detected Faces: {len(faces)}"
        for i, (x, y, w, h) in enumerate(faces, start=1):
            results_text += f"\nFace {i}: X={x}, Y={y}, W={w}, H={h}"
        self.result_label.setText(results_text)

    def display_usedTime(self, usedTime):
        usedTime_text = f"used time is : {usedTime}"
        self.usedTime_label.setText(usedTime_text)

    def display_back2center(self, vector):
        vector_text = f"vector is : {vector}"
        self.backCenter_label.setText(vector_text)
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
        # configPath =  r'E:\yq\code\DataAndModel\dlc-models\iteration-0\Test2Jul27-trainset95shuffle1\test\pose_cfg.yaml'
        configPath =  r'E:\yq\code\EleMice2mins-Test-2024-10-16\dlc-models\iteration-0\EleMice2minsOct16-trainset95shuffle1\test\pose_cfg.yaml'
        # weightPath =  r"E:\yq\code\DataAndModel\\dlc-models\\iteration-0\\Test2Jul27-trainset95shuffle1\\train\\snapshot-100000"
        weightPath =  r"E:\yq\code\EleMice2mins-Test-2024-10-16\dlc-models\iteration-0\EleMice2minsOct16-trainset95shuffle1\train\snapshot-300000"
        path_test_config = Path(configPath)
        self.dlc_cfg = load_config(str(path_test_config))
        self.dlc_cfg[
            "init_weights"] = weightPath

        cfg = {}
        cfg["cropping"] = False
        self.cfg = cfg

        self.sess, self.inputs, self.outputs = setup_GPUpose_prediction(self.dlc_cfg, allow_growth=False)
        # self.sess, self.inputs, self.outputs = setup_pose_prediction(self.dlc_cfg, allow_growth=False)
        pass
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealTimeDetectionApp()
    window.show()
    sys.exit(app.exec_())

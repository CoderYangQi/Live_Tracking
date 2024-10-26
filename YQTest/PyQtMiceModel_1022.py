from queue import Queue
import sys
import threading
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
# 加载机械臂控制DLL
# a=dll.GA_ZeroPos(1,8)
dPrfPosX = c_double(0)
dPrfPosY = c_double(0)

# 定义队列用于在线程之间传递 xValue 和 yValue
command_queue = Queue()

# 机械轴控制类
class MechanicalAxisController:
    def __init__(self):
        self.init_motion_system()

    # 初始化系统
    def init_motion_system(self):
        self.dll = CDLL(r"E:\yq\code\MovingInterpolation2D\GAS.dll")

        print(self.dll)

        self.dll.GA_StartDebugLog(1)

        print('测试开始')
        a=self.dll.GA_OpenByIP(b'192.168.0.2',b'192.168.0.1',0,0)
        print('打开板卡GA_Open返回值:',a)


        a=self.dll.GA_StartLog()
        print('打开日志功能，平时不用可以关闭本段代码，返回值:',a)
        "产生的日志文件夹名字为 RunTimeLog 可以在电脑里面搜索该文件夹。通常在Pyhon安装路径"

        a=self.dll. GA_Reset()
        print('复位板卡GA_Reset返回值:',a)

        a=self.dll.GA_EncOff(1)
        print('关闭轴1编码器:',a)

        a=self.dll.GA_ZeroPos(1,1)
        print('清零轴1零位，返回值:',a)

        a=self.dll.GA_AxisOn(1)
        print('使能轴1返回值:',a)

        a=self.dll.GA_SetCrdPrmSingleEX(1,2,1,2,0,0,0,0,0,0,c_double(2000),c_double(5),0,1,0,0,0,0,0,0,0,0)
        print('建立2维坐标系，返回值:',a)
        dPrfPosx = c_double(0)
        dPrfPosy = c_double(0)
        speedx = 300
        speedy = 300
        a=self.dll.GA_InitLookAheadSingleEX(1,0,speedx,speedy,200,50,4000,4000,2,2,2,2,2,5,5,5,5,5,1,1,1,1,1)
        print('初始化前瞻，返回值:',a)

        a = self.dll.GA_CrdStart(1,0)
        print('启动坐标系运动,返回值:',a)

    # 接收并执行 x 和 y 的插补运动
    def execute_motion(self, xValue, yValue):
        print(f"执行插补运动: X={xValue}, Y={yValue}")
        a = self.dll.GA_LnXY(1, int(xValue), int(yValue), c_double(20.5), c_double(0.9), 0, 0, 2)
        # control the movement 

        a=self.dll.GA_CrdData(1,0,0)
        ######
        ######
        print('将数据压入控制卡')
        print(f'插入2维插补数据 X={xValue}, Y={yValue}, 返回值: {a}')
        a = self.dll.GA_GetPrfPos(1, byref(dPrfPosX),1,0)
        a = self.dll.GA_GetPrfPos(1, byref(dPrfPosY),1,0)
        dValuex = dPrfPosX
        dValuey = dPrfPosY
        print('获取轴1脉冲位置，返回值：',a,'获取值：',dValuex)
        print('获取轴2脉冲位置，返回值：',a,'获取值：',dValuey)

    # 控制端线程，接收 xValue 和 yValue 并控制机械轴
    def control_mechanical_axis(self):
        while True:
            # 等待从队列获取 xValue 和 yValue
            if not command_queue.empty():
                xValue, yValue = command_queue.get()
                self.execute_motion(int(xValue), int(yValue))
            time.sleep(0.01)  # 避免频繁查询队列消耗CPU

# 发送指令线程，模拟目标检测，定时传入 xValue 和 yValue
def send_motion_commands():
    xValues = [-10000, -5000, 10000, 5000]  # 示例 x 坐标列表
    yValues = [-10000, -5000, 10000, 5000]  # 示例 y 坐标列表
    while True:
        for xValue, yValue in zip(xValues, yValues):
            # 将新的指令放入队列
            command_queue.put((xValue, yValue))
            print(f"发送指令: X={xValue}, Y={yValue}")

            # 每次发送后等待2秒，模拟目标检测返回值的间隔
            time.sleep(2)
class RealTimeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Pose Detection")

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
        self.poseinit = [424,  233]
        self.frameCount = 0
        #### camera index
        self.index = 2


        self.mycalib = calib()
        self.mycalib.Test()
        self.worldInit = self.mycalib.projectPointNoUnistort(self.poseinit[0], self.poseinit[1])
        self.safeAnchor = np.array([[250, 100],
                                    [1000, 100],
                                    [1000, 900],
                                    [250, 900]], np.int32)
        # self.init_threshold()
   
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

    def judge_point(self, point):
        x_min = np.min(self.safeAnchor[:,0])
        y_min = np.min(self.safeAnchor[:,1])
        x_max = np.max(self.safeAnchor[:,0])
        y_max = np.max(self.safeAnchor[:,1])
        if point[0] > x_min and point[0] < x_max and point[1] > y_min and point[1] < y_max:
            return True
        else:
            return False 

    def update_frame(self):
        ret, frame = self.capture.read()
        flag = True
        TFGPUinference = True
        cfg = self.cfg
        usedTime = 0
        if ret and flag:
            self.frameCount += 1
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
            
            # 使用 cv2.pointPolygonTest 来检测点是否在多边形内部
            # result = cv2.pointPolygonTest(self.safeAnchor, midPose, False)
            # flag = True
            # if result >= 0:  # 点在多边形内或边上
            #     pass
            # else:
            #     flag = False
            #     print(f"Error: Point {midPose} is outside the safe anchor area!")
            #     # raise ValueError(f"Error: Point {midPose} is outside the safe anchor area!")

            flag = self.judge_point(midPose)
            if flag:  # 点在多边形内或边上
                pass
            else:
                print(f"Error: Point {midPose} is outside the safe anchor area!")
                # raise ValueError(f"Error: Point {midPose} is outside the safe anchor area!")
            
            # if abs((midPose[0] - self.poseinit[0])) > 50 or abs(midPose[1] - self.poseinit[1]) > 50 :
            if self.frameCount % 4 == 0 and flag:
                # world_coords1 = self.mycalib.projectPointNoUnistort(self.poseinit[0], self.poseinit[1])
                world_coords2 = self.mycalib.projectPointNoUnistort(midPose[0], midPose[1])
                # diff = wor
                vector = self.worldInit - world_coords2
                print(f"vector is {vector}")
                block = 12800 / 72
                distX = vector[0] * block
                distY = vector[1] * block
                print(f"distX is {distX} distY is {distY} ")
                self.send_xy(distX,distY)

                # self.moveDll.moveXY(int(distX), int(distY))
                
                # time.sleep(2)
                # self.poseinit = midPose



            # PredictedData[
            # self.counter, :
            # ] = (
            #     pose.flatten()
            # )  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
            # elif counter >= nframes:
            #     break
            # counter += 1

            # pbar.close()
        # elif ret and not flag:
        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        #     # Draw rectangles around faces
        #     for (x, y, w, h) in faces:
        #         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #     self.display_frame(frame)
        #     self.display_results(faces)


        else:
            self.stop_camera()

    def send_xy(self,xValue, yValue):
        command_queue.put((xValue, yValue))
        print(f"发送指令: X={xValue}, Y={yValue}")
        pass
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

    # 启动机械臂控制线程
    arm_controller = MechanicalAxisController()
    control_thread = threading.Thread(target=arm_controller.control_mechanical_axis)
    control_thread.daemon = True  # 设置为守护线程
    control_thread.start()
    
    sys.exit(app.exec_())

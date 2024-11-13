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
z_queue = Queue()

# 机械轴控制类
class MechanicalAxisController:
    def __init__(self):
        self.init_motion_system()

    # 初始化系统
    def init_motion_system(self):
        self.dll = CDLL(r"E:\yq\code\github\Live_Tracking\YQTest\GAS.dll")

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



        #############################################
        #############################################
        #############################################
       
        # init axis == 3
        axis = 3
        a=self.dll.GA_EncOff(axis)
        print(f'关闭轴{axis}编码器:',a)

        # a=self.dll.GA_ZeroPos(1,8)
        print('清零轴1零位，返回值:',a)

        a=self.dll.GA_AxisOn(axis)
        print('使能轴1返回值:',a)

        a=self.dll.GA_PrfTrap(axis)
        print('设置轴1进入点位模式，返回值:',a)

        a=self.dll.GA_SetTrapPrmSingle(axis,c_double(1.0),c_double(1.0),c_double(0.0),0)
        print('设置轴1点位运动参数，返回值:',a)


    # 接收并执行 x 和 y 的插补运动
    def execute_motion(self, xValue, yValue):

        # xy stop
        axis = 2
        lMask = (0x0001 << (axis - 1))  # 计算掩码值
        lOption = 0  # 根据函数要求设置此参数
        # 软件stop 
        a = self.dll.GA_Stop(lMask, lOption)


        axis = 1
        lMask = (0x0001 << (axis - 1))  # 计算掩码值
        a = self.dll.GA_Stop(lMask, lOption)

        ############
        ############


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

    # 接收并执行 z 的 运动
    def execute_z(self, zValue):
        axis = 3
        lMask = (0x0001 << (axis - 1))  # 计算掩码值
        lOption = 0  # 根据函数要求设置此参数
        # 软件stop 
        a = self.dll.GA_Stop(lMask, lOption)

        # flip z value
        zValue = - zValue

        print(f"执行插补运动: Z={zValue}")
        a=self.dll.GA_SetPos(axis,zValue)
        print(f'设置轴3运动目标位置为{zValue}脉冲的位置，返回值:',a)
        a=self.dll.GA_SetVel(axis,c_double(20))
        print('设置轴3运动速度为7.5脉冲/毫秒，返回值:',a)
        a=self.dll.GA_Update(2**(axis - 1))
        print('启动轴3运动')


    # 控制端线程，接收 xValue 和 yValue 并控制机械轴
    def control_mechanical_axis(self):
        while True:
            # 等待从队列获取 xValue 和 yValue
            if not command_queue.empty():
                xValue, yValue = command_queue.get()
                self.execute_motion(int(xValue), int(yValue))
            if not z_queue.empty():
                zValue = z_queue.get()
                self.execute_z(int(zValue))
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
        self.poseinit = [890,  434]
        self.poseinit2 = [1259,  453]
        self.frameCount = 0
        #### camera index

        # init camera vector
        self.initVec = [0,1]
        self.prevVec = [0,1]


        # data 置信度判断和像素距离判断
        self.confThreshold = 0.6
        self.distanceThreshold = 50

        # 初始化摄像头索引和状态
        self.camera1_index = 1  # 第一个摄像头的索引
        self.camera2_index = 2  # 第二个摄像头的索引
        self.active_camera = None
        self.capture = None

        # 初始化标志，用于指示是否切换到第二个摄像头
        self.use_camera2 = False

        # 初始一个累加器 记录每次检测之后的 axis3 小老鼠旋转的绝对位置
        self.pulseState = 0

        # self.index = 2

        # init camera 1
        self.init_cameraMatrix()        
        self.init_camera2_Matrix()
        self.worldInit = self.mycalib.projectPointNoUnistort(self.poseinit[0], self.poseinit[1])
        self.worldInit2 = self.camera2calib.projectPointNoUnistort(self.poseinit2[0], self.poseinit2[1])
        self.safeAnchor = np.array([[250, 100],
                                    [1000, 100],
                                    [1000, 900],
                                    [250, 900]], np.int32)
        self.safeAnchor = np.array([[0, 0],
                                    [2000, 0],
                                    [2000, 1900],
                                    [0, 1900]], np.int32)
        # self.init_threshold()
    def init_cameraMatrix(self):
        fx = 964.12
        fy = 969.67
        cx = 968.65
        cy = 557.79
        k1 = -0.17
        k2 = 0.49
        k3 = -0.56
        p1 = 0
        p2 = 0.0

        self.dist_coeffs = np.array([k1, k2, p1, p2, k3])  # 你的畸变系数
        self.mycalib = calib()
        self.mycalib.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])  # 你的相机内参矩阵
        self.mycalib.dist_coeffs = np.array([k1, k2, p1, p2, k3])  # 你的畸变系数

        
        # refine
        image_points = np.array([[1031, 334],
                    [1028, 99],
                    [1331, 93],
                    [1332, 330]
                    ] , dtype= np.float64)
        object_points = np.array([[0,0,0],
                       [250,0,0],
                       [250,320,0],
                       [0,320,0]],dtype=np.float64)
        # 使用PnP求解外参 (旋转向量rvec和平移向量tvec)
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                        self.mycalib.camera_matrix, self.mycalib.dist_coeffs)
        self.mycalib.rvec = rvec
        self.mycalib.tvec = tvec

        # self.mycalib.rvec = np.array ([[ 0.00616052],
        # [ 0.05821707],
        # [-1.58579181]])

        # self.mycalib.tvec = np.array([[  49.39697075],
        # [-239.42566489],
        # [1018.93725814]])

        self.mycalib.Test()
        # self.init_threshold()
    def init_camera2_Matrix(self):
        print(f"init_camera2_Matrix")
        fx = 995.62
        fy = 1006.27
        cx = 944.59
        cy = 586.57
        k1 = -0.06
        k2 = 0.15
        k3 = -0.11
        p1 = 0.01
        p2 = -0.01

        

        self.camera2calib = calib()
        self.camera2calib.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])  # 你的相机内参矩阵
        self.camera2calib.dist_coeffs = np.array([k1, k2, p1, p2, k3])  # 你的畸变系数

        # refine
        image_points = np.array([[1388, 356],
                    [1379, 130],
                    [1668, 126],
                    [1680, 350]
                    ] , dtype= np.float64)
        object_points = np.array([[0,0,0],
                       [250,0,0],
                       [250,320,0],
                       [0,320,0]],dtype=np.float64)
        # 使用PnP求解外参 (旋转向量rvec和平移向量tvec)
        retval, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                        self.camera2calib.camera_matrix, self.camera2calib.dist_coeffs)


        # self.camera2calib.rvec = np.array ([[-0.17508757],
        # [-0.1618264 ],
        # [-1.57177937]])
        # self.camera2calib.tvec = np.array([[ 399.2642267 ],
        # [ -24.76443787],
        # [1112.85500664]])
        self.camera2calib.rvec = rvec
        self.camera2calib.tvec = tvec

        self.camera2calib.Test()

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
        """启动第一个摄像头并开始追踪"""
        self.camera1 = cv2.VideoCapture(self.camera1_index)
        self.camera2 = cv2.VideoCapture(self.camera2_index)
        self.capture = self.camera1

        # 设置摄像头参数
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        # 设置摄像头参数
        self.camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camera2.set(cv2.CAP_PROP_FPS, 30)

        self.start_camera_button.setEnabled(False)
        self.stop_camera_button.setEnabled(True)
        self.use_camera2 = False  # 设置为使用第一个摄像头
        self.timer.start(20)


    def switch_to_camera2(self):
        """切换到第二个摄像头"""
        # if self.capture:
        #     self.capture.release()

        self.capture = self.camera2

        self.use_camera2 = True
        print("Switched to Camera 2")

    def switch_to_camera1(self):
        """切换到第一个摄像头"""
        # if self.capture:
        #     self.capture.release()

        self.capture = self.camera1

        self.use_camera2 = False
        print("Switched to Camera 1")




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
    def calculate_rotation_angle(self,v1, v2):
        # 将二维数组转换为一维数组，避免形状错误
        v1 = np.array(v1).flatten()  # 将 v1 变为 1D 数组
        v2 = np.array(v2).flatten()  # 将 v2 变为 1D 数组
        # 计算点积
        dot_product = np.dot(v1, v2)

        # 计算叉积的z分量，适用于二维向量 (A_x * B_y - A_y * B_x)
        cross_product_z = v1[0] * v2[1] - v1[1] * v2[0]

        # 计算旋转角度，atan2 会返回弧度，考虑点积和叉积的符号
        angle_radians = np.arctan2(cross_product_z, dot_product)

        # 将弧度转换为角度
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees
    
    def calculateMid(self, pose):

        # 置信度阈值
        confidence_threshold = self.confThreshold
        distance = self.distanceThreshold

        # 筛选有效点
        valid_points = pose[pose[:, 2] >= confidence_threshold]

        # 如果有有效点，则计算质心
        if valid_points.shape[0] > 0:
            # 计算有效点的质心
            centroid = np.mean(valid_points[:, :2], axis=0)
            print(f"质心坐标: {centroid}")

            print(f"质心坐标: {centroid}")

            # 计算每个有效点到质心的偏移量
            offsets = np.linalg.norm(pose[:, :2] - centroid, axis=1)
            # 统计偏移量大于 150 的数量
            count_large_offsets = np.sum(offsets > distance)


            # 输出每个点的偏移量
            for i, offset in enumerate(offsets):
                print(f"点 {i + 1} 到质心的偏移量: {offset}")
            return centroid, count_large_offsets
        else:
            print("没有找到有效的点，无法计算质心")
            return None, 5

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
            centroid, count_large_offsets = self.calculateMid(pose)

            if centroid is None or count_large_offsets > 4 and self.use_camera2 is not None:
                if self.use_camera2 is False:
                    self.switch_to_camera2()
                else:
                    self.switch_to_camera1()
                # return

                # return
            if count_large_offsets > 2:
                pass
                # return
            else:
                a = self.action(frame, centroid, pose, usedTime)

            
        else:
            pass
            # self.stop_camera()
    def action(self, frame, centroid, pose, usedTime):
        # center
        if centroid is None:
            pass
        else:
            vector = [frame.shape[0]//2 - centroid[0],frame.shape[1]//2 - centroid[1]]
            self.display_back2center(vector)
            print(f"vector is {vector}")

        
        print(f"used time is {usedTime}")
        print(f"center is {centroid}")

        # 获取箭头的起点和终点
        start_point = tuple(pose[3, 0:2].astype(int))  # 起点 (pose[0])
        end_point = tuple(pose[0, 0:2].astype(int))    # 终点 (pose[3])
        

        # 在图像上绘制箭头
        arrow_color = (0, 255, 0)  # 绿色箭头
        arrow_thickness = 2

        cv2.arrowedLine(frame, start_point,end_point, arrow_color, arrow_thickness, tipLength=0.3)
        self.counter += 1
        self.display_frame(frame)
        self.display_pose_results(pose)
        self.poseList.append(pose)
        self.display_usedTime(usedTime)

        # 创建保存图像的文件夹
        output_folder = "YQTest\Debug"
        os.makedirs(output_folder, exist_ok=True)
        # 保存帧到指定文件夹，文件名包含帧数
        frame_filename = os.path.join(output_folder, f"frame_{self.frameCount}.jpg")
        cv2.imwrite(frame_filename, frame)

        if pose is None or centroid is None:
            return None
        midPose = centroid
        ## z calculate the pos
        arrowVec = pose[0,0:2] - pose[3,0:2]
        angleRes = self.calculate_rotation_angle(self.initVec, arrowVec)
        print(f"angleRes is {angleRes}")

        nearRes = self.calculate_rotation_angle(self.prevVec, arrowVec)
        print(f"nearRes is {nearRes}")
        self.prevVec = arrowVec

        angleScale = 12800 * 5.02 * 2 / 360
        # ang2pulse = angleRes * angleScale
        ang2pulse = nearRes * angleScale
        self.pulseState += ang2pulse
        print(f"ang2pulse is {ang2pulse}")
        
        flag = self.judge_point(midPose)
        if flag:  # 点在多边形内或边上
            pass
        else:
            print(f"Error: Point {midPose} is outside the safe anchor area!")
            # raise ValueError(f"Error: Point {midPose} is outside the safe anchor area!")
        
        # if abs((midPose[0] - self.poseinit[0])) > 50 or abs(midPose[1] - self.poseinit[1]) > 50 :
        if self.frameCount % 1 == 0 and flag:
            #z move
            ############
            ############
            
            self.send_z(self.pulseState)



            #######################
            #######################
            # xy move
            # world_coords1 = self.mycalib.projectPointNoUnistort(self.poseinit[0], self.poseinit[1])
            if self.use_camera2:
                # replace
                world_coords2 = self.camera2calib.projectPointNoUnistort(midPose[0], midPose[1])
                vector = self.worldInit2 - world_coords2

                pass
            else:
                world_coords2 = self.mycalib.projectPointNoUnistort(midPose[0], midPose[1])
                vector = self.worldInit - world_coords2

            # diff = wor
            print(f"vector is {vector}")
            block = 12800 / 72
            distX = vector[0] * block
            distY = vector[1] * block
            print(f"distX is {distX} distY is {distY} ")
            self.send_xy(distX,distY)
    def send_xy(self,xValue, yValue):
        command_queue.put((xValue, yValue))
        print(f"发送指令: X={xValue}, Y={yValue}")
        pass
    def send_z(self,zValue):
        z_queue.put((zValue))
        print(f"发送指令: Z={zValue}")
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
        configPath =  r'E:\yq\code\DLC_Project\dlc-models\iteration-0\Mice1024Oct24-trainset95shuffle1\test\pose_cfg.yaml'
        # weightPath =  r"E:\yq\code\DataAndModel\\dlc-models\\iteration-0\\Test2Jul27-trainset95shuffle1\\train\\snapshot-100000"
        weightPath =  r"E:\yq\code\DLC_Project\dlc-models\iteration-0\Mice1024Oct24-trainset95shuffle1\train\snapshot-200000"
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

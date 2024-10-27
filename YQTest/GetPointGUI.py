import cv2
import numpy as np


import cv2

# 打开视频文件
video_path = r'D:\Backup\Documents\S-EYE Files\Images\20241027-215330-844.mp4'  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 提取第 N 帧 (可以根据需要修改)
frame_number = 24  # 你想提取的帧数（如第100帧）
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # 设置要读取的视频帧位置

# 读取该帧
ret, frame = cap.read()

if ret:
    # 显示提取的帧
    cv2.imshow('Extracted Frame', frame)
    
    # 保存该帧为图像
    save_path = 'extracted_frame2.jpg'
    cv2.imwrite(save_path, frame)
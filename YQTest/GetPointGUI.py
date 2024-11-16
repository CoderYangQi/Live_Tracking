import cv2
import time

# 初始化两个摄像头
camera_1 = cv2.VideoCapture(0)  # 摄像头1
camera_2 = cv2.VideoCapture(2)  # 摄像头2

# 设置摄像头分辨率（可选）
camera_1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera_2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not camera_1.isOpened() or not camera_2.isOpened():
    print("无法打开摄像头，请检查设备连接！")
    exit()

try:
    
    start = time.time()
    while True:
        # 读取摄像头1的帧
        ret1, frame1 = camera_1.read()
        # 读取摄像头2的帧
        ret2, frame2 = camera_2.read()

        if ret1 and ret2:
            # 显示摄像头1的帧
            cv2.imshow('Camera 1', frame1)
            # 显示摄像头2的帧
            cv2.imshow('Camera 2', frame2)

            # 保存帧
            cv2.imwrite('frame1.jpg', frame1)
            cv2.imwrite('frame2.jpg', frame2)

            # 等待1秒保存帧（或根据需求调整间隔时间）
            time.sleep(0.3)
        # end = time.time()
        # if 
        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("程序已中断。")

finally:
    # 释放资源
    camera_1.release()
    camera_2.release()
    cv2.destroyAllWindows()

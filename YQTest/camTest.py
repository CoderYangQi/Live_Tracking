import cv2

# 初始化一个列表，用于存储摄像头对象
cameras = []
index = 0  # 摄像头索引从0开始

# 尝试打开多个摄像头
while True:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        break
    cameras.append(cap)
    index += 1

# 如果没有检测到摄像头
if not cameras:
    print("未检测到任何摄像头！")
    exit()

print(f"检测到 {len(cameras)} 个摄像头。")

try:
    while True:
        # 读取所有摄像头的帧并显示
        for i, cap in enumerate(cameras):
            ret, frame = cap.read()
            if ret:
                # 显示每个摄像头的画面
                cv2.imshow(f'Camera {i}', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("程序已中断。")

finally:
    # 释放所有摄像头资源
    for cap in cameras:
        cap.release()
    cv2.destroyAllWindows()

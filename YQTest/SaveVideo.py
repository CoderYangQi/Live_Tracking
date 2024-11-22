import cv2
import threading
from datetime import datetime

class CameraRecorder(threading.Thread):
    """
    摄像头录像线程类
    """
    def __init__(self, camera_index, output_file, resolution=(1920, 1080), fps=30):
        super().__init__()
        self.camera_index = camera_index
        self.output_file = output_file
        self.resolution = resolution
        self.fps = fps
        self.running = True  # 控制线程运行的标志

    def run(self):
        """
        启动摄像头录像
        """
        # 初始化摄像头
        camera = cv2.VideoCapture(self.camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # 检查摄像头是否成功打开
        if not camera.isOpened():
            print(f"无法打开摄像头 {self.camera_index}")
            return

        # 初始化视频保存
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.output_file, fourcc, self.fps, self.resolution)

        print(f"开始录像: {self.output_file}")
        while self.running:
            ret, frame = camera.read()
            if ret:
                video_writer.write(frame)  # 保存帧
                cv2.imshow(f"Camera {self.camera_index}", frame)  # 显示画面

            # 按下 'q' 键停止录像
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        # 释放资源
        camera.release()
        video_writer.release()
        cv2.destroyAllWindows()
        print(f"录像结束: {self.output_file}")

    def stop(self):
        """
        停止录像线程
        """
        self.running = False


# 获取当前时间戳作为文件名前缀
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 创建录像线程
camera_1_recorder = CameraRecorder(0, f"{timestamp}_camera1.mp4")
camera_2_recorder = CameraRecorder(2, f"{timestamp}_camera2.mp4")

# 启动录像线程
camera_1_recorder.start()
camera_2_recorder.start()

try:
    # 等待线程完成
    camera_1_recorder.join()
    camera_2_recorder.join()
except KeyboardInterrupt:
    print("用户中断，停止录像...")
    camera_1_recorder.stop()
    camera_2_recorder.stop()

    # 等待线程安全退出
    camera_1_recorder.join()
    camera_2_recorder.join()

print("所有录像已结束，摄像头资源已释放。")

# F:\yq\code\Live_Tracking\YQTest\Debug

import cv2
import os
import glob

def images_to_video(image_folder, output_path, fps=30):
    """
    将图像序列合并为视频。
    
    :param image_folder: 包含 JPG 图像的文件夹路径
    :param output_path: 输出视频路径（例如 output.mp4）
    :param fps: 视频帧率，默认30
    """
    # 获取文件夹中按自然数排列的 JPG 文件
    # images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    images = glob.glob(os.path.join(image_folder, "*.jpg"))

    # 确保有图像
    if not images:
        print("未找到任何图像文件！")
        return

    # 读取第一张图像以确定帧大小
    first_image = cv2.imread(images[0])
    height, width, layers = first_image.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 将所有图像写入视频
    for image_path in images:
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.write(frame)

    # 释放资源
    video.release()
    print(f"视频已保存到: {output_path}")

# 示例用法
image_folder = r'F:\yq\code\Live_Tracking\YQTest\Debug'       # JPG 图像所在的文件夹路径
output_video = 'output.mp4'  # 输出视频路径
fps = 30                     # 设置帧率（可以根据需要调整）

images_to_video(image_folder, output_video, fps)

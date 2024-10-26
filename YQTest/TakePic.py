import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # 打开摄像头
        self.cap = cv2.VideoCapture(1)

        # 在Tkinter窗口中创建一个Label来显示摄像头的画面
        self.label = Label(window)
        self.label.pack()

        # 拍照按钮
        self.btn_snapshot = Button(window, text="拍照", width=20, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # 关闭按钮
        self.btn_exit = Button(window, text="退出", width=20, command=self.exit_app)
        self.btn_exit.pack(anchor=tk.CENTER, expand=True)

        # 更新视频流
        self.update()

        self.window.mainloop()

    def update(self):
        # 读取一帧图像
        ret, frame = self.cap.read()

        if ret:
            # 转换为 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 将图像转换为 PIL 格式，再转换为 ImageTk 格式
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # 更新Label上的图像
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        # 每10毫秒更新一次图像
        self.window.after(10, self.update)

    def snapshot(self):
        # 读取当前帧
        ret, frame = self.cap.read()

        if ret:
            # 保存图像
            cv2.imwrite("snapshot.png", frame)
            print("拍照成功，保存为 snapshot.png")

    def exit_app(self):
        # 释放摄像头并关闭窗口
        self.cap.release()
        self.window.quit()

# 创建Tkinter窗口并运行应用
root = tk.Tk()
app = CameraApp(root, "摄像头实时显示")

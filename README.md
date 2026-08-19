# Pose-Tracking  

## Detection Demo

[![MotionDetection demo preview](docs/media/motion_detection_demo.gif)](https://gitee.com/yqustc/MotionDetection/blob/master/docs/media/out_detect_new_720p.mp4)

Click the animated preview to view the full video. For the complete detection
implementation, documentation, and source code, visit the
[MotionDetection project](https://gitee.com/yqustc/MotionDetection).

> **Demo note:** The system uses a dual-camera setup. The software automatically
> displays the camera feed currently providing valid data, so the video may briefly
> jump when switching between camera views. This is expected camera-switching
> behavior rather than instability in the detection results.

## Usage
-- YQTest/PyQtMiceModel.py
change the model config path into your path.
```python
path_test_config = Path(
            r'D:\USERS\yq\code\MotionTracking\DeepLabCut\YQScripts\testdata\Test2-DLCTest2-2024-07-27\dlc-models\iteration-0\Test2Jul27-trainset95shuffle1\test\pose_cfg.yaml')
        self.dlc_cfg = load_config(str(path_test_config))
        self.dlc_cfg[
            "init_weights"] = "D:\\USERS\\yq\\code\\MotionTracking\\DeepLabCut\\YQScripts\\testdata\\Test2-DLCTest2-2024-07-27\\dlc-models\\iteration-0\\Test2Jul27-trainset95shuffle1\\train\\snapshot-100000"

```
### datasets
```angular2html
data:  https://rec.ustc.edu.cn/share/f5e65790-6206-11ef-8a13-9b0f90b6450e

```
## requirement
My env is TF2.10 & cuda
1. deeplabcut https://github.com/DeepLabCut/DeepLabCut/tree/main
2. dlc-live https://github.com/DeepLabCut/DeepLabCut-live
3. PyQt5

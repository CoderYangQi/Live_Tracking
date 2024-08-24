# Pose-Tracking  

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

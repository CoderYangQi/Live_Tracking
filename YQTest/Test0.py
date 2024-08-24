"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


import sys
import os
import shutil
import warnings

from yq_benchmark import benchmark_videos,benchmark,Refine_benchmark
import urllib.request
import argparse
from pathlib import Path
from dlclibrary.dlcmodelzoo.modelzoo_download import (
    download_huggingface_model,
)


# MODEL_NAME = "superanimal_quadruped"
# MODEL_NAME = "superanimal_quadruped_dlcrnet"
MODEL_NAME = "full_dog"
# SNAPSHOT_NAME = "snapshot-700000.pb"
SNAPSHOT_NAME = "snapshot-75000.pb"


def urllib_pbar(count, blockSize, totalSize):
    percent = int(count * blockSize * 100 / totalSize)
    outstr = f"{round(percent)}%"
    sys.stdout.write(outstr)
    sys.stdout.write("\b"*len(outstr))
    sys.stdout.flush()



if __name__ == "__main__":
    # main()
    display = True
    model_dir = r"D:\USERS\yq\code\MotionTracking\DLC_Live\DLC_Dog_resnet_50_iteration-0_shuffle-0"
    model_dir = Path(model_dir)
    video_file = r'D:\USERS\yq\code\MotionTracking\DLC_Live\check_install_dog_clip.avi'
    # benchmark_videos(str(model_dir), video_file, display=display, resize=0.5, pcutoff=0.25)
    model_path = str(model_dir)
    video_path = video_file
    tf_config = None;dynamic = (False, 0.5, 10)
    n_frames = 1000
    display_radius = 3
    this_inf_times, this_im_size, TFGPUinference, meta = Refine_benchmark(
        model_path,
        video_path,
        tf_config=tf_config,
        resize=0.5,
        pixels=None,
        cropping=None,
        dynamic=dynamic,
        n_frames=n_frames,
        print_rate=False,
        display=display,
        pcutoff=0.25,
        display_radius=display_radius,
        cmap='bmy',
        save_poses=False,
        save_video=False,
        output=None,
    )

    # inf_times.append(this_inf_times)
    # im_size_out.append(this_im_size)
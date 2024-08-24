import os.path

import cv2
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow.nnets.factory import PoseNetFactory
from deeplabcut.pose_estimation_tensorflow.config import load_config
from pathlib import Path
from skimage.util import img_as_ubyte
import numpy as np
from tqdm import tqdm
from deeplabcut.pose_estimation_tensorflow.core import predict

# from tensorflow.python.framework import graph_util
# from tensorflow.python.framework import graph_io

def setup_pose_prediction(cfg, allow_growth=False, collect_extra=False):
    tf.compat.v1.reset_default_graph()
    inputs = tf.compat.v1.placeholder(
        tf.float32, shape=[cfg["batch_size"], None, None, 3]
    )
    net_heads = PoseNetFactory.create(cfg).test(inputs)
    extra_dict = {}
    outputs = [net_heads["part_prob"]]
    if cfg["location_refinement"]:
        outputs.append(net_heads["locref"])

    if ("multi-animal" in cfg["dataset_type"]) and cfg["partaffinityfield_predict"]:
        print("Activating extracting of PAFs")
        outputs.append(net_heads["pairwise_pred"])

    outputs.append(net_heads["peak_inds"])

    if collect_extra:
        extra_dict["features"] = net_heads["features"]

    restorer = tf.compat.v1.train.Saver()

    if allow_growth:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
    else:
        sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg["init_weights"])

    if collect_extra:
        return sess, inputs, outputs, extra_dict
    else:
        return sess, inputs, outputs
def setup_GPUpose_prediction(cfg, allow_growth=False):
    tf.compat.v1.reset_default_graph()
    inputs = tf.compat.v1.placeholder(
        tf.float32, shape=[cfg["batch_size"], None, None, 3]
    )
    net_heads = PoseNetFactory.create(cfg).inference(inputs)
    outputs = [net_heads["pose"]]

    restorer = tf.compat.v1.train.Saver()

    if allow_growth:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
    else:
        sess = tf.compat.v1.Session()

    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg["init_weights"])



    return sess, inputs, outputs
def checkcropping(cfg, cap):
    print(
        "Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file."
        % (cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"])
    )
    nx = cfg["x2"] - cfg["x1"]
    ny = cfg["y2"] - cfg["y1"]
    if nx > 0 and ny > 0:
        pass
    else:
        raise Exception("Please check the order of cropping parameter!")
    if (
        cfg["x1"] >= 0
        and cfg["x2"] < int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 1)
        and cfg["y1"] >= 0
        and cfg["y2"] < int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 1)
    ):
        pass  # good cropping box
    else:
        raise Exception("Please check the boundary of cropping!")
    return int(ny), int(nx)
def GetPoseS_GTF(cfg, dlc_cfg, sess, inputs, outputs, cap, nframes):
    """Non batch wise pose estimation for video cap."""
    if cfg["cropping"]:
        ny, nx = checkcropping(cfg, cap)

    pose_tensor = predict.extract_GPUprediction(
        outputs, dlc_cfg
    )  # extract_output_tensor(outputs, dlc_cfg)
    PredictedData = np.zeros((nframes, 3 * len(dlc_cfg["all_joints_names"])))
    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))
    while cap.isOpened():
        if counter != 0 and counter % step == 0:
            pbar.update(step)

        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg["cropping"]:
                frame = img_as_ubyte(
                    frame[cfg["y1"] : cfg["y2"], cfg["x1"] : cfg["x2"]]
                )
            else:
                frame = img_as_ubyte(frame)

            pose = sess.run(
                pose_tensor,
                feed_dict={inputs: np.expand_dims(frame, axis=0).astype(float)},
            )
            pose[:, [0, 1, 2]] = pose[:, [1, 0, 2]]
            # pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
            PredictedData[
                counter, :
            ] = (
                pose.flatten()
            )  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
        elif counter >= nframes:
            break
        counter += 1

    pbar.close()
    return PredictedData, nframes
def converFunc():
    # Step 1: Load the checkpoint
    checkpoint_path = r'D:\USERS\yq\code\MotionTracking\DeepLabCut\YQScripts\testdata\Test2-DLCTest2-2024-07-27\dlc-models\iteration-0\Test2Jul27-trainset95shuffle1\train\snapshot-100000'
    # meta_path = r'D:\USERS\yq\code\MotionTracking\DeepLabCut\YQScripts\testdata\Test2-DLCTest2-2024-07-27\dlc-models\iteration-0\Test2Jul27-trainset95shuffle1\snapshot-100000.meta' # Your .meta file

    meta_path = checkpoint_path + '.meta'

    # Reset the default graph
    tf.compat.v1.reset_default_graph()

    # Start a session
    with tf.compat.v1.Session() as sess:
        # Restore the graph from the .meta file
        saver = tf.compat.v1.train.import_meta_graph(meta_path, clear_devices=True)

        # Restore the weights
        saver.restore(sess, checkpoint_path)

        # Get the graph definition
        graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        # Print all operation names to identify output nodes
        for op in tf.compat.v1.get_default_graph().get_operations():
            print(op.name)

        # Step 2: Freeze the graph
        # Convert variables to constants
        # output_node_names = ['output_node_name']  # Replace with the actual output node name(s)
        output_node_names = ['pose']  # Replace with the actual output node name(s)
        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            output_node_names
        )

        # Step 3: Save the frozen graph as a .pb file
        output_path = 'frozen_model.pb'
        with tf.io.gfile.GFile(output_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

    print(f"Model saved to {output_path}")


if __name__ == '__main__':
    # converFunc()
    path_test_config = Path(
        r'D:\USERS\yq\code\MotionTracking\DeepLabCut\YQScripts\testdata\Test2-DLCTest2-2024-07-27\dlc-models\iteration-0\Test2Jul27-trainset95shuffle1\test\pose_cfg.yaml')
    dlc_cfg = load_config(str(path_test_config))
    dlc_cfg["init_weights"] = "D:\\USERS\\yq\\code\\MotionTracking\\DeepLabCut\\YQScripts\\testdata\\Test2-DLCTest2-2024-07-27\\dlc-models\\iteration-0\\Test2Jul27-trainset95shuffle1\\train\\snapshot-100000"

    sess, inputs, outputs = setup_GPUpose_prediction(dlc_cfg, allow_growth=False)
    TFGPUinference = True
    cfg = {}
    cfg["cropping"] = False
    root = r"D:\USERS\yq\code\MotionTracking\DeepLabCut\YQScripts\testdata\Test20s-YQ-2024-07-28\videos"
    video = os.path.join(root, "test_20s.mp4")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(
            "Video could not be opened. Please check that the the file integrity."
        )
    # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = nframes * 1.0 / fps
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    if TFGPUinference:
        PredictedData, nframes = GetPoseS_GTF(
            cfg, dlc_cfg, sess, inputs, outputs, cap, nframes
        )
        print(PredictedData)
    else:
        pass
    # PredictedData, nframes = GetPoseS(
    #     cfg, dlc_cfg, sess, inputs, outputs, cap, nframes
    # )




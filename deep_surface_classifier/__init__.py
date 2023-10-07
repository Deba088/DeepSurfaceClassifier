import os
import logging
from typing import List

import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

from deep_surface_classifier.model.inception_v3 import inception_v3_model
from deep_surface_classifier.model.resnet_v2 import resnet_v2_model
from deep_surface_classifier.model.vgg_19 import vgg_19_model
from deep_surface_classifier.train_test_data import train_test_split
from deep_surface_classifier.utils import data_generator, preprocess_image
from deep_surface_classifier.eval_model import eval_model
from deep_surface_classifier.predict import predict_class

__all__ = ["train", "predict"]

tf.get_logger().setLevel(logging.ERROR)

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "train_1000")

# Get list of class names
# CLASS_NAMES = sorted(os.listdir(DATASET_DIR))
CLASS_NAMES = ["dry_asphalt_smooth", "dry_gravel", "fresh_snow", "ice", "wet_mud"]

# Log number of classes and class names
NUM_CLASSES = len(CLASS_NAMES)
logging.info(f"Number of classes: {NUM_CLASSES}")
logging.info(f"Class names: {CLASS_NAMES}")

# Define constants
BATCH_SIZE = 64
TEST_SIZE = 0.3
EPOCHS = 10


def train(model_name, dataset_dir=None, model_weights_dir=None, class_names=None,
        test_size=None, epochs=None, batch_size=None, num_layers_to_unfreeze=0, use_weight=False):
    if dataset_dir is None:
        dataset_dir = DATASET_DIR

    if class_names is None:
        class_names = CLASS_NAMES

    if test_size is None:
        test_size = TEST_SIZE

    if epochs is None:
        epochs = EPOCHS

    if batch_size is None:
        batch_size = BATCH_SIZE

    num_classes = len(class_names)

    train_data, test_data = train_test_split(dataset_dir, class_names, test_size)

    model, history, img_height, img_weight = (None, None, None, None)
    if model_name == "inception_v3":
        model, history, img_height, img_weight = inception_v3_model(train_data, test_data, model_weights_dir,
                                                 dataset_dir, batch_size, epochs, class_names, num_classes,
                                                 num_layers_to_unfreeze, use_weight)

    elif model_name == "resnet_v2":
        model, history, img_height, img_weight = resnet_v2_model(train_data, test_data, model_weights_dir,
                                                 dataset_dir, batch_size, epochs, class_names, num_classes,
                                                 num_layers_to_unfreeze, use_weight)

    elif model_name == "vgg_19":
        model, history, img_height, img_weight = vgg_19_model(train_data, test_data, model_weights_dir,
                                                 dataset_dir, batch_size, epochs, class_names, num_classes,
                                                 num_layers_to_unfreeze, use_weight)

    f1, precision, recall, conf_matrix, avg_time_taken_per_image = eval_model(model, test_data, dataset_dir,
                                                                   batch_size, img_height, img_weight, epochs,
                                                                   class_names, num_classes)

    return history, f1, precision, recall, conf_matrix, avg_time_taken_per_image


def predict(model_name, model_weights_dir, img_path: List):
    model_config = {
        "inception_v3": {
            "path": os.path.join(model_weights_dir, "inception_v3.h5"),
            "img_height": 299,
            "img_weight": 299
        },
        "resnet_v2": {
            "path": os.path.join(model_weights_dir, "resnet_v2.h5"),
            "img_height": 299,
            "img_weight": 299
        },
        "vgg_19": {
            "path": os.path.join(model_weights_dir, "vgg_19.h5"),
            "img_height": 224,
            "img_weight": 224
        }
    }

    model_path = model_config[model_name]["path"]
    img_height = model_config[model_name]["img_height"]
    img_weight = model_config[model_name]["img_weight"]

    frame_list = []
    for i in img_path:
        img = cv2.imread(i)
        frame = preprocess_image(img, img_height, img_weight)
        frame_list.append(frame)
    frame_list = np.array(frame_list)

    prediction = predict_class(model_path, frame_list, CLASS_NAMES)
    return prediction


def predict_video(model_name, model_weights_dir, input_video_path):
    model_config = {
        "inception_v3": {
            "path": os.path.join(model_weights_dir, "inception_v3.h5"),
            "img_height": 299,
            "img_weight": 299
        },
        "resnet_v2": {
            "path": os.path.join(model_weights_dir, "resnet_v2.h5"),
            "img_height": 299,
            "img_weight": 299
        },
        "vgg_19": {
            "path": os.path.join(model_weights_dir, "vgg_19.h5"),
            "img_height": 224,
            "img_weight": 224
        }
    }

    model_path = model_config[model_name]["path"]
    img_height = model_config[model_name]["img_height"]
    img_weight = model_config[model_name]["img_weight"]

    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    input_batch = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count/fps

        height, width, channels = frame.shape
        part_width = width//5
        part_height = height//3
        x1 = 2 * part_width
        y1 = 2 * part_height
        x2 = 3 * part_width
        y2 = 5 * part_height
        frame = frame[y1:y2, x1:x2]

        cv2.imwrite(f"C:/Users/ADMIN/Desktop/Project/DeepSurfaceClassifier/output/{frame_count}.jpg", frame)

        input_batch.append({"timestamp": timestamp, "frame": preprocess_image(frame, img_height, img_weight)})

        frame_skip_interval = 4
        for _ in range(frame_skip_interval - 1):
            cap.read()
            frame_count += 1

        frame_count += 1
    cap.release()

    df = pd.DataFrame(input_batch)
    df['frame'] = df['frame'].apply(lambda x: x.reshape((1, 299, 299, 3)))
    frame_list = np.concatenate(df['frame'].to_numpy(), axis=0)

    prediction = predict_class(model_path, frame_list, CLASS_NAMES)
    df["predicted_class"] = prediction
    df = df[["timestamp", "predicted_class"]]

    return df

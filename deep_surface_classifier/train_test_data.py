import os
import random

import numpy as np


def get_img_list(dataset_dir, class_names):
    # Store list of images as (class_name, image_name)
    img_list = []

    for class_name in class_names:
        img_dir = os.path.join(dataset_dir, class_name)
        for img in os.listdir(img_dir):
            img_list.append((img, class_name))

    img_list = np.array(img_list)

    # Shuffle the list of images
    np.random.shuffle(img_list)
    return img_list


def train_test_split(dataset_dir, class_names, test_size):
    img_list = get_img_list(dataset_dir, class_names)

    # Split into train and test sets
    train_data, test_data = [], []
    for i in img_list:
        if random.random() < test_size:
            test_data.append(i)
        else:
            train_data.append(i)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    train_data_length = len(train_data)
    test_data_length = len(test_data)

    print(f"No of total data: {len(img_list)}")
    print(f"No of train data: {train_data_length}")
    print(f"No of test data: {test_data_length}")

    return train_data, test_data

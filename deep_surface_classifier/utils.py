import os

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import tensorflow as tf


# Preprocess image
def preprocess_image(frame, img_height,img_weight):
    frame = cv2.resize(frame, (img_height, img_weight))
    frame = img_to_array(frame)
    frame = frame / 255.0
    return frame


# Load data into train and test data in the required format
def data_generator(data, dataset_dir, batch_size, img_height, img_weight, epochs, class_names, num_classes):
    num_batches = len(data) // batch_size

    for epoch in range(epochs):
        np.random.shuffle(data)

        # Loop through the num batches
        for batch_number in range(num_batches):
            # Initialize x_batch and y_batch
            x_batch, y_batch = [], []

            for each_data in data[batch_size * batch_number : batch_size * (batch_number + 1)]:
                file_path = os.path.join(os.path.join(dataset_dir, each_data[1]), each_data[0])

                # Read image file
                img = cv2.imread(file_path)

                # Preprocess image
                frame = preprocess_image(img, img_height,img_weight)

                x_batch.append(frame)
                y_batch.append(class_names.index(each_data[1]))

            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            y_batch = tf.constant(y_batch)
            y_batch = tf.one_hot(y_batch, depth=num_classes)

            yield x_batch, y_batch

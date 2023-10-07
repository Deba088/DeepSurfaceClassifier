from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from deep_surface_classifier.utils import data_generator


def eval_model(model, test_data, dataset_dir, batch_size, img_height, img_weight, epochs, class_names, num_classes):
    # Evaluate the model on the test dataset
    # Reset the test generator to start from the beginning
    test_generator = data_generator(test_data, dataset_dir, batch_size, img_height, img_weight,
                                    epochs, class_names, num_classes)
    y_true = []
    y_pred = []

    start_time = datetime.now()
    for _ in range(len(test_data) // batch_size):
        X_batch, y_batch = next(test_generator)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(model.predict(X_batch), axis=1))
    end_time = datetime.now()

    time_taken = (end_time - start_time).total_seconds()
    avg_time_taken_per_image = time_taken / ((len(test_data) // batch_size) * batch_size)

    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    return f1, precision, recall, conf_matrix, avg_time_taken_per_image

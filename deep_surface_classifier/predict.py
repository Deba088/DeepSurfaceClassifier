from datetime import datetime

from tensorflow import keras
import numpy as np


def predict_class(model_path, frame, class_names):
    model = keras.models.load_model(model_path)

    start_time = datetime.now()
    prediction = model.predict(frame)
    end_time = datetime.now()

    print(f"Time taken: {end_time-start_time}")
    print(prediction)
    prediction = np.argmax(prediction, axis=1)
    print(prediction)

    predicted_class = []
    for j in prediction:
        predicted_class.append(class_names[j])

    return predicted_class


# def predict_class_from_video(model_path, frame, class_names):
#     model = keras.models.load_model(model_path)



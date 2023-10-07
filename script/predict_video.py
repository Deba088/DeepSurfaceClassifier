import os

import deep_surface_classifier as dsc

root_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_videos")
input_video_data = "road_driving.MP4"
input_data_path = os.path.join(root_path, input_video_data)

model_weights_dir = "C:/Users/ADMIN/Desktop/Project/DeepSurfaceClassifier/model_weights"

predicted_class = dsc.predict_video("inception_v3", model_weights_dir, input_data_path)
print(predicted_class)

predicted_class.to_csv("C:/Users/ADMIN/Desktop/Project/DeepSurfaceClassifier/output/data.csv", header=True, index=False)

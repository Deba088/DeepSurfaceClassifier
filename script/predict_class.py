import os

import deep_surface_classifier as dsc

root_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_1000")
input_data = [
    "dry_asphalt_severe/202201270007448-dry-asphalt-severe.jpg",
    "dry_asphalt_severe/202201271605545-dry-asphalt-severe.jpg",
    "dry_gravel/202202112236512-dry-gravel.jpg"
]

input_data_path = []

for i in input_data:
    input_data_path.append(os.path.join(root_path, i))

model_weights_dir = "C:/Users/ADMIN/Desktop/Project/DeepSurfaceClassifier/model_weights"

predicted_class = dsc.predict("resnet_v2", model_weights_dir, input_data_path)
print(predicted_class)

import numpy as np
import cv2
import json
import onnxruntime as rt
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# Load the labels from the labels_map.txt
with open("labels_map.txt", "r") as f:
    labels_map = json.load(f)

# Set the image file dimensions
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    img -= [127.0, 127.0, 127.0]  # Normalize to [-1, 1]
    img /= [128.0, 128.0, 128.0]
    return img

def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

# Initialize COCO API for image annotations
coco = COCO("annotations/instances_val2017.json")  # Point to your coco annotations file
image_ids = coco.getImgIds()

# Load the ONNX model using onnxruntime
onnx_session = rt.InferenceSession("your_model.onnx")

# Function to get Top 1% prediction
def get_top_1_percent(predictions, top_k):
    top_indices = np.argsort(predictions)[::-1][:top_k]
    return top_indices

# Initialize a list to store all predictions
all_predictions = []

# Process entire COCO dataset
top_k = max(1, len(labels_map) // 100)  # Top 1% (10 classes for 1000 classes)
for img_id in image_ids:
    # Load image
    img_info = coco.loadImgs([img_id])[0]
    img_file = img_info['file_name']
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocess image
    img_processed = pre_process_edgetpu(img, (224, 224, 3))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_processed, axis=0)

    # Run inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: img_batch})[0]
    
    # Get top 1% predictions for this image
    top_indices = get_top_1_percent(outputs[0], top_k)

    # Store predictions in the all_predictions list
    for idx in top_indices:
        class_name = labels_map[str(idx)]  # Use labels_map for mapping index to class name
        all_predictions.append({
            "image": img_file,
            "class": class_name,
            "probability": outputs[0][idx]
        })

# Sort all predictions by probability in descending order
all_predictions_sorted = sorted(all_predictions, key=lambda x: x['probability'], reverse=True)

# Calculate the top 1% across the entire dataset
total_predictions = len(all_predictions_sorted)
top_1_percent_count = int(total_predictions * 0.01)  # 1% of total predictions

# Print the top 1% predictions
print("\nTop 1% Predictions for Entire Dataset:")
for i in range(top_1_percent_count):
    prediction = all_predictions_sorted[i]
    print(f"Rank {i+1}: Image: {prediction['image']}, Class: {prediction['class']}, Probability: {prediction['probability']:.5f}")

# Optionally: Save the sorted predictions to a file
with open("top_1_percent_predictions.json", "w") as f:
    json.dump(all_predictions_sorted[:top_1_percent_count], f, indent=4)

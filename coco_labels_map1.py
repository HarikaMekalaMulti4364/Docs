import numpy as np
import json
import matplotlib.pyplot as plt
import onnxruntime as rt
import cv2
import os
from pathlib import Path

# Load the labels from the labels_map.txt file
def load_labels(label_map_file):
    with open(label_map_file, "r") as file:
        return json.load(file)

labels = load_labels("labels_map.txt")

# Set image file dimensions to 224x224 by resizing and cropping image from center
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # Converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

# Resize the image with a proportional scale
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

# Crop the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

# Load and preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = pre_process_edgetpu(img, (224, 224, 3))
    return img

# Load model
def load_model(model_path):
    return rt.InferenceSession(model_path)

# Run inference on the model
def predict(model, img_batch):
    results = model.run(["Softmax:0"], {"images:0": img_batch})[0]
    return results

# Display top-k predictions
def print_top_k(results, k=5):
    result = reversed(results[0].argsort()[-k:])
    for r in result:
        print(f"Index: {r}, Label: {labels[str(r)]}, Confidence: {results[0][r]}")

# Calculate Top-1% accuracy on the dataset
def calculate_top_1_percent_accuracy(dataset_path, model):
    correct_top_1 = 0
    total_images = 0

    for img_file in os.listdir(dataset_path):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(dataset_path, img_file)
            img = preprocess_image(img_path)
            img_batch = np.expand_dims(img, axis=0)

            # Run inference
            results = predict(model, img_batch)

            # Get top-1 predicted index
            top_1_idx = results[0].argmax()
            top_1_label = labels[str(top_1_idx)]

            # Get true label (assuming true label is in the filename for simplicity)
            true_label = img_file.split("_")[0]  # Adjust based on your dataset structure

            if top_1_label == true_label:
                correct_top_1 += 1

            total_images += 1

    top_1_percent_accuracy = (correct_top_1 / total_images) * 100
    print(f"Top-1% Accuracy: {top_1_percent_accuracy:.2f}%")

# Main execution
def main():
    model_path = "efficientnet-lite4.onnx"  # Path to the ONNX model
    dataset_path = "path/to/coco/val/images"  # Path to COCO validation images

    # Load the model
    model = load_model(model_path)

    # Calculate top-1% accuracy
    calculate_top_1_percent_accuracy(dataset_path, model)

# Run the main function
if __name__ == "__main__":
    main()

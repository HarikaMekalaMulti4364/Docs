import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import onnxruntime as ort

# Paths
workdir = "workdir"
coco_gt_path = os.path.join(workdir, "annotations/instances_val2017.json")
coco_images_path = os.path.join(workdir, "val2017")

# Load COCO ground truth
gt_coco = COCO(coco_gt_path)

# ONNX model
onnx_model_path = "model.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_shape = (1, 3, 640, 640)
input_name = session.get_inputs()[0].name

# Prepare output file
results = []

# Inference for first image only
first_img_id = gt_coco.getImgIds()[0]
img_info = gt_coco.loadImgs(first_img_id)[0]
img_path = os.path.join(coco_images_path, img_info["file_name"])

# Load and preprocess image
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (640, 640))
img_input = img_resized.transpose(2, 0, 1).astype(np.int8)
img_input = np.expand_dims(img_input, axis=0)

# Run inference
outputs = session.run(None, {input_name: img_input})
pred_boxes, pred_classes, pred_scores = outputs  # Shapes: (1, 25200, 4), (1, 25200), (1, 25200)

# Dump first image predictions
print(f"First image predictions:\nBoxes: {pred_boxes[0][:5]}\nClasses: {pred_classes[0][:5]}\nScores: {pred_scores[0][:5]}")

# Convert to COCO format
for i in range(25200):
    results.append({
        "image_id": first_img_id,
        "category_id": int(pred_classes[0, i]) + 1,  # COCO classes start from 1
        "bbox": [float(x) for x in pred_boxes[0, i]],
        "score": float(pred_scores[0, i])
    })

# Save results
first_img_predictions_path = os.path.join(workdir, "first_image_predictions.json")
with open(first_img_predictions_path, "w") as f:
    json.dump(results, f)

# Draw and save predictions on the image
for i in range(10):  # Draw first 10 detections
    box = pred_boxes[0][i]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    score = pred_scores[0][i]
    class_id = int(pred_classes[0][i]) + 1
    
    if score > 0.3:  # Only draw high-confidence detections
        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_resized, f"{class_id}:{score:.2f}", (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

pred_image_path = os.path.join(workdir, "first_image_predictions.jpg")
cv2.imwrite(pred_image_path, img_resized)

# COCO Evaluation
dt_coco_first = gt_coco.loadRes(first_img_predictions_path)
coco_eval_first = COCOeval(gt_coco, dt_coco_first, "bbox")
coco_eval_first.params.imgIds = [first_img_id]
coco_eval_first.evaluate()
coco_eval_first.accumulate()
coco_eval_first.summarize()

print("mAP for first image:")
for i, metric in enumerate(["AP@0.5:0.95", "AP@0.5", "AP@0.75", "AP@small", "AP@medium", "AP@large"]):
    print(f"{metric}: {coco_eval_first.stats[i]:.4f}")

print(f"Predictions saved to {pred_image_path}")

import os
import json
import cv2
import numpy as np
import torch
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

# Get first image
img_id = gt_coco.getImgIds()[0]
img_info = gt_coco.loadImgs(img_id)[0]
img_path = os.path.join(coco_images_path, img_info["file_name"])

# Load and preprocess image
img = cv2.imread(img_path)
h, w, _ = img.shape
img_resized = cv2.resize(img, (640, 640)).astype(np.float32)
img_resized = img_resized / 255.0  # Normalize if required
img_input = img_resized.transpose(2, 0, 1).astype(np.int8)
img_input = np.expand_dims(img_input, axis=0)

# Run inference
outputs = session.run(None, {input_name: img_input})
pred_boxes, pred_classes, pred_scores = outputs  # Shapes: (1, 25200, 4), (1, 25200), (1, 25200)

# Convert int8 output back to float if required
pred_boxes = pred_boxes.astype(np.float32)
pred_classes = pred_classes.astype(np.int32)
pred_scores = pred_scores.astype(np.float32)

# Rescale boxes to original image size
scale_x = w / 640.0
scale_y = h / 640.0
pred_boxes[0][:, [0, 2]] *= scale_x
pred_boxes[0][:, [1, 3]] *= scale_y

# Clip negative values
pred_boxes[0] = np.clip(pred_boxes[0], 0, [w, h, w, h])

# Draw predictions
for i in range(25200):
    if pred_scores[0, i] > 0.3:  # Confidence threshold
        x1, y1, x2, y2 = pred_boxes[0][i]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{int(pred_classes[0, i])}: {pred_scores[0, i]:.2f}"
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save and display result
output_img_path = os.path.join(workdir, "result.jpg")
cv2.imwrite(output_img_path, img)
print(f"Result saved to {output_img_path}")

# Convert to COCO format
results = []
for i in range(25200):
    if pred_scores[0, i] > 0.3:
        results.append({
            "image_id": img_id,
            "category_id": int(pred_classes[0, i]) + 1,  # COCO classes start from 1
            "bbox": [float(x) for x in pred_boxes[0, i]],
            "score": float(pred_scores[0, i])
        })

# Save results
coco_results_path = os.path.join(workdir, "coco_results.json")
with open(coco_results_path, "w") as f:
    json.dump(results, f)

# COCO Evaluation
dt_coco = gt_coco.loadRes(coco_results_path)
coco_eval = COCOeval(gt_coco, dt_coco, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print("Overall mAP:")
for i, metric in enumerate(["AP@0.5:0.95", "AP@0.5", "AP@0.75", "AP@small", "AP@medium", "AP@large"]):
    print(f"{metric}: {coco_eval.stats[i]:.4f}")

print(f"Predictions saved to {coco_results_path}")

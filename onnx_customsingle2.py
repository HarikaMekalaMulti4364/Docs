import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Paths
workdir = "workdir"
coco_gt_path = os.path.join(workdir, "annotations/instances_val2017.json")
coco_images_path = os.path.join(workdir, "val2017")

# Load COCO ground truth
gt_coco = COCO(coco_gt_path)

# Get first image
img_id = gt_coco.getImgIds()[0]
img_info = gt_coco.loadImgs(img_id)[0]
img_path = os.path.join(coco_images_path, img_info["file_name"])

# Load image
img = cv2.imread(img_path)
h, w, _ = img.shape  # Original image size

# Load precomputed predictions
pred_boxes = np.load("pred_boxes.npy").astype(np.float32)  # (1, 25200, 4)
pred_classes = np.load("pred_classes.npy").astype(np.int32)  # (1, 25200)
pred_scores = np.load("pred_scores.npy").astype(np.float32)  # (1, 25200)

# Remove batch dimension
pred_boxes = pred_boxes[0]  # (25200, 4)
pred_classes = pred_classes[0]  # (25200,)
pred_scores = pred_scores[0]  # (25200,)

# Apply confidence threshold
conf_threshold = 0.3
valid_indices = pred_scores > conf_threshold

pred_boxes = pred_boxes[valid_indices]
pred_classes = pred_classes[valid_indices]
pred_scores = pred_scores[valid_indices]

# Draw predictions
for i in range(len(pred_scores)):
    x1, y1, x2, y2 = map(int, pred_boxes[i])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{int(pred_classes[i])}: {pred_scores[i]:.2f}"
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save and display result
output_img_path = os.path.join(workdir, "result.jpg")
cv2.imwrite(output_img_path, img)
print(f"Result saved to {output_img_path}")

# Convert to COCO format
results = [
    {
        "image_id": img_id,
        "category_id": int(pred_classes[i]) + 1,  # COCO classes start from 1
        "bbox": [float(x) for x in pred_boxes[i]],
        "score": float(pred_scores[i])
    }
    for i in range(len(pred_scores))
]

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

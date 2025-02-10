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

# Inference loop
for img_id in tqdm(gt_coco.getImgIds()):
    img_info = gt_coco.loadImgs(img_id)[0]
    img_path = os.path.join(coco_images_path, img_info["file_name"])
    
    # Load and preprocess image
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.transpose(2, 0, 1) / 255.0
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_name: img_input})
    pred_boxes, pred_classes, pred_scores = outputs  # Shapes: (1, 25200, 4), (1, 25200), (1, 25200)
    
    # Convert to COCO format
    for i in range(25200):
        if pred_scores[0, i] > 0.05:  # Confidence threshold
            x, y, w, h = pred_boxes[0, i]
            results.append({
                "image_id": img_id,
                "category_id": int(pred_classes[0, i]) + 1,  # COCO classes start from 1
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(pred_scores[0, i])
            })
    
    # Visualization for a single image
    if img_id == gt_coco.getImgIds()[0]:
        vis_img = img_resized.copy()
        for i in range(25200):
            if pred_scores[0, i] > 0.3:
                x, y, w, h = map(int, pred_boxes[0, i])
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(workdir, "sample_prediction.jpg"), vis_img)

# Save results
predictions_path = os.path.join(workdir, "predictions.json")
with open(predictions_path, "w") as f:
    json.dump(results, f)

# COCO Evaluation
dt_coco = gt_coco.loadRes(predictions_path)
coco_eval = COCOeval(gt_coco, dt_coco, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print("mAP Results:")
for i, metric in enumerate(["AP@0.5:0.95", "AP@0.5", "AP@0.75", "AP@small", "AP@medium", "AP@large"]):
    print(f"{metric}: {coco_eval.stats[i]:.4f}")

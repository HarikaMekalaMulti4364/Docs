import os
import json
import cv2
import numpy as np
import torch
import torchvision.ops as ops
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import onnxruntime as ort
import matplotlib.pyplot as plt

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
img_resized = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0  # Normalize
img_input = img_resized.transpose(2, 0, 1).astype(np.float32)
img_input = np.expand_dims(img_input, axis=0)

# Run inference
outputs = session.run(None, {input_name: img_input})
pred_boxes, pred_classes, pred_scores = outputs  # Shapes: (1, 25200, 4), (1, 25200), (1, 25200)

# Convert function
def convert(z):
    z = torch.cat(z, 1)
    box = z[:, :, :4]
    conf = z[:, :, 4:5]
    score = z[:, :, 5:]
    score *= conf
    convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                   dtype=torch.float32, device=z.device)
    box @= convert_matrix  
    return box, score

# Convert and rescale boxes
box, score = convert([torch.tensor(pred_boxes), torch.tensor(pred_scores)])
def rescale_boxes(box, img_w, img_h, input_size=640):
    scale_x = img_w / input_size
    scale_y = img_h / input_size
    box[:, :, [0, 2]] *= scale_x
    box[:, :, [1, 3]] *= scale_y
    return box
box = rescale_boxes(box, w, h)

# Apply Non-Maximum Suppression (NMS)
def nms(boxes, scores, iou_threshold=0.5):
    keep = ops.nms(boxes[0], scores[0].max(dim=-1)[0], iou_threshold)
    return boxes[:, keep], scores[:, keep]
box, score = nms(box, score)

# Convert to COCO format
results = []
for i in range(box.shape[1]):
    for j in range(box.shape[2]):
        if score[0, i, j] > 0.3:
            results.append({
                "image_id": img_id,
                "category_id": j + 1,
                "bbox": [float(x) for x in box[0, i]],
                "score": float(score[0, i, j])
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

# Output mAP scores
metrics = ["AP@0.5:0.95", "AP@0.5", "AP@0.75", "AP@small", "AP@medium", "AP@large"]
print("Overall mAP:")
for i, metric in enumerate(metrics):
    print(f"{metric}: {coco_eval.stats[i]:.4f}")

print(f"Predictions saved to {coco_results_path}")

# Plot detected boxes
def plot_boxes(image, boxes):
    for box in boxes[0]:
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

plot_boxes(img, box)

import torch
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from pathlib import Path
from collections import defaultdict

def test(model, dataloader, device):
    model.eval()
    jdict = []
    with torch.no_grad():
        for images, image_ids in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                image_id = image_ids[i].item()
                
                # Removing batch dimension
                boxes = output["boxes"].squeeze(0).cpu().numpy().tolist()
                scores = output["scores"].squeeze(0).cpu().numpy().tolist()
                classes = output["labels"].squeeze(0).cpu().numpy().tolist()
                
                jdict.append({
                    "image_id": image_id,
                    "boxes": boxes,
                    "scores": scores,
                    "classes": classes
                })
    
    return jdict

def compute_mAP(predictions, ground_truths, iou_thresholds=[0.5]):
    """
    Computes mean Average Precision (mAP) for object detection.
    Args:
    - predictions (list of dict): List of predicted results, each containing
      "image_id", "boxes", "scores", and "classes".
    - ground_truths (dict): COCO-style ground truth annotations with keys like "images", "annotations", etc.
    - iou_thresholds (list): List of IoU thresholds to calculate mAP at different levels.
    Returns:
    - mAP: Mean Average Precision at the given IoU thresholds.
    """
    coco_pred = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": str(i)} for i in range(1, 91)]  # assuming 90 categories for COCO
    }

    coco_gt = COCO(ground_truths)
    
    for pred in predictions:
        image_id = pred["image_id"]
        boxes = pred["boxes"]
        scores = pred["scores"]
        classes = pred["classes"]
        
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            coco_pred["annotations"].append({
                "image_id": image_id,
                "category_id": cls,
                "bbox": box,
                "score": score
            })
    
    # Convert predictions to COCO format
    coco_pred["images"] = [{"id": img_id} for img_id in coco_pred["annotations"]]
    
    # Evaluate predictions using pycocotools
    coco_dt = coco_gt.loadRes(coco_pred["annotations"])  # Load prediction results
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")  # Evaluating bounding boxes
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats[0]  # This is the mAP at IoU = 0.5

# Example usage:

# Load model and data loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("path_to_trained_model.pth")
model.to(device)

# Assuming dataloader is defined elsewhere
predictions = test(model, dataloader, device)

# Load ground truth annotations (COCO format)
ground_truth_path = "path_to_ground_truth_annotations.json"
with open(ground_truth_path, 'r') as f:
    ground_truths = json.load(f)

# Calculate mAP
mAP = compute_mAP(predictions, ground_truths, iou_thresholds=[0.5])
print(f"mAP: {mAP}")

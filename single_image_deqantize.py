import onnxruntime as ort
import numpy as np
import cv2
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

# ----------------- 1. Load ONNX Model -----------------
onnx_model_path = "your_model.onnx"
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# ----------------- 2. Load COCO Dataset -----------------
coco_annotation_path = "instances_val2017.json"  # Update path to your COCO annotations
coco = COCO(coco_annotation_path)

# Get the first image
first_img_id = coco.getImgIds()[0]  # Get first image ID
first_img_info = coco.loadImgs([first_img_id])[0]
image_path = os.path.join("val2017", first_img_info["file_name"])  # Update val2017 path

# ----------------- 3. Preprocessing Function -----------------
def preprocess_image(image_path, input_size=640):
    """
    Loads an image, resizes to 640x640, and normalizes it for inference.
    """
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]

    # Resize to (640,640)
    img_resized = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # Convert to int8 and transpose to (1,3,640,640)
    img_resized = img_resized.astype(np.int8)
    img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
    img_resized = np.expand_dims(img_resized, axis=0)   # Add batch dim (1,3,640,640)

    return img_resized, orig_w, orig_h, img

# ----------------- 4. Convert xywh to xyxy Format -----------------
def xywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

# ----------------- 5. Rescale Boxes to Original Image -----------------
def rescale_boxes(boxes, orig_w, orig_h, target_size=640):
    scale_w = orig_w / target_size
    scale_h = orig_h / target_size
    boxes[:, [0, 2]] *= scale_w  # Scale x-coordinates
    boxes[:, [1, 3]] *= scale_h  # Scale y-coordinates
    return boxes

# ----------------- 6. Run ONNX Inference -----------------
def run_inference(image_path):
    img, orig_w, orig_h, orig_img = preprocess_image(image_path)

    inputs = {session.get_inputs()[0].name: img}
    pred_boxes, pred_scores, pred_labels = session.run(None, inputs)

    pred_boxes = pred_boxes.reshape(-1, 4)   # (25200, 4)
    pred_scores = pred_scores.flatten()      # (25200,)
    pred_labels = pred_labels.flatten()      # (25200,)

    pred_boxes = xywh_to_xyxy(pred_boxes)
    pred_boxes = rescale_boxes(pred_boxes, orig_w, orig_h)

    return pred_boxes, pred_scores, pred_labels, orig_img

# ----------------- 7. Apply Non-Maximum Suppression (NMS) -----------------
def apply_nms(boxes, scores, labels, iou_threshold=0.5, score_threshold=0.3):
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold, iou_threshold)
    if len(keep) == 0:
        return np.array([]), np.array([]), np.array([])
    
    keep = keep.flatten()
    return boxes[keep], scores[keep], labels[keep]

# ----------------- 8. Convert Predictions to COCO Format -----------------
def convert_to_coco_format(boxes, scores, labels, img_id):
    results = []
    for i in range(len(boxes)):
        results.append({
            "image_id": img_id,
            "category_id": int(labels[i]),
            "bbox": [float(boxes[i][0]), float(boxes[i][1]), 
                     float(boxes[i][2] - boxes[i][0]), float(boxes[i][3] - boxes[i][1])],
            "score": float(scores[i])
        })
    return results

# ----------------- 9. Compute mAP using COCO API -----------------
def compute_map(ground_truth_json, prediction_json):
    coco_gt = COCO(ground_truth_json)
    coco_dt = coco_gt.loadRes(prediction_json)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

# ----------------- 10. Draw Bounding Boxes on Image -----------------
def draw_predictions(image, boxes, scores, labels, conf_threshold=0.3):
    for i in range(len(boxes)):
        if scores[i] < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, boxes[i])
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{int(labels[i])}: {scores[i]:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# ----------------- 11. Get Ground Truth for the Image -----------------
def get_ground_truth(image_id):
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
    gt_boxes = []
    gt_labels = []
    
    for ann in annotations:
        x, y, w, h = ann['bbox']
        gt_boxes.append([x, y, x + w, y + h])
        gt_labels.append(ann['category_id'])

    return np.array(gt_boxes), np.array(gt_labels)

# ----------------- 12. Main Execution -----------------
if __name__ == "__main__":
    pred_boxes, pred_scores, pred_labels, orig_img = run_inference(image_path)

    conf_threshold = 0.3
    mask = pred_scores > conf_threshold
    pred_boxes, pred_scores, pred_labels = pred_boxes[mask], pred_scores[mask], pred_labels[mask]

    pred_boxes, pred_scores, pred_labels = apply_nms(pred_boxes, pred_scores, pred_labels)

    results = convert_to_coco_format(pred_boxes, pred_scores, pred_labels, img_id=first_img_id)

    with open("predictions.json", "w") as f:
        json.dump(results, f)

    compute_map(coco_annotation_path, "predictions.json")

    output_img = draw_predictions(orig_img, pred_boxes, pred_scores, pred_labels)
    cv2.imwrite("output.jpg", output_img)

    cv2.imshow("Detections", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

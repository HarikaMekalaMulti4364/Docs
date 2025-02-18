import numpy as np
import cv2

def apply_score_and_iou_threshold(boxes, scores, class_ids, score_threshold, iou_threshold):
    """
    Apply score threshold and IOU threshold using Non-Maximum Suppression (NMS).
    
    :param boxes: Array of bounding boxes with shape (N, 4), where N is the number of detections.
    :param scores: Array of detection scores with shape (N,)
    :param class_ids: Array of class IDs with shape (N,)
    :param score_threshold: Minimum score for a detection to be kept.
    :param iou_threshold: IOU threshold for Non-Maximum Suppression.
    
    :return: Filtered boxes, scores, and class_ids after applying thresholds and NMS.
    """
    # Apply score threshold
    valid_indices = np.where(scores >= score_threshold)[0]
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]

    # Apply Non-Maximum Suppression (NMS)
    # NMS needs boxes in (x1, y1, x2, y2) format and scores
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold, iou_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()  # Flatten the result of NMS
        return boxes[indices], scores[indices], class_ids[indices]
    else:
        return np.array([]), np.array([]), np.array([])

# Example Usage:
boxes = np.array([[50, 30, 200, 150], [55, 35, 205, 155], [100, 80, 250, 200]])
scores = np.array([0.9, 0.85, 0.95])
class_ids = np.array([1, 1, 2])

score_threshold = 0.8
iou_threshold = 0.5

filtered_boxes, filtered_scores, filtered_class_ids = apply_score_and_iou_threshold(boxes, scores, class_ids, score_threshold, iou_threshold)

print("Filtered boxes:", filtered_boxes)
print("Filtered scores:", filtered_scores)
print("Filtered class_ids:", filtered_class_ids)

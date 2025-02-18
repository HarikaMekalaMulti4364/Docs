import torch
import torchvision.ops as ops

def apply_score_and_iou_threshold(boxes, scores, score_threshold, iou_threshold):
    """
    Apply score threshold and IOU threshold using Non-Maximum Suppression (NMS) from torchvision.

    :param boxes: Tensor of bounding boxes with shape (N, 4), where N is the number of detections.
    :param scores: Tensor of detection scores with shape (N,)
    :param score_threshold: Minimum score for a detection to be kept.
    :param iou_threshold: IOU threshold for Non-Maximum Suppression.

    :return: Filtered boxes and scores after applying thresholds and NMS.
    """
    # Apply score threshold
    valid_indices = scores >= score_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]

    # Apply Non-Maximum Suppression (NMS)
    keep_indices = ops.nms(boxes, scores, iou_threshold)

    # Filter boxes and scores using the indices from NMS
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]

    return boxes, scores

# Example Usage:
boxes = torch.tensor([[50, 30, 200, 150], [55, 35, 205, 155], [100, 80, 250, 200]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.85, 0.95], dtype=torch.float32)

score_threshold = 0.8
iou_threshold = 0.5

filtered_boxes, filtered_scores = apply_score_and_iou_threshold(boxes, scores, score_threshold, iou_threshold)

print("Filtered boxes:", filtered_boxes)
print("Filtered scores:", filtered_scores)

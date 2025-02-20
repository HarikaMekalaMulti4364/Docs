# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Collection

import torch
import cv2
import numpy as np

# podm comes from the object-detection-metrics pip package
from podm.metrics import (  # type: ignore
    BoundingBox,
    MetricPerClass,
    get_pascal_voc_metrics,
)

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.utils.bounding_box_processing import batched_nms


class DetectionEvaluator(BaseEvaluator):
    """Evaluator for comparing a batched image output."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
    ):
        self.reset()
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.image_height = image_height
        self.image_width = image_width
        self.scale_x = 1 / image_width
        self.scale_y = 1 / image_height

    def add_batch(self, output: Collection[torch.Tensor], gt: Collection[torch.Tensor], image: np.ndarray, save_path: str = "output_predictions.jpg"):
        """
        Adds a batch of predictions and ground truths for evaluation.

        Args:
            output: Tuple of predicted bounding boxes, scores, and class indices.
            gt: Tuple of ground truth values including bounding boxes and labels.
            image: The original image for visualization.
            save_path: Path to save the annotated image.
        """
        image_ids, _, _, all_bboxes, all_classes, all_num_boxes = gt
        pred_boxes, pred_scores, pred_class_idx = output

        for i in range(len(image_ids)):
            image_id = image_ids[i]
            bboxes = all_bboxes[i][: all_num_boxes[i].item()]
            classes = all_classes[i][: all_num_boxes[i].item()]
            if bboxes.numel() == 0:
                continue

            # Reuse NMS utility
            (
                after_nms_pred_boxes,
                after_nms_pred_scores,
                after_nms_pred_class_idx,
            ) = batched_nms(
                self.nms_iou_threshold,
                self.nms_score_threshold,
                pred_boxes[i : i + 1],
                pred_scores[i : i + 1],
                pred_class_idx[i : i + 1],
            )

            # Collect GT and prediction boxes
            gt_bb_entry = [
                BoundingBox.of_bbox(
                    image_id, cat, bbox[0], bbox[1], bbox[2], bbox[3], 1.0
                )
                for cat, bbox in zip(classes.tolist(), bboxes.tolist())
            ]

            pd_bb_entry = [
                BoundingBox.of_bbox(
                    image_id,
                    pred_cat,
                    pred_bbox[0] * self.scale_x,
                    pred_bbox[1] * self.scale_y,
                    pred_bbox[2] * self.scale_x,
                    pred_bbox[3] * self.scale_y,
                    pred_score,
                )
                for pred_cat, pred_score, pred_bbox in zip(
                    after_nms_pred_class_idx[0].tolist(),
                    after_nms_pred_scores[0].tolist(),
                    after_nms_pred_boxes[0].tolist(),
                )
            ]

            # Compute mean average precision
            self._update_mAP(gt_bb_entry, pd_bb_entry)

            # Visualize and save predictions
            self.plot_predictions(image, bboxes.tolist(), after_nms_pred_boxes[0].tolist(), 
                                  classes.tolist(), after_nms_pred_class_idx[0].tolist(), 
                                  after_nms_pred_scores[0].tolist(), save_path)

    def reset(self):
        self.gt_bb = []
        self.pd_bb = []
        self.results = {}

    def _update_mAP(self, gt_bb_entry, pd_bb_entry):
        self.gt_bb += gt_bb_entry
        self.pd_bb += pd_bb_entry

        self.results = get_pascal_voc_metrics(
            self.gt_bb, self.pd_bb, self.nms_iou_threshold
        )
        self.mAP = MetricPerClass.mAP(self.results)

    def get_accuracy_score(self):
        return self.mAP

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.3f} mAP"

    def plot_predictions(self, image, gt_boxes, pd_boxes, gt_labels, pd_labels, pd_scores, save_path):
        """
        Plots ground truth and predicted bounding boxes on the image.

        Args:
            image (np.ndarray): The original image.
            gt_boxes (list): List of ground truth boxes in (x, y, w, h) format.
            pd_boxes (list): List of predicted boxes in (x, y, w, h) format.
            gt_labels (list): List of ground truth class labels.
            pd_labels (list): List of predicted class labels.
            pd_scores (list): List of confidence scores for predicted boxes.
            save_path (str): Path to save the image with drawn bounding boxes.
        """
        img = image.copy()

        # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
        def convert_bbox_format(boxes, img_h, img_w):
            return [
                (int(box[0] * img_w), int(box[1] * img_h), 
                 int((box[0] + box[2]) * img_w), int((box[1] + box[3]) * img_h))
                for box in boxes
            ]

        img_h, img_w = img.shape[:2]
        gt_boxes = convert_bbox_format(gt_boxes, img_h, img_w)
        pd_boxes = convert_bbox_format(pd_boxes, img_h, img_w)

        # Draw Ground Truth Boxes (Green)
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT: {label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

        # Draw Predicted Boxes (Red)
        for box, label, score in zip(pd_boxes, pd_labels, pd_scores):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{label}: {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2)

        # Show and save the image
        cv2.imshow("Predictions", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(save_path, img)

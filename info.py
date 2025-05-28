coco_predictions = coco_ground_truth.loadRes(predictions)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  pycocotools/coco.py", line 329, in loadRes
    if 'caption' in anns[0]:
                    ~~~~^^^
IndexError: list index out of range



postprocess without letterbox

def unletterbox_boxes(boxes, input_shape, original_shape):
    """
    Reverses letterboxing to map boxes from letterboxed image space to original image space.
    `boxes`: Tensor[N, 4] in xyxy format
    `input_shape`: (h, w) after letterboxing, e.g., (640, 640)
    `original_shape`: (h, w) of original image before resize
    """
    ih, iw = input_shape
    oh, ow = original_shape

    # Determine scale and padding
    scale = min(iw / ow, ih / oh)
    pad_w = (iw - ow * scale) / 2
    pad_h = (ih - oh * scale) / 2

    # Adjust boxes
    boxes = boxes.clone()
    boxes[:, [0, 2]] -= pad_w  # x padding
    boxes[:, [1, 3]] -= pad_h  # y padding
    boxes /= scale

    # Clip to original image dimensions
    boxes[:, 0].clamp_(0, ow)
    boxes[:, 1].clamp_(0, oh)
    boxes[:, 2].clamp_(0, ow)
    boxes[:, 3].clamp_(0, oh)
    return boxes


if out:
    bboxes, scores, labels = out
    bboxes = unletterbox_boxes(bboxes, (640, 640), input_shape)  # <-- fix here



















preprocess with letterbox
# Copyright 2023 Synopsys, Inc.
# This Synopsys software and all associated documentation are proprietary
# to Synopsys, Inc. and may only be used pursuant to the terms and conditions
# of a written license agreement with Synopsys, Inc.
# All other use, reproduction, modification, or distribution of the Synopsys
# software or the associated documentation is strictly prohibited.

from pathlib import Path
import numpy as np
import cv2
import math


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


def load_image(img_path, img_size):
    """
    Loads an image by index, returning the image, its original dimensions, and resized dimensions.

    Returns (im, original hw, resized hw)
    """
    im = cv2.imread(str(img_path))
    h0, w0 = im.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_LINEAR
        im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
    return im, (h0, w0)


def GetDataLoader(dataset, max_input=None, **kwargs):
    extensions = [".jpg", ".jpeg", ".png"]
    path_to_all_files = sorted(Path(dataset).glob("*"))
    paths_to_images = [image_path for image_path in path_to_all_files if image_path.suffix.lower() in extensions]
    for image_path in paths_to_images[:max_input]:
        img, (h0, w0) = load_image(image_path, 640)
        img = letterbox(img, auto=False, scaleup=False)
        img = preprocess(img)
        yield img, [h0, w0]


def preprocess(image):
    preprocessed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed_img = preprocessed_img[np.newaxis, ...].astype(np.float32) / 255.
    preprocessed_img = preprocessed_img.transpose(0, 3, 1, 2)
    return preprocessed_img




preprocess without letterbox
# Copyright 2023 Synopsys, Inc.
# This Synopsys software and all associated documentation are proprietary
# to Synopsys, Inc. and may only be used pursuant to the terms and conditions
# of a written license agreement with Synopsys, Inc.
# All other use, reproduction, modification, or distribution of the Synopsys
# software or the associated documentation is strictly prohibited.

from pathlib import Path
import cv2

def GetDataLoader(dataset, max_input=None, **kwargs):
    extensions = [".jpg", ".jpeg", ".png"]
    path_to_all_files = sorted(Path(dataset).glob("*"))
    paths_to_images = [image_path for image_path in path_to_all_files if image_path.suffix.lower() in extensions]
    for image_path in paths_to_images[:max_input]:
        img = cv2.imread(str(image_path))
        # img = cv2.cvtColor(img)
        img = preprocess(img)
       
        yield img#.astype(float32)


def preprocess(image):

    # Need further adjustment for quantization and accuracy test

    h, w, c = image.shape

    h_in = 640
    w_in = 640

    image = cv2.resize(image, [h_in, w_in], interpolation = cv2.INTER_LINEAR)
    img_data = image.transpose(2, 0, 1)

    img_data = img_data.reshape(1, 3, h_in, w_in)
    
    img_data = img_data.astype('float32')
    
    return img_data




postprocess without letterbox
# Copyright 2023 Synopsys, Inc.
# This Synopsys software and all associated documentation are proprietary
# to Synopsys, Inc. and may only be used pursuant to the terms and conditions
# of a written license agreement with Synopsys, Inc.
# All other use, reproduction, modification, or distribution of the Synopsys
# software or the associated documentation is strictly prohibited.

import numpy as np
import torch
from typing import List, Dict
from mmcv.ops.nms import batched_nms
from nnac.accuracy.datasets.coco import coco_ids


def bbox_cxcywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    "Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)."
    cx, cy, w, h = np.split(bbox, 4, axis=-1)
    cx, cy, w, h = cx.squeeze(-1), cy.squeeze(-1), w.squeeze(-1), h.squeeze(-1)
    bbox_new = np.stack([(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)], axis=-1)
    return bbox_new

def meshgrid(x, y, row_major):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx

def single_level_grid_priors(featmap_size, level_idx, dtype, device, with_stride):
    feat_h, feat_w = featmap_size
    strides = [(8, 8), (16, 16), (32, 32)]
    offset = 0
    stride_w, stride_h = strides[level_idx]
    shift_x = (torch.arange(0, feat_w, device=device) +
                offset) * stride_w
    # keep featmap_size as Tensor instead of int, so that we
    # can convert to ONNX correctly
    shift_x = shift_x.to(dtype)

    shift_y = (torch.arange(0, feat_h, device=device) +
                offset) * stride_h
    # keep featmap_size as Tensor instead of int, so that we
    # can convert to ONNX correctly
    shift_y = shift_y.to(dtype)
    shift_xx, shift_yy = meshgrid(shift_x, shift_y, True)
    if not with_stride:
        shifts = torch.stack([shift_xx, shift_yy], dim=-1)
    else:
        # use `shape[0]` instead of `len(shift_xx)` for ONNX export
        stride_w = shift_xx.new_full((shift_xx.shape[0], ),
                                        stride_w).to(dtype)
        stride_h = shift_xx.new_full((shift_yy.shape[0], ),
                                        stride_h).to(dtype)
        shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h],
                                dim=-1)
    all_points = shifts.to(device)
    return all_points

def grid_priors(featmap_sizes, dtype, device, with_stride):
    num_levels = 3
    multi_level_priors = []
    for i in range(num_levels):
        priors = single_level_grid_priors(
            featmap_sizes[i],
            level_idx=i,
            dtype=dtype,
            device=device,
            with_stride=with_stride)
        multi_level_priors.append(priors)
    return multi_level_priors

def bbox_decode(priors, bbox_preds):
    xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
    whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

    tl_x = (xys[..., 0] - whs[..., 0] / 2)
    tl_y = (xys[..., 1] - whs[..., 1] / 2)
    br_x = (xys[..., 0] + whs[..., 0] / 2)
    br_y = (xys[..., 1] + whs[..., 1] / 2)

    decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    return decoded_bboxes

def bbox_post_process(bboxes, scores, labels):
    if bboxes.numel() > 0:
        nms_cfg = {'type': 'nms', 'iou_threshold': 0.65}
        det_bboxes, keep_idxs = batched_nms(bboxes, scores, labels, nms_cfg)
        bboxes, scores, labels = bboxes[keep_idxs], scores[keep_idxs], labels[keep_idxs]
        scores = det_bboxes[:, -1]
        return bboxes, scores, labels


IM_HEIGHT = 640
IM_WIDTH = 640
def postprocess(nn_output: List, input_shape) -> List[Dict]:
    """Post-processing for DETR_mmlab
    :param nn_output: output data from NN
    :type nn_output: list[numpy_array]
    :return: list of COCO-dictionaries
    :rtype: list[dict()]
    """
    # print("\n *************************",nn_output[1].shape)
    # print("\n *************************",nn_output[2].shape)
    # print("\n *************************",nn_output[3])
    # print("\n *************************",nn_output[4])
    # print("\n *************************",nn_output[5])
    # print("\n *************************",nn_output[6])
    # exit()
    # o1 = torch.from_numpy(nn_output[0][np.newaxis, ...]).reshape(1, -1, 80)
    # o2 = torch.from_numpy(nn_output[1][np.newaxis, ...]).reshape(1, -1, 80)
    # o3 = torch.from_numpy(nn_output[2][np.newaxis, ...]).reshape(1, -1, 80)
    # # concat4 = torch.concat(o1, o2, o3)
    # print(o1.shape)
    # print(o2.shape)
    # print(o3.shape)
    # exit()
    # print(type(torch.from_numpy(nn_output[0][np.newaxis, ...].reshape(1, -1, 80)))
    strides = (8,16,32)
    offset=0
    cls_scores = []
    cls_scores.append(torch.from_numpy(nn_output[2][np.newaxis, ...].transpose(0,3,1,2)))
    cls_scores.append(torch.from_numpy(nn_output[1][np.newaxis, ...].transpose(0,3,1,2)))
    cls_scores.append(torch.from_numpy(nn_output[0][np.newaxis, ...].transpose(0,3,1,2)))
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
    mlvl_priors =grid_priors(
        featmap_sizes,
        dtype=cls_scores[0].dtype,
        device=cls_scores[0].device,
        with_stride=True)
    # print(mlvl_priors)
    bbox_preds = []
    bbox_preds.append(torch.from_numpy(nn_output[5][np.newaxis, ...].transpose(0,3,1,2)))
    bbox_preds.append(torch.from_numpy(nn_output[4][np.newaxis, ...].transpose(0,3,1,2)))
    bbox_preds.append(torch.from_numpy(nn_output[3][np.newaxis, ...].transpose(0,3,1,2)))
    objectnesses = []
    objectnesses.append(torch.from_numpy(nn_output[8][np.newaxis, ...].transpose(0,3,1,2)))
    objectnesses.append(torch.from_numpy(nn_output[7][np.newaxis, ...].transpose(0,3,1,2)))
    objectnesses.append(torch.from_numpy(nn_output[6][np.newaxis, ...].transpose(0,3,1,2)))
    flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(1, -1,
                                                  80)
            for cls_score in cls_scores
        ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(1, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_objectness = [
        objectness.permute(0, 2, 3, 1).reshape(1, -1)
        for objectness in objectnesses
    ]

    flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    # print(flatten_bbox_preds)
    flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
    flatten_priors = torch.cat(mlvl_priors)
    # print(flatten_priors.shape)
    # exit()
    flatten_bboxes = bbox_decode(flatten_priors, flatten_bbox_preds)
    max_scores, labels = torch.max(flatten_cls_scores[0], 1)
    valid_mask = flatten_objectness[0] * max_scores >= 0.01
    bboxes = flatten_bboxes[0][valid_mask]
    scores = max_scores[valid_mask] * flatten_objectness[0][valid_mask]
    labels = labels[valid_mask]
    out = bbox_post_process(bboxes, scores, labels)
    results=[]
    for score, bbox, label in zip(scores, bboxes, labels):
        xmin, ymin, xmax, ymax = bbox
        entry = {
            "category_id": coco_ids[int(label)],
            "score": float(score),
            "bbox": [xmin/640 , ymin/640 , (xmax - xmin)/640 , (ymax - ymin)/640 ],
        }
        results.append(entry)
    return results
    # exit()

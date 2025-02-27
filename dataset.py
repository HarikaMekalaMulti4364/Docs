import os
import json
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

def _download_data(self) -> None:
    # Define dataset path
    dataset_dir = os.path.join(LOCAL_STORE_DEFAULT_PATH, "coco-2017")

    # Ensure dataset exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"COCO dataset not found in {dataset_dir}. Please download it manually."
        )

    # Define train/val split
    split_str = "val2017" if self.split == DatasetSplit.VAL else "train2017"
    ann_file = os.path.join(dataset_dir, "annotations", f"instances_{split_str}.json")
    image_dir = os.path.join(dataset_dir, split_str)

    # Load dataset directly using CocoDetection
    self.dataset = CocoDetection(root=image_dir, annFile=ann_file)

    # Sorting the dataset by image path for deterministic ordering
    self.dataset.samples = sorted(self.dataset.samples, key=lambda x: x[0])



import os
from torchvision.datasets import CocoDetection

def _download_data(self) -> None:
    # Define dataset path
    dataset_dir = os.path.join(LOCAL_STORE_DEFAULT_PATH, "coco-2017")

    # Ensure dataset exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"COCO dataset not found in {dataset_dir}. Please download it manually."
        )

    # Define train/val split
    split_str = "val2017" if self.split == DatasetSplit.VAL else "train2017"
    ann_file = os.path.join(dataset_dir, "annotations", f"instances_{split_str}.json")
    image_dir = os.path.join(dataset_dir, split_str)

    # Load dataset using CocoDetection
    self.dataset = CocoDetection(root=image_dir, annFile=ann_file)

    # Extract image file paths
    img_id_to_path = {img_id: os.path.join(image_dir, img_info["file_name"]) 
                      for img_id, img_info in self.dataset.coco.imgs.items()}

    # Sort dataset based on image file paths
    self.dataset = sorted(self.dataset, key=lambda x: img_id_to_path[x[1][0]['image_id']])



import torch
import torchvision.transforms.functional as F
from PIL import Image
from pathlib import Path

def __getitem__(self, item):
    """
    Returns a tuple of input image tensor and label data.

    Label data is a tuple with the following entries:
      - Image ID within the original dataset
      - height (in pixels)
      - width (in pixels)
      - bounding box data with shape (self.max_boxes, 4)
        - The 4 should be normalized (x, y, w, h)
      - labels with shape (self.max_boxes,)
      - number of actual boxes present
    """

    image, annotations = self.dataset[item]  # Load image and annotations from CocoDetection
    image = image.convert("RGB")  # Ensure RGB format
    width, height = image.size

    boxes = []
    labels = []

    for ann in annotations:
        x, y, w, h = ann["bbox"]  # COCO format: (x, y, width, height)
        category_id = ann["category_id"]

        # Normalize bounding box coordinates
        boxes.append([x / width, y / height, (x + w) / width, (y + h) / height])
        labels.append(category_id)

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Pad boxes and labels to max_boxes
    num_boxes = len(labels)
    if num_boxes == 0:
        boxes = torch.zeros((self.max_boxes, 4), dtype=torch.float32)
        labels = torch.zeros(self.max_boxes, dtype=torch.long)
    elif num_boxes > self.max_boxes:
        raise ValueError(
            f"Sample has more boxes than max_boxes {self.max_boxes}. "
            "Re-initialize the dataset with a larger value for max_boxes."
        )
    else:
        boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - num_boxes), value=0)
        labels = F.pad(labels, (0, self.max_boxes - num_boxes), value=0)

    # Resize image
    image = image.resize(self.target_image_size)
    image = app_to_net_image_inputs(image)[1].squeeze(0)

    return image, (
        int(Path(self.dataset.coco.imgs[annotations[0]["image_id"]]["file_name"]).stem),
        height,
        width,
        boxes,
        labels,
        torch.tensor([num_boxes]),
    )




import torch
import torchvision.transforms.functional as F
from PIL import Image
from pathlib import Path

def __getitem__(self, item):
    """
    Returns a tuple of input image tensor and label data.

    Label data is a tuple with the following entries:
      - Image ID within the original dataset (from filename)
      - height (in pixels)
      - width (in pixels)
      - bounding box data with shape (self.max_boxes, 4)
        - The 4 should be normalized (x, y, w, h)
      - labels with shape (self.max_boxes,)
      - number of actual boxes present
    """

    image, annotations = self.dataset[item]  # Load image and annotations
    image = image.convert("RGB")  # Ensure RGB format
    width, height = image.size

    boxes = []
    labels = []

    for ann in annotations:
        x, y, w, h = ann["bbox"]  # COCO format: (x, y, width, height)
        category_id = ann["category_id"]

        # Normalize bounding box coordinates
        boxes.append([x / width, y / height, (x + w) / width, (y + h) / height])
        labels.append(category_id)

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Pad boxes and labels to max_boxes
    num_boxes = len(labels)
    if num_boxes == 0:
        boxes = torch.zeros((self.max_boxes, 4), dtype=torch.float32)
        labels = torch.zeros(self.max_boxes, dtype=torch.long)
    elif num_boxes > self.max_boxes:
        raise ValueError(
            f"Sample has more boxes than max_boxes {self.max_boxes}. "
            "Re-initialize the dataset with a larger value for max_boxes."
        )
    else:
        boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - num_boxes), value=0)
        labels = F.pad(labels, (0, self.max_boxes - num_boxes), value=0)

    # Resize image
    image = image.resize(self.target_image_size)
    image = app_to_net_image_inputs(image)[1].squeeze(0)

    # Extract image ID from filepath
    image_path = Path(self.dataset.coco.imgs[item]["file_name"])
    image_id = int(image_path.stem) if image_path.stem.isdigit() else item

    return image, (
        image_id,
        height,
        width,
        boxes,
        labels,
        torch.tensor([num_boxes]),
    )

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








Previously, FiftyOne was used to manage dataset loading, which automatically handled dataset indexing, sample ordering, and annotations. Now, the dataset is directly loaded using torchvision.datasets.CocoDetection, which does not internally manage ID-to-index mappings in the same way.







import fiftyone as fo

setup_fiftyone_env()

split_str = "validation" if self.split == DatasetSplit.VAL else "train"

# Load dataset from local directory instead of FiftyOne zoo
self.dataset = fo.Dataset.from_dir(
    dataset_dir=self.dataset_dir,
    dataset_type=fo.types.COCODetectionDataset,
    labels_path=f"{self.dataset_dir}/annotations/instances_{split_str}2017.json"
)

# Shuffle the dataset if needed
if self.shuffle:
    self.dataset.shuffle()

# Limit to max_samples if specified
if self.num_samples:
    self.dataset = self.dataset.limit(self.num_samples)

# Sort by filepath to ensure deterministic order
self.dataset = self.dataset.sort_by("filepath")



 images_dir = os.path.join(self.dataset_dir, split_str)
    annotations_path = os.path.join(self.dataset_dir, "annotations", f"instances_{split_str}.json")

    # Load COCO annotations
    self.coco = COCO(annotations_path)
    self.image_ids = list(self.coco.imgs.keys())
    
    # Store image file paths in a list (sorted for consistency)
    self.dataset = sorted([
        os.path.join(images_dir, self.coco.imgs[img_id]["file_name"])
        for img_id in self.image_ids
    ])




# Get the image ID for COCO
image_id = self.coco.getImgIds()[item]  # Assuming 'item' is the dataset index

# Get annotations for this image
annotation_ids = self.coco.getAnnIds(imgIds=image_id)
annotations = self.coco.loadAnns(annotation_ids)

boxes = []
labels = []

if annotations:  # Ensure there are annotations
    for annotation in annotations:
        category_id = annotation["category_id"]

        # Convert COCO category_id to self.label_map index
        if category_id not in self.label_map.values():
            print(f"Warning: Invalid label {category_id}")
            continue

        # COCO format: (x, y, width, height)
        x, y, w, h = annotation["bbox"]
        boxes.append([x, y, x + w, y + h])

        # Map category_id to label index
        label_index = list(self.label_map.values()).index(category_id)
        labels.append(label_index)

# Convert to PyTorch tensors
boxes = torch.tensor(boxes, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.int64)

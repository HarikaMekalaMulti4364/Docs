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

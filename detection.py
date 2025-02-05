import torch
import json
import time
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Assuming you have a model called 'model' and data loader 'dataloader'

def time_synchronized():
    # This is a simple function to sync and time the operations.
    return time.time()

def evaluate(dataloader, model, save_json=True, save_path="results.json", is_coco=True, coco91class=None, ground_truth_annotations="annotations.json"):
    model.eval()  # Set model to evaluation mode
    jdict = []  # Initialize the list for storing predictions
    
    t1 = time_synchronized()  # Start the timer

    # Iterate over the data
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        # Forward pass through the model
        out = model(img)  # out is a tuple of (boxes, scores, classes)

        boxes = out[0]  # Shape: (1, 25200, 4)
        scores = out[1]  # Shape: (1, 25200)
        classes = out[2]  # Shape: (1, 25200)

        # Loop over the batch for predictions
        for si, pred in enumerate(zip(boxes, scores, classes)):  # Iterate over the batch
            pred_boxes, pred_scores, pred_classes = pred
            image_id = int(paths[si].stem) if paths[si].stem.isnumeric() else paths[si].stem

            for i in range(len(pred_boxes)):
                bbox = pred_boxes[i].tolist()  # Get bounding box coordinates
                score = pred_scores[i].item()  # Get score
                class_id = int(pred_classes[i].item())  # Get class label

                # Append to the json dict
                jdict.append({
                    'image_id': image_id,
                    'category_id': coco91class[class_id] if is_coco else class_id,
                    'bbox': [round(x, 3) for x in bbox],
                    'score': round(score, 5)
                })

        # Time per image
        t1 += time_synchronized() - t

    # Save predictions to a JSON file
    if save_json:
        with open(save_path, 'w') as f:
            json.dump(jdict, f)

    # COCO evaluation
    # Load the ground truth annotations (assuming you're working with COCO format)
    coco_gt = COCO(ground_truth_annotations)  # Path to the ground truth annotations file
    coco_dt = coco_gt.loadRes(save_path)  # Load predictions (detected results)

    # Initialize COCOeval for evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')  # Use 'bbox' for bounding box evaluation

    # Run the evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()  # This will print the evaluation results, including metrics like AP

    # Optionally, save the results to a file if needed
    eval_results = {
        'AP': coco_eval.stats[0],  # AP (Average Precision)
        'AP50': coco_eval.stats[1],  # AP at IoU=0.5
        'AP75': coco_eval.stats[2],  # AP at IoU=0.75
        'AP_small': coco_eval.stats[3],  # AP for small objects
        'AP_medium': coco_eval.stats[4],  # AP for medium objects
        'AP_large': coco_eval.stats[5],  # AP for large objects
        'AR': coco_eval.stats[6],  # AR (Average Recall)
        'AR50': coco_eval.stats[7],  # AR at IoU=0.5
        'AR75': coco_eval.stats[8],  # AR at IoU=0.75
    }

    # You can save these evaluation metrics as well if you want
    with open('evaluation_results.json', 'w') as eval_file:
        json.dump(eval_results, eval_file)

# Example usage
if __name__ == "__main__":
    # Assuming coco91class is a dictionary or list mapping class indexes to class IDs
    coco91class = [i for i in range(91)]  # Replace with actual coco91 class mapping
    ground_truth_annotations = "path_to_coco_annotations.json"  # Replace with actual path
    dataloader = None  # Replace with actual data loader

    model = None  # Replace with your trained model

    # Call evaluate function
    evaluate(dataloader, model, save_json=True, save_path="predictions.json", is_coco=True, coco91class=coco91class, ground_truth_annotations=ground_truth_annotations)

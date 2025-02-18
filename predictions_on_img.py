import cv2

def plot_predictions(image, predictions, labels_dict, color=(0, 255, 0)):
    """
    Draws bounding boxes, class labels, and confidence scores on the image.

    :param image: The image (NumPy array)
    :param predictions: List of dicts containing 'score', 'class_id', 'box'
    :param labels_dict: Dictionary mapping class_id to label name (COCO2017_IDS_TO_LABELS)
    :param color: Color for bounding box (default: green)
    """
    for pred in predictions:
        score = pred["score"]
        class_id = pred["class_id"]
        box = pred["box"]  # Assuming (x1, y1, x2, y2) format

        x1, y1, x2, y2 = map(int, box)  # Convert to integer

        # Get class name from dictionary
        class_name = labels_dict.get(class_id, "Unknown")

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label text: "Class: Score"
        label = f"{class_name}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw text background
        cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)

        # Put text on the image
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

# Example Usage:
# predictions = [{"score": 0.85, "class_id": 1, "box": [50, 30, 200, 150]}]
# image = cv2.imread("image.jpg")
# output_image = plot_predictions(image, predictions, COCO2017_IDS_TO_LABELS)
# cv2.imshow("Detections", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

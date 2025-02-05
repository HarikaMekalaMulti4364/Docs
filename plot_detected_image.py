import matplotlib.pyplot as plt
import cv2

# Function to plot and save image
def plot_and_save_image(img, targets, paths, save_path, names):
    # Create a figure to display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
    plt.axis('off')  # Hide axes

    # You can add any additional plotting for targets or paths here
    for target in targets:
        # Assuming targets contain bounding boxes and other details (customize as needed)
        plt.plot([target[0], target[2]], [target[1], target[3]], 'r-', linewidth=2)

    # Add names as labels (or customize for your task)
    for i, path in enumerate(paths):
        plt.text(10, 10 + i * 20, f"{names[i]}: {path}", color='white', fontsize=12)

    # Save the image to the specified file path
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# For plotting labels
f_labels = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
plot_and_save_image(img, targets, paths, f_labels, names)

# For plotting predictions
f_pred = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
plot_and_save_image(img, output_to_target(out), paths, f_pred, names)

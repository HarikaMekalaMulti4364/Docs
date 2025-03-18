import onnxruntime as ort
import numpy as np
import pickle  # To load .pb file
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load ONNX model
onnx_model_path = "sub_pixel_cnn_2016.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Load the .pb input file
pb_path = "input.pb"  # Replace with your actual .pb file path
with open(pb_path, "rb") as f:
    img_input = pickle.load(f)  # Deserialize

# Ensure input shape is (1,1,224,224)
img_input = np.array(img_input, dtype=np.float32)
if img_input.shape != (1, 1, 224, 224):
    raise ValueError(f"Unexpected input shape {img_input.shape}, expected (1,1,224,224)")

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: img_input})[0]  # Output shape: (1,1,672,672)

# Convert output back to image
img_out_y = Image.fromarray((output[0, 0] * 255.0).clip(0, 255).astype(np.uint8))

# Load ground truth HR image (for metric calculation)
hr_img_path = "ground_truth_hr.jpg"  # Provide the actual HR image path
hr_img = Image.open(hr_img_path).convert("L")  # Convert to grayscale
hr_img = hr_img.resize((672, 672), Image.BICUBIC)  # Ensure same resolution

# Convert HR image to NumPy array
hr_array = np.asarray(hr_img, dtype=np.float32)

# Convert super-resolved output to NumPy array
sr_array = np.asarray(img_out_y, dtype=np.float32)

# Compute PSNR & SSIM
psnr_value = psnr(hr_array, sr_array, data_range=255)
ssim_value = ssim(hr_array, sr_array, data_range=255)

print(f"PSNR: {psnr_value:.4f} dB")
print(f"SSIM: {ssim_value:.4f}")

# Display output image
plt.subplot(1, 2, 1)
plt.imshow(hr_img, cmap="gray")
plt.title("Ground Truth HR Image")

plt.subplot(1, 2, 2)
plt.imshow(img_out_y, cmap="gray")
plt.title("Super-Resolved Output")

plt.show()

# Save output
img_out_y.save("output_super_resolved.jpg")

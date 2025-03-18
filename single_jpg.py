import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
from resizeimage import resizeimage
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load ONNX model
onnx_model_path = "sub_pixel_cnn_2016.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Load and preprocess the image
image_path = "input.jpg"  # Change this to your actual image path
orig_img = Image.open(image_path)

# Save original size
orig_size = orig_img.size  # (width, height)

# Convert to YCbCr and extract Y-channel
img_ycbcr = orig_img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()
img_ndarray = np.asarray(img_y).astype(np.float32) / 255.0

# Resize for model input (224x224)
img_resized = resizeimage.resize_cover(orig_img, [224, 224], validate=False)
img_y_resized = img_resized.convert('YCbCr').split()[0]  # Y-channel only
img_input = np.asarray(img_y_resized).astype(np.float32) / 255.0
img_input = np.expand_dims(np.expand_dims(img_input, axis=0), axis=0)  # (1,1,224,224)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: img_input})[0]  # (1,1,672,672)

# Convert output back to image format
img_out_y = Image.fromarray((output[0, 0] * 255.0).clip(0, 255).astype(np.uint8))

# Resize color channels to match output
img_cb = img_cb.resize(img_out_y.size, Image.BICUBIC)
img_cr = img_cr.resize(img_out_y.size, Image.BICUBIC)

# Merge channels and convert to RGB
final_img = Image.merge("YCbCr", [img_out_y, img_cb, img_cr]).convert("RGB")

# Resize original image to match output size (for evaluation)
orig_img_resized = orig_img.resize(img_out_y.size, Image.BICUBIC)

# Convert images to grayscale numpy arrays for PSNR/SSIM
orig_gray = np.array(orig_img_resized.convert("L"))
sr_gray = np.array(final_img.convert("L"))

# Compute PSNR & SSIM
psnr_value = psnr(orig_gray, sr_gray, data_range=255)
ssim_value = ssim(orig_gray, sr_gray, data_range=255)

# Display results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(orig_img)
plt.title("Original Image")
plt.subplot(1,2,2)
plt.imshow(final_img)
plt.title(f"Super-Resolved Image\nPSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.4f}")
plt.show()

# Save output
final_img.save("output_super_resolved.jpg")

# Print metrics
print(f"PSNR: {psnr_value:.2f} dB")
print(f"SSIM: {ssim_value:.4f}")

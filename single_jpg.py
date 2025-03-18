import onnxruntime as ort
import numpy as np
from PIL import Image
from resizeimage import resizeimage
import matplotlib.pyplot as plt

# Load ONNX model
onnx_model_path = "sub_pixel_cnn_2016.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Load and preprocess the image
image_path = "input.jpg"  # Change this to your actual image path
orig_img = Image.open(image_path)

# Save original size
orig_size = orig_img.size  # (width, height)

# Resize to model's expected input shape (224x224)
img = resizeimage.resize_cover(orig_img, [224, 224], validate=False)
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

# Convert to numpy array and normalize
img_ndarray = np.asarray(img_y).astype(np.float32) / 255.0
img_input = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)  # Shape: (1,1,224,224)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: img_input})[0]  # Shape: (1,1,672,672)

# Convert output back to image format
img_out_y = Image.fromarray((output[0, 0] * 255.0).clip(0, 255).astype(np.uint8))

# Resize color channels to match output
img_cb = img_cb.resize(img_out_y.size, Image.BICUBIC)
img_cr = img_cr.resize(img_out_y.size, Image.BICUBIC)

# Merge channels and convert to RGB
final_img = Image.merge("YCbCr", [img_out_y, img_cb, img_cr]).convert("RGB")

# Display results
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(orig_img)
plt.title("Original Image")
plt.subplot(1,2,2)
plt.imshow(final_img)
plt.title("Super-Resolved Image")
plt.show()

# Save output
final_img.save("output_super_resolved.jpg")

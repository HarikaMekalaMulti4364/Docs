import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from resizeimage import resizeimage
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Directories
base_dir = "BSD300"
test_img_dir = os.path.join(base_dir, "images", "test")  # HR images
sr_dir = os.path.join(base_dir, "SR_output", "test")  # Super-resolved output
os.makedirs(sr_dir, exist_ok=True)

# Load test image filenames
test_list_path = os.path.join(base_dir, "iids_test.txt")
with open(test_list_path, "r") as f:
    test_images = [line.strip() for line in f.readlines()]

# Load ONNX model
model_path = "super-resolution-10.onnx"
ort_session = ort.InferenceSession(model_path)

# Prepare for evaluation
psnr_scores, ssim_scores = [], []

# Process each test image
for img_name in tqdm(test_images):
    hr_img_path = os.path.join(test_img_dir, img_name)
    sr_img_path = os.path.join(sr_dir, img_name)

    # Load HR Image
    orig_img = Image.open(hr_img_path)

    # Resize to model input size (224×224)
    lr_img = resizeimage.resize_cover(orig_img, [224, 224], validate=False)

    # Convert to YCbCr
    img_ycbcr = lr_img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    # Convert Y channel to numpy & normalize
    img_ndarray = np.asarray(img_y).astype(np.float32) / 255.0
    img_input = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)  # Shape: (1,1,224,224)

    # Run ONNX model
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    img_out_y = ort_session.run([output_name], {input_name: img_input})[0]  # Output: (1,1,672,672)

    # Post-processing
    img_out_y = np.squeeze(img_out_y)  # Remove batch & channel dimensions
    img_out_y = (img_out_y * 255.0).clip(0, 255).astype(np.uint8)  # Denormalize

    # Convert numpy to PIL Image
    img_out_y = Image.fromarray(img_out_y)

    # Resize Cb and Cr channels to match SR resolution (672×672)
    img_cb = img_cb.resize(img_out_y.size, Image.BICUBIC)
    img_cr = img_cr.resize(img_out_y.size, Image.BICUBIC)

    # Merge channels back
    final_img = Image.merge("YCbCr", [img_out_y, img_cb, img_cr]).convert("RGB")

    # Save output
    final_img.save(sr_img_path)

    # Convert HR Image to numpy for evaluation
    hr_np = np.array(orig_img.resize((672, 672), Image.BICUBIC))  # Resize HR for comparison
    sr_np = np.array(final_img)

    # Compute PSNR & SSIM
    psnr_value = psnr(hr_np, sr_np, data_range=255)
    ssim_value = ssim(hr_np, sr_np, multichannel=True, data_range=255)

    psnr_scores.append(psnr_value)
    ssim_scores.append(ssim_value)

# Compute Average PSNR & SSIM
avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores)

print(f"\nEvaluation Complete!")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")

# Show an example
plt.imshow(final_img)
plt.title("Super-Resolved Image")
plt.axis("off")
plt.show()

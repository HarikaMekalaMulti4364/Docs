python evaluate.py --onnx_model sub_pixel_cnn_2016.onnx \
                   --test_dir path/to/test/images \
                   --test_list path/to/iids_test.txt \
                   --output_dir output_results \
                   --device cpu  # or cuda


import os
import onnxruntime as ort
import numpy as np
from PIL import Image
from resizeimage import resizeimage
import torch
import argparse
from math import log10
import matplotlib.pyplot as plt

# Define Argument Parser
parser = argparse.ArgumentParser(description="ONNX Super Resolution Evaluation")
parser.add_argument('--onnx_model', type=str, required=True, help="Path to the ONNX model")
parser.add_argument('--test_dir', type=str, required=True, help="Path to the test images directory")
parser.add_argument('--test_list', type=str, required=True, help="Path to the test image list (iids_test.txt)")
parser.add_argument('--output_dir', type=str, default='output_images', help="Path to save output images")
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help="Inference device")
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Load ONNX model
providers = ['CUDAExecutionProvider'] if args.device == 'cuda' else ['CPUExecutionProvider']
session = ort.InferenceSession(args.onnx_model, providers=providers)

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Read test image list
with open(args.test_list, 'r') as f:
    test_images = [line.strip() + ".jpg" for line in f.readlines()]

# Define Mean Squared Error (MSE) function
def mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

# Evaluate PSNR
total_psnr = 0
count = 0

for img_name in test_images:
    img_path = os.path.join(args.test_dir, img_name)

    # Load original image
    orig_img = Image.open(img_path).convert("RGB")
    width, height = orig_img.size
    print(f"Processing {img_name}, Original Shape: ({height}, {width})")

    # Preprocessing
    img = resizeimage.resize_cover(orig_img, [224, 224], validate=False)
    img_ycbcr = img.convert('YCbCr')
    img_y_0, img_cb, img_cr = img_ycbcr.split()

    img_ndarray = np.asarray(img_y_0).astype(np.float32) / 255.0
    img_input = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)

    # Run ONNX model inference
    img_out_y = session.run([output_name], {input_name: img_input})[0]
    img_out_y = img_out_y.squeeze() * 255.0
    img_out_y = Image.fromarray(np.uint8(np.clip(img_out_y, 0, 255)))

    # Post-processing (merge YCbCr and convert to RGB)
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]
    ).convert("RGB")

    # Save output image
    output_path = os.path.join(args.output_dir, img_name)
    final_img.save(output_path)

    # Compute PSNR
    orig_resized = orig_img.resize(img_out_y.size, Image.BICUBIC)
    orig_np = np.asarray(orig_resized, dtype=np.float32)
    output_np = np.asarray(final_img, dtype=np.float32)

    mse_value = mse(orig_np, output_np)
    if mse_value > 0:
        psnr = 10 * log10(255.0 ** 2 / mse_value)
        total_psnr += psnr
        count += 1
        print(f"PSNR for {img_name}: {psnr:.2f} dB")

# Compute average PSNR
if count > 0:
    avg_psnr = total_psnr / count
    print(f"===> Average PSNR on Test Set: {avg_psnr:.2f} dB")
else:
    print("No valid PSNR computed.")

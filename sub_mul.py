import numpy as np

# Assume processed_img is already (1, 3, 520, 520) int8
# Simulated processed image
processed_img = np.random.randint(0, 255, (1, 3, 520, 520), dtype=np.int8)

# Given quantized tensor
q_tensor = np.array([127, 112, 86], dtype=np.int8)
scale = 0.0019013556884601712
zero_point = -128

# Step 1: Dequantize the tensor
dequantized_tensor = (q_tensor.astype(np.float32) - zero_point) * scale  # Shape: (3,)

# Step 2: Broadcast it to match processed_img shape (1, 3, 520, 520)
dequantized_tensor = dequantized_tensor.reshape(1, 3, 1, 1)  # Expand dims for broadcasting

# Step 3: Convert processed_img to float before subtraction
processed_img_float = processed_img.astype(np.float32)

# Step 4: Perform subtraction
output_float = processed_img_float - dequantized_tensor  # Element-wise subtraction

# Step 5: Requantize back to int8
output_int8 = np.clip(np.round(output_float / scale) + zero_point, -128, 127).astype(np.int8)

print(output_int8.shape)  # Should be (1, 3, 520, 520)

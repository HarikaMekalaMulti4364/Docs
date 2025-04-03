import tensorflow as tf

class TFModel(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 5], dtype=tf.float32)])
    def __call__(self, x):
        return x * tf.nn.relu6(x + 3) / 6  # Manual HardSwish implementation

# Create and save TFLite model
tf_model = TFModel()
converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_model.__call__.get_concrete_function()])
tflite_model = converter.convert()

# Save TFLite model
with open("hardswish_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully!")

# Step 2: Define an equivalent ONNX model with HardSwish
class ONNXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        return self.hardswish(x)

# Create PyTorch model
onnx_model = ONNXModel()

# Convert to ONNX
dummy_input = torch.randn(1, 5)
torch.onnx.export(onnx_model, dummy_input, "hardswish_model.onnx", 
                  input_names=["input"], output_names=["output"], opset_version=11)

print("TFLite and ONNX models have been successfully created!")

import numpy as np
import tensorflow as tf
import onnxruntime as ort
import time

# Load TFLite model
tflite_model_path = "hardswish_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load ONNX model
onnx_model_path = "hardswish_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Generate the same random input for both models
input_data = np.random.randn(1, 5).astype(np.float32)

# Run inference on TFLite model
interpreter.set_tensor(input_details[0]['index'], input_data)
start_tflite = time.time()
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])
end_tflite = time.time()

# Run inference on ONNX model
start_onnx = time.time()
onnx_output = ort_session.run(None, {"input": input_data})[0]
end_onnx = time.time()

# Compare outputs
diff = np.abs(tflite_output - onnx_output)
mean_diff = np.mean(diff)
max_diff = np.max(diff)

# Print comparison results
print(f"TFLite Output: {tflite_output}")
print(f"ONNX Output: {onnx_output}")
print(f"Mean Absolute Difference: {mean_diff}")
print(f"Max Difference: {max_diff}")
print(f"TFLite Inference Time: {end_tflite - start_tflite:.6f} sec")
print(f"ONNX Inference Time: {end_onnx - start_onnx:.6f} sec")

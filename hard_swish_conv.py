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

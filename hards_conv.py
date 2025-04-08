import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import time
import onnx
import onnxruntime as ort

# ====================== TFLITE MODEL ======================
class TFModel(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 5], dtype=tf.float32)])
    def __call__(self, x):
        return tf.keras.activations.hard_silu(x)  # Equivalent to HardSwish

# Create and convert to TFLite
tf_model = TFModel()
concrete_func = tf_model.__call__.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save TFLite model
tflite_model_path = "hardswish_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
print("✅ TFLite model using tf.keras.activations.hard_silu created!")

# ====================== ONNX MODEL ======================
class ONNXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        return self.hardswish(x)

# Create and export to ONNX
onnx_model = ONNXModel()
dummy_input = torch.randn(1, 5)
onnx_model_path = "hardswish_model.onnx"
torch.onnx.export(onnx_model, dummy_input, onnx_model_path,
                  input_names=["input"], output_names=["output"], opset_version=11)
print("✅ ONNX model with nn.Hardswish() created!")

# ====================== RUN INFERENCE ======================

# Generate the same input
input_data = np.random.randn(1, 5).astype(np.float32)

# --- Run TFLite inference ---
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
start_tflite = time.time()
interpreter.invoke()
end_tflite = time.time()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

# --- Run ONNX inference ---
ort_session = ort.InferenceSession(onnx_model_path)
start_onnx = time.time()
onnx_output = ort_session.run(None, {"input": input_data})[0]
end_onnx = time.time()

# ====================== COMPARE OUTPUTS ======================
diff = np.abs(tflite_output - onnx_output)
mean_diff = np.mean(diff)
max_diff = np.max(diff)

print("\n🔍 Output Comparison")
print(f"Input: {input_data}")
print(f"TFLite Output : {tflite_output}")
print(f"ONNX Output   : {onnx_output}")
print(f"Mean Absolute Difference: {mean_diff:.8f}")
print(f"Max  Absolute Difference: {max_diff:.8f}")
print(f"TFLite Inference Time: {end_tflite - start_tflite:.6f} sec")
print(f"ONNX   Inference Time: {end_onnx - start_onnx:.6f} sec")

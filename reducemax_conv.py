import tensorflow as tf
import torch
import onnx
import onnxruntime as ort
import numpy as np
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper

# ========================== Step 1: Create & Save TFLite Model ========================== #
class ReduceMaxModelTF(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 3, 3], dtype=tf.float32)])
    def __call__(self, x):
        return tf.math.reduce_max(x, axis=2, keepdims=True)

# Create TensorFlow model
model_tf = ReduceMaxModelTF()

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [model_tf.__call__.get_concrete_function()], trackable_obj=model_tf
)
tflite_model = converter.convert()

# Save the TFLite model
with open("reducemax_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as 'reducemax_model.tflite'")

# ========================== Step 2: Create & Save ONNX Model ========================== #
# Define input and output tensor
input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 3])
output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 1])

# Define 'axes' as an initializer tensor (Fixing InvalidGraph issue)
axes_tensor = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes")

# Define ReduceMax Node
reduce_max_node = helper.make_node(
    "ReduceMax",
    inputs=["input", "axes"],  # Now takes 'axes' as an input
    outputs=["output"],
    keepdims=1
)

# Create graph
graph = helper.make_graph(
    [reduce_max_node],
    "ReduceMaxGraph",
    [input_tensor],
    [output_tensor],
    [axes_tensor]  # Include 'axes' tensor in initializers
)

# Create ONNX model
onnx_model = helper.make_model(graph, producer_name="onnx-reducemax")
onnx.save(onnx_model, "reducemax_model.onnx")

print("✅ Fixed ONNX model saved as 'reducemax_model.onnx'")

# ========================== Step 3: Run Inference & Compare Outputs ========================== #
# Generate a random input tensor
input_data = np.random.rand(1, 3, 3).astype(np.float32)

# Run TFLite Inference
interpreter = tf.lite.Interpreter(model_path="reducemax_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

# Run ONNX Inference
ort_session = ort.InferenceSession("reducemax_model.onnx")
onnx_output = ort_session.run(None, {"input": input_data})[0]

# Compare outputs
print("\n🔍 Input Tensor:\n", input_data)
print("\n📌 TFLite Output:\n", tflite_output)
print("\n📌 ONNX Output:\n", onnx_output)

# Check if outputs match
if np.allclose(tflite_output, onnx_output, atol=1e-5):
    print("\n✅ Outputs Match! The ONNX and TFLite models are equivalent.")
else:
    print("\n❌ Outputs Mismatch! Further debugging needed.")

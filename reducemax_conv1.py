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

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 3, 3], dtype=tf.float32), 
        tf.TensorSpec(shape=[1], dtype=tf.int32)  # Axes as input
    ])
    def __call__(self, x, axes):
        return tf.math.reduce_max(x, axis=axes[0], keepdims=True)

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
axes_tensor = helper.make_tensor_value_info("axes", onnx.TensorProto.INT64, [1])  # Axes as input
output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 1])

# Define ReduceMax Node
reduce_max_node = helper.make_node(
    "ReduceMax",
    inputs=["input", "axes"],  # Axes as an input instead of initializer
    outputs=["output"],
    keepdims=1
)

# Create graph
graph = helper.make_graph(
    [reduce_max_node],
    "ReduceMaxGraph",
    [input_tensor, axes_tensor],  # Both 'input' and 'axes' are now inputs
    [output_tensor]
)

# Create ONNX model
onnx_model = helper.make_model(graph, producer_name="onnx-reducemax")

# Fix attribute issue by explicitly specifying axes data type
onnx_model.graph.input[1].type.tensor_type.elem_type = onnx.TensorProto.INT64

# Save ONNX model
onnx.save(onnx_model, "reducemax_model.onnx")

print("✅ Fixed ONNX model saved as 'reducemax_model.onnx'")

# ========================== Step 3: Run Inference & Compare Outputs ========================== #
# Generate a random input tensor
input_data = np.random.rand(1, 3, 3).astype(np.float32)
axes_data = np.array([2], dtype=np.int64)  # Reducing along last axis

# Run TFLite Inference
interpreter = tf.lite.Interpreter(model_path="reducemax_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Identify correct indices for input tensor and axes
tensor_index = None
axes_index = None

for i, inp in enumerate(input_details):
    if inp['dtype'] == np.float32:
        tensor_index = inp['index']
    elif inp['dtype'] == np.int32:
        axes_index = inp['index']

# Ensure indices were found
assert tensor_index is not None, "Failed to find tensor input index"
assert axes_index is not None, "Failed to find axes input index"

# Set the tensors correctly
interpreter.set_tensor(tensor_index, input_data)
interpreter.set_tensor(axes_index, axes_data.astype(np.int32))  # Axes must be INT32

# Run inference
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

# Run ONNX Inference
ort_session = ort.InferenceSession("reducemax_model.onnx")
onnx_output = ort_session.run(None, {"input": input_data, "axes": axes_data})[0]

# Compare outputs
print("\n🔍 Input Tensor:\n", input_data)
print("\n📌 TFLite Output:\n", tflite_output)
print("\n📌 ONNX Output:\n", onnx_output)

# Check if outputs match
if np.allclose(tflite_output, onnx_output, atol=1e-5):
    print("\n✅ Outputs Match! The ONNX and TFLite models are equivalent.")
else:
    print("\n❌ Outputs Mismatch! Further debugging needed.")

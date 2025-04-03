import tensorflow as tf
import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto
import tensorflow.lite as tflite

# ================================
# Step 1: Create TensorFlow Model
# ================================
class FillModel(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[2], dtype=tf.int64)])
    def fill_tensor(self, shape):
        value = tf.constant(5.0, dtype=tf.float32)  # Fill value
        return tf.fill(shape, value)

# Create model instance
fill_model = FillModel()

# Convert TensorFlow model to TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [fill_model.fill_tensor.get_concrete_function()]
)
tflite_model = converter.convert()

# Save the TFLite model
with open("fill_model.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ TFLite model saved: fill_model.tflite")

# ================================
# Step 2: Create ONNX Model
# ================================
node = helper.make_node(
    "ConstantOfShape",
    inputs=["shape"],
    outputs=["output"],
    value=helper.make_tensor(
        name="value",
        data_type=TensorProto.FLOAT,
        dims=[1],  # Must be 1D
        vals=[5.0],
    ),
)

# Define model inputs/outputs
shape_tensor = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])

# Create ONNX graph & model
graph = helper.make_graph([node], "Fill_Example", [shape_tensor], [output_tensor])
model = helper.make_model(graph)
onnx.save(model, "fill_example.onnx")
print("✅ ONNX model saved: fill_example.onnx")

# ================================
# Step 3: Run Inference
# ================================

# Run ONNX model
onnx_input = {"shape": np.array([2, 3], dtype=np.int64)}
ort_session = ort.InferenceSession("fill_example.onnx")
onnx_result = ort_session.run(None, onnx_input)[0]
print("\nONNX ConstantOfShape Output:\n", onnx_result)

# Run TFLite model
interpreter = tflite.Interpreter(model_path="fill_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input and run inference
shape_input = np.array([2, 3], dtype=np.int64)
interpreter.set_tensor(input_details[0]['index'], shape_input)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

print("\nTFLite Fill Output:\n", tflite_output)

# ================================
# Step 4: Compare Outputs
# ================================
match = np.array_equal(onnx_result, tflite_output)
print("\n✅ Outputs Match:", match)

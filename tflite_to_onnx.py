import tensorflow as tf
import tf2onnx
import onnx
import numpy as np

# Paths
tflite_model_path = "model.tflite"
saved_model_dir = "saved_model"
onnx_model_path = "model.onnx"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Convert TFLite to TensorFlow SavedModel
def representative_dataset():
    for _ in range(100):
        yield [tf.random.normal([1, 224, 224, 3], dtype=tf.float32)]  # Adjust shape as needed

converter = tf.lite.TFLiteConverter.from_saved_model(tflite_model_path)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tf.saved_model.save(interpreter, saved_model_dir)

# Convert TensorFlow SavedModel to ONNX
model_proto, _ = tf2onnx.convert.from_saved_model(saved_model_dir, opset=13)
with open(onnx_model_path, "wb") as f:
    f.write(model_proto.SerializeToString())

# Verify ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX Model Graph:\n", onnx.helper.printable_graph(onnx_model.graph))

print(f"Conversion successful! ONNX model saved at {onnx_model_path}")

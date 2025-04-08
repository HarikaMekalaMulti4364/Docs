import tensorflow as tf
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

# 1. Build TFLite Model using tf.raw_ops.Shape
def build_tflite_shape_model():
    class ShapeModel(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)])
        def __call__(self, x):
            return tf.raw_ops.Shape(input=x, out_type=tf.dtypes.int64)

    model = ShapeModel()
    concrete_func = model.__call__.get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with open("shape_model.tflite", "wb") as f:
        f.write(tflite_model)

# 2. Build ONNX Model using Shape + Cast
def build_onnx_shape_cast_model():
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 4])
    shape_out = helper.make_tensor_value_info('shape_out', TensorProto.FLOAT, [None])

    shape_node = helper.make_node('Shape', ['input'], ['shape'])
    cast_node = helper.make_node('Cast', ['shape'], ['shape_out'], to=TensorProto.FLOAT)

    graph = helper.make_graph([shape_node, cast_node], 'ShapeCastGraph', [input_tensor], [shape_out])
    model = helper.make_model(graph, producer_name='shape_cast_model')
    onnx.save(model, 'shape_model.onnx')

# 3. Run Inference and Compare Outputs
def run_and_compare():
    input_data = np.random.rand(5, 4).astype(np.float32)

    # Run TFLite
    interpreter = tf.lite.Interpreter(model_path="shape_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize input tensor to match shape
    interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    print("TFLite Output:", tflite_output)

    # Run ONNX
    session = ort.InferenceSession("shape_model.onnx")
    onnx_output = session.run(None, {"input": input_data})[0]
    print("ONNX Output:", onnx_output)

    # Compare
    if np.allclose(tflite_output.astype(np.float32), onnx_output, rtol=1e-3, atol=1e-3):
        print("✅ Outputs match!")
    else:
        print("❌ Outputs differ!")

# Run everything
print("Creating TFLite model using Shape...")
build_tflite_shape_model()

print("Creating ONNX model using Shape + Cast...")
build_onnx_shape_cast_model()

print("Running inference and comparing outputs...")
run_and_compare()

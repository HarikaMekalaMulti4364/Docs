import tensorflow as tf
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

# Step 1: Create TFLite model with dynamic axis
def create_tflite_model():
    class ReduceProdModule(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[2, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[1], dtype=tf.int32)
        ])
        def __call__(self, x, axis):
            return tf.math.reduce_prod(x, axis=axis)

    model = ReduceProdModule()
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    with open("reduce_prod_model.tflite", "wb") as f:
        f.write(tflite_model)

    return "reduce_prod_model.tflite"

# Step 2: Create ONNX model with dynamic axis input
# Fix: axes is an attribute, not an input
def create_onnx_model():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])

    reduce_node = helper.make_node(
        "ReduceProd",
        inputs=["input"],             # only input tensor
        outputs=["output"],
        axes=[1],                     # static attribute
        keepdims=0
    )

    graph = helper.make_graph(
        [reduce_node],
        "ReduceProdGraphFixed",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, "reduce_prod_model.onnx")
    return "reduce_prod_model.onnx"

# Step 3: Compare outputs
def compare_outputs(tflite_path, onnx_path):
    input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    axis = np.array([1], dtype=np.int32)

    # TFLite inference
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.set_tensor(input_details[1]['index'], axis)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    # ONNX inference
    sess = ort.InferenceSession(onnx_path)
    onnx_output = sess.run(None, {"input": input_data})[0]

    # Compare
    print("TFLite Output:", tflite_output)
    print("ONNX Output:  ", onnx_output)
    print("Are outputs close? ->", np.allclose(tflite_output, onnx_output, atol=1e-5))

# Run all
if __name__ == "__main__":
    tflite_path = create_tflite_model()
    onnx_path = create_onnx_model()
    compare_outputs(tflite_path, onnx_path)

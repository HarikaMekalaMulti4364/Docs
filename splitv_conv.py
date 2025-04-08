import tensorflow as tf
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import tempfile

# ----------------------------
# 1. Build TFLite model (SplitV)
# ----------------------------
def build_tflite_splitv_model():
    class SplitVModel(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[6], dtype=tf.float32)])
        def __call__(self, x):
            return tf.raw_ops.SplitV(
                value=x,
                size_splits=[2, 2, 2],
                axis=0,
                num_split=3
            )

    model = SplitVModel()
    concrete_func = model.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    tflite_model = converter.convert()

    with open("splitv_model.tflite", "wb") as f:
        f.write(tflite_model)

# ----------------------------
# 2. Build ONNX model (Split)
# ----------------------------
def build_onnx_split_model():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [6])
    split_tensor = helper.make_tensor_value_info("split", TensorProto.INT64, [3])
    outputs = [
        helper.make_tensor_value_info(f"output_{i}", TensorProto.FLOAT, [2])
        for i in range(3)
    ]

    node = helper.make_node(
        "Split",
        inputs=["input", "split"],  # Use split as input
        outputs=["output_0", "output_1", "output_2"],
        axis=0
    )

    graph = helper.make_graph(
        [node],
        "SplitGraph",
        [input_tensor, split_tensor],
        outputs
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, "split_model.onnx")

# ----------------------------
# 3. Run and compare
# ----------------------------
def run_and_compare():
    input_data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)

    # TFLite inference
    interpreter = tf.lite.Interpreter(model_path="splitv_model.tflite")
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    tflite_outputs = [
        interpreter.get_tensor(interpreter.get_output_details()[i]['index'])
        for i in range(3)
    ]

    # ONNX inference
    ort_session = ort.InferenceSession("split_model.onnx")
    split_sizes = np.array([2, 2, 2], dtype=np.int64)
    onnx_outputs = ort_session.run(None, {"input": input_data, "split": split_sizes})

    # Compare
    for i in range(3):
        print(f"\n--- Output {i} ---")
        print("TFLite:", tflite_outputs[i])
        print("ONNX:  ", onnx_outputs[i])
        print("Close? ", np.allclose(tflite_outputs[i], onnx_outputs[i]))

# ----------------------------
# Run everything
# ----------------------------
if __name__ == "__main__":
    print("Creating TFLite model using tf.raw_ops.SplitV...")
    build_tflite_splitv_model()

    print("Creating ONNX model using Split...")
    build_onnx_split_model()

    print("Running inference and comparing outputs...")
    run_and_compare()

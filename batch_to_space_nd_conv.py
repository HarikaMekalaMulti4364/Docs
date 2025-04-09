import tensorflow as tf
import numpy as np
import onnx
import onnxruntime as ort
import onnx.helper
import onnx.numpy_helper

def create_tflite_batch_to_space_model():
    class B2SModel(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[4, 1, 1, 1], dtype=tf.float32)
        ])
        def b2s(self, x):
            return tf.raw_ops.BatchToSpaceND(
                input=x,
                block_shape=[2, 2],
                crops=[[0, 0], [0, 0]]
            )

    model = B2SModel()
    concrete_func = model.b2s.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    tflite_model = converter.convert()
    with open("batch_to_space_model.tflite", "wb") as f:
        f.write(tflite_model)
    return "batch_to_space_model.tflite"

def create_onnx_batch_to_space_model():
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4, 1, 1, 1])
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2, 2, 1])

    # Reshape: [4,1,1,1] → [1,4,1,1]
    reshape_shape = onnx.helper.make_tensor("reshape_shape", onnx.TensorProto.INT64, [4], [1, 4, 1, 1])
    reshape_node = onnx.helper.make_node("Reshape", ["x", "reshape_shape"], ["reshaped"])

    # Transpose: NHWC → NCHW
    t1 = onnx.helper.make_node("Transpose", ["reshaped"], ["t1_out"], perm=[0, 1, 2, 3])  # No real change

    # DepthToSpace with blocksize=2 (C=4 is divisible by 4)
    d2s = onnx.helper.make_node("DepthToSpace", ["t1_out"], ["d2s_out"], blocksize=2)

    # Transpose back: NCHW → NHWC
    t2 = onnx.helper.make_node("Transpose", ["d2s_out"], ["t2_out"], perm=[0, 2, 3, 1])

    # Final Slice (dummy since no crop)
    starts = onnx.helper.make_tensor("starts", onnx.TensorProto.INT64, [4], [0, 0, 0, 0])
    ends = onnx.helper.make_tensor("ends", onnx.TensorProto.INT64, [4], [1, 2, 2, 1])
    axes = onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [4], [0, 1, 2, 3])
    steps = onnx.helper.make_tensor("steps", onnx.TensorProto.INT64, [4], [1, 1, 1, 1])

    slice_node = onnx.helper.make_node("Slice", ["t2_out", "starts", "ends", "axes", "steps"], ["output"])

    graph = onnx.helper.make_graph(
        [reshape_node, t1, d2s, t2, slice_node],
        "B2S_with_DepthToSpace",
        [x],
        [y],
        initializer=[reshape_shape, starts, ends, axes, steps]
    )

    model = onnx.helper.make_model(graph, producer_name="onnx-b2s-d2s-style")
    onnx.save(model, "batch_to_space_model.onnx")
    return "batch_to_space_model.onnx"


def compare_batch_to_space_outputs(tflite_path, onnx_path):
    input_data = np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]], dtype=np.float32)  # (4,1,1,1)

    # Run TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    # Run ONNX
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {"x": input_data})[0]

    expected_output = np.array([[[[1.], [2.]], [[3.], [4.]]]], dtype=np.float32)

    print("Expected Output:\n", expected_output)
    print("TFLite Output:\n", tflite_output)
    print("ONNX Output:\n", onnx_output)
    print("Are outputs similar?", np.allclose(tflite_output, onnx_output))

if __name__ == "__main__":
    tflite_path = create_tflite_batch_to_space_model()
    onnx_path = create_onnx_batch_to_space_model()
    compare_batch_to_space_outputs(tflite_path, onnx_path)

import numpy as np
import tensorflow as tf
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
import os

# Step 1: Create TFLite model using tf.raw_ops.Sum
def build_and_save_tflite_sum_model():
    class SumModel(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[2, 4], dtype=tf.float32)])
        def __call__(self, x):
            return tf.raw_ops.Sum(input=x, axis=[1], keep_dims=True)

    model = SumModel()
    tf.saved_model.save(model, "sum_tf_model", signatures=model.__call__.get_concrete_function())

    converter = tf.lite.TFLiteConverter.from_saved_model("sum_tf_model")
    tflite_model = converter.convert()

    with open("sum_model.tflite", "wb") as f:
        f.write(tflite_model)

# Step 2: Create ONNX model using ReduceSum (opset ≥ 13)
def build_and_save_onnx_reducesum_model():
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 4])
    axes_tensor = helper.make_tensor_value_info('axes', TensorProto.INT64, [1])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 1])

    axes_initializer = helper.make_tensor(
        name='axes',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]
    )

    reduce_sum_node = helper.make_node(
        'ReduceSum',
        inputs=['input', 'axes'],
        outputs=['output'],
        keepdims=1
    )

    graph = helper.make_graph(
        [reduce_sum_node],
        'reducesum_graph',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[axes_initializer]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.save(model, "reducesum_model.onnx")

# Step 3: Run inference and compare outputs
def run_and_compare():
    input_data = np.array([[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)

    # TFLite inference
    interpreter = tf.lite.Interpreter(model_path="sum_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])

    # ONNX inference
    ort_session = ort.InferenceSession("reducesum_model.onnx")
    onnx_output = ort_session.run(None, {'input': input_data})[0]

    print("Input:\n", input_data)
    print("TFLite (tf.raw_ops.Sum) Output:\n", tflite_output)
    print("ONNX (ReduceSum) Output:\n", onnx_output)
    print("Are outputs close?", np.allclose(tflite_output, onnx_output, atol=1e-5))

if __name__ == "__main__":
    os.system("rm -rf sum_tf_model sum_model.tflite reducesum_model.onnx")

    print("Creating TFLite model using tf.raw_ops.Sum...")
    build_and_save_tflite_sum_model()

    print("Creating ONNX model using ReduceSum...")
    build_and_save_onnx_reducesum_model()

    print("Running inference and comparing outputs...")
    run_and_compare()

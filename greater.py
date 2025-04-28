def test_greater(self):
    import os
    import tensorflow as tf

    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "greater.tflite")
    convert_filename = os.path.join(out_dir, "greater.onnx")

    class CustomModel(tf.Module):
        def __init__(self, name):
            super().__init__(name=name)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.float32),
        ])
        def __call__(self, x):
            threshold = tf.constant(0.5, dtype=tf.float32)
            y = tf.math.greater(x, threshold)
            return y

    model = CustomModel("test_greater")
    self.convert_saved_model(model, filename)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        import onnx
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)




import numpy as np
from onnx import helper, TensorProto

@Converter.Register("placeholder_for_greater_op_codes")
def parse_placeholder_for_greater_op_codes(parser):
    input0_name = parser.get_tensor_name(parser.inputs[0])  # input tensor (x)
    input1_name = parser.get_tensor_name(parser.inputs[1])  # threshold tensor (0.5)
    output_name = parser.get_tensor_name(parser.outputs[0])  # output tensor
    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()

    parser.add_onnx_operator(
        "Greater",
        [input0_name, input1_name],
        [output_name],
        {},
        ip_quant_params,
        op_quant_params
    )

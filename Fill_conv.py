def test_fill(self):
    import os
    import tensorflow as tf

    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "fill.tflite")
    convert_filename = os.path.join(out_dir, "fill.onnx")

    class CustomModel(tf.Module):
        def __init__(self, name):
            super().__init__(name=name)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[2], dtype=tf.int32),  # shape input
        ])
        def __call__(self, shape):
            value = tf.constant(2.0, dtype=tf.float32)  # Constant scalar value
            y = tf.fill(dims=shape, value=value)
            return y

    model = CustomModel("test_fill")
    self.convert_saved_model(model, filename)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        import onnx
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)

import numpy as np
from onnx import helper, TensorProto

@Converter.Register("FILL")
def parse_FILL(parser):
    shape_name = parser.get_tensor_name(parser.inputs[0])  # shape input
    output_name = parser.get_tensor_name(parser.outputs[0])  # output
    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()

    # Set scalar value directly (must match TF model's hardcoded value)
    scalar_value = 2.0
    value_tensor = np.array([scalar_value], dtype=np.float32)

    value_attr = helper.make_tensor(
        name="value",  # name is required but arbitrary
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=value_tensor
    )

    parser.add_onnx_operator(
        "ConstantOfShape",
        [shape_name],
        [output_name],
        {"value": value_attr},
        ip_quant_params,
        op_quant_params
    )

 axes_name = output_name + "/axes"
    axes = np.array(squeeze_dims.tolist()).astype(
        np.int64
    )  # convert list to array; onnx only accepts int64
    parser.add_ndarray_to_tensor_dict(axes_name, axes)
    parser.add_onnx_operator("Squeeze", [input_name, axes_name], [output_name], {}, ip_quant_params, op_quant_params)





def test_cumsum(self):
    import os
    import tensorflow as tf

    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "cumsum.tflite")
    convert_filename = os.path.join(out_dir, "cumsum.onnx")

    class CustomModel(tf.Module):
        def __init__(self, name):
            super().__init__(name=name)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.float32),
        ])
        def __call__(self, x):
            # Use tf.keras.ops.cumsum
            y = tf.keras.ops.cumsum(x)
            return y

    model = CustomModel("test_cumsum")
    self.convert_saved_model(model, filename)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        import onnx
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)




from onnx import helper, TensorProto

@Converter.Register("CUMSUM")
def parse_CUMSUM(parser):
    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])
    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()

    axis = 0  # Default axis=0 unless otherwise specified in your model

    # Create axis constant tensor
    axis_name = parser.create_intermediate_tensor_name()
    parser.add_initializer(
        axis_name,
        TensorProto.INT32,
        [],
        [axis]
    )

    parser.add_onnx_operator(
        "CumSum",
        [input_name, axis_name],
        [output_name],
        {},
        ip_quant_params,
        op_quant_params
    )

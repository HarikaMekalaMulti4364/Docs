def test_reduce_max(self):
    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "reduce_max.tflite")
    convert_filename = os.path.join(out_dir, "reduce_max.onnx")

    class CustomModel(tf.Module):
        def __init__(self, name):
            super().__init__(name=name)

        @tf.function(input_signature=[tf.TensorSpec([10, 10], tf.float32)])
        def __call__(self, x):
            y = tf.math.reduce_max(input_tensor=x)
            return y

    reduce_max = CustomModel("test")
    self.convert_saved_model(reduce_max, filename)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)


@Converter.Register("REDUCE_MAX")
def parse_REDUCE_MAX(parser):
    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])
    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()
    axes = list(parser.get_constant_node(parser.inputs[1]))
    keepdims = parser.option.KeepDims()
    parser.add_onnx_operator(
        "ReduceMax", [input_name], [output_name], {"axes": axes, "keepdims": keepdims},
        ip_quant_params, op_quant_params
    )

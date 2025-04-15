def test_reduce_prod(self):
        out_dir = self.generate_out_dir()
        filename = os.path.join(out_dir, "reduce_prod.tflite")
        convert_filename = os.path.join(out_dir, "reduce_prod.onnx")

        class CustomModel(tf.Module):
            def __init__(self, name):
                super().__init__(name=name)

            @tf.function(input_signature=[tf.TensorSpec([10, 10], tf.float32)])
            def __call__(self, x):
                y = tf.math.reduce_prod(input_tensor=x)
                return y

        reduce_prod = CustomModel("test")
        self.convert_saved_model(reduce_prod, filename)

        is_pass, model_def, _, _ = tflite2mwnn(filename)
        if not self.savespace:
            onnx.save(model_def, convert_filename)
        self.assertTrue(is_pass)

@Converter.Register("REDUCE_PROD")
def parse_REDUCE_PROD(parser):
    # input_names = parser.write_inputs_to_onnx_net()
    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])
    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()
    axes = list(parser.get_constant_node(parser.inputs[1]))
    keepdims = parser.option.KeepDims()
    parser.add_onnx_operator(
        "ReduceProd", [input_name], [output_name], {"axes": axes, "keepdims": keepdims},
        ip_quant_params, op_quant_params
    )

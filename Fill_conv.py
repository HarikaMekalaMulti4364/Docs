def test_fill(self):
    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "fill.tflite")
    convert_filename = os.path.join(out_dir, "fill.onnx")

    class CustomModel(tf.Module):
        def __init__(self, name):
            super().__init__(name=name)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[2], dtype=tf.int32),  # shape tensor
            tf.TensorSpec(shape=[], dtype=tf.float32),  # scalar value
        ])
        def __call__(self, shape, value):
            y = tf.fill(dims=shape, value=value)
            return y

    model = CustomModel("test_fill")
    self.convert_saved_model(model, filename)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)


@Converter.Register("FILL")
def parse_FILL(parser):
    shape_name = parser.get_tensor_name(parser.inputs[0])  # Shape tensor
    value_name = parser.get_tensor_name(parser.inputs[1])  # Scalar value
    output_name = parser.get_tensor_name(parser.outputs[0])

    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()

    # ONNX ConstantOfShape expects the value as an attribute, not a dynamic tensor.
    # So you need to make sure value is constant
    value = parser.get_constant_node(parser.inputs[1])
    value_tensor = numpy.array(value).astype(numpy.float32)
    const_value_name = parser.make_const(name="fill_value", np_val=value_tensor)

    parser.add_onnx_operator(
        "ConstantOfShape",
        [shape_name],
        [output_name],
        {"value": helper.make_tensor("value", TensorProto.FLOAT, [], value_tensor.tolist())},
        ip_quant_params,
        op_quant_params
    )

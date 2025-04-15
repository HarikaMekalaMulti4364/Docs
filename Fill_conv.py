def test_fill(self):
    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "fill.tflite")
    convert_filename = os.path.join(out_dir, "fill.onnx")

    class CustomModel(tf.Module):
        def __init__(self, name):
            super().__init__(name=name)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[2], dtype=tf.int32),     # Shape tensor
            tf.TensorSpec(shape=[1], dtype=tf.float32),   # Scalar value as 1-element tensor
        ])
        def __call__(self, shape, value):
            y = tf.fill(dims=shape, value=value[0])  # Extract scalar from 1-element tensor
            return y

    model = CustomModel("test_fill")
    self.convert_saved_model(model, filename)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)


@Converter.Register("FILL")
def parse_FILL(parser):
    shape_name = parser.get_tensor_name(parser.inputs[0])   # Shape tensor
    value_name = parser.get_tensor_name(parser.inputs[1])   # Dynamic scalar tensor
    output_name = parser.get_tensor_name(parser.outputs[0])

    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()

    # ONNX doesn’t support dynamic value in ConstantOfShape directly.
    # So use Shape -> Gather -> Unsqueeze to extract the scalar
    value_reshaped = parser.make_reshape(value_name, [], "scalar_fill_value")  # [1] -> scalar

    parser.add_onnx_operator(
        "ConstantOfShape",
        [shape_name],
        [output_name],
        {"value": None},   # Pass via dynamic input instead of attribute
        ip_quant_params,
        op_quant_params,
        value_tensor=value_reshaped  # Custom param in your parser to handle dynamic init value
    )

#add
def test_ADD_quantize(self):
    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "add_quantize.tflite")
    convert_filename = os.path.join(out_dir, "add_quantize.onnx")

    class AddModel(tf.Module):
        def __init__(self, name):
            super().__init__(name=name)

        @tf.function(input_signature=[
            tf.TensorSpec([1, 64, 64, 3], tf.float32),
            tf.TensorSpec([1, 64, 64, 3], tf.float32)
        ])
        def __call__(self, x, y):
            return tf.add(x, y)

    model = AddModel("test_add")
    self.convert_saved_model(model, filename, quantize=True, input_shape=(1, 64, 64, 3), num_inputs=2)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)

@Converter.Register("ADD")
def parse_ADD(parser):
    """ONNX: Add
    TFLite: ADD
    Element-wise addition between two tensors.
    """

    input_names = [parser.get_tensor_name(i) for i in parser.inputs]
    output_name = parser.get_tensor_name(parser.outputs[0])

    ipqp = parser._get_input_quantization_params()
    if len(ipqp) < 2:
        ipqp.extend([None] * (2 - len(ipqp)))  # ensure 2 inputs

    opqp = parser._get_output_quantization_params()

    # Align input shapes if required by broadcasting
    input_shapes = parser.inputs_shape
    if input_shapes[0] != input_shapes[1]:
        # Optional: Insert broadcast logic or reshaping here
        pass

    parser.add_onnx_operator(
        "Add",
        input_names,
        [output_name],
        attr_dict={},  # No additional attributes for Add
        input_qparams=ipqp,
        output_qparams=opqp
    )

    generate_activation_transpose(
        parser, output_name, output_name, parser.activation_function, opqp[0]
    )


#Argmax
def test_ARG_MAX(self):
    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "argmax.tflite")
    convert_filename = os.path.join(out_dir, "argmax.onnx")

    class ArgMaxModel(tf.Module):
        def __init__(self, name):
            super().__init__(name=name)

        @tf.function(input_signature=[tf.TensorSpec([1, 16, 16, 10], tf.float32)])
        def __call__(self, x):
            return tf.argmax(x, axis=-1, output_type=tf.int32)

    model = ArgMaxModel("test_argmax")
    self.convert_saved_model(model, filename, quantize=False, input_shape=(1, 16, 16, 10))

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)


@Converter.Register("ARG_MAX")
def parse_ARG_MAX(parser):
    """ONNX: ArgMax + Cast
    TFLite ArgMax outputs int32 but ONNX ArgMax returns int64.
    So, we cast the result from int64 to int32.
    """

    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])

    # Axis from TFLite (normally constant input)
    axis = int(parser.get_constant_node(parser.inputs[1]))

    intermediate_output = output_name + "/ArgMax"

    parser.add_onnx_operator(
        "ArgMax",
        [input_name],
        [intermediate_output],
        attr_dict={"axis": axis, "keepdims": 0},
    )

    # Cast to int32 to match TFLite output type
    parser.add_onnx_operator(
        "Cast",
        [intermediate_output],
        [output_name],
        attr_dict={"to": onnx.TensorProto.INT32}
    )

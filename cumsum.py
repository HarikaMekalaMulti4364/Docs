from onnx import helper

@Converter.Register("GELU")
def parse_GELU(parser):
    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])
    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()

    # Add ONNX Gelu operator
    parser.add_onnx_operator(
        "Gelu",
        [input_name],
        [output_name],
        {"approximate": 0},  # Optional: 0 = exact, 1 = tanh approximation
        ip_quant_params,
        op_quant_params
    )


def test_gelu(self):
    import os
    import tensorflow as tf

    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "gelu.tflite")
    convert_filename = os.path.join(out_dir, "gelu.onnx")

    class CustomModel(tf.Module):
        def __init__(self, name):
            super().__init__(name=name)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.float32),
        ])
        def __call__(self, x):
            # Use tf.keras.activations.gelu (defaults to approximate=False)
            y = tf.keras.activations.gelu(x)
            return y

    model = CustomModel("test_gelu")
    self.convert_saved_model(model, filename)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        import onnx
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)













@Converter.Register("CONV_3D")
def parse_CONV_3D(parser):
    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])
    padding = parser.option.Padding()
    input_shape = parser.inputs_shape[0]  # e.g., [1, D, H, W, C]
    
    strides = [
        parser.option.StrideD(),
        parser.option.StrideH(),
        parser.option.StrideW()
    ]
    
    dilation = [
        parser.option.DilationDFactor(),
        parser.option.DilationHFactor(),
        parser.option.DilationWFactor()
    ]
    
    weight_shape = parser.inputs_shape[1]  # usually [out_channels, kD, kH, kW, in_channels]
    kernel_shape = weight_shape[1:4]  # [kD, kH, kW]

    # Pads returned as [front, back, top, bottom, left, right]
    pads = Parser.get_pads_3d(input_shape, strides, kernel_shape, padding, None, dilation)

    # ONNX expects pads as: [front, top, left, back, bottom, right]
    param_dict = dict(
        dilations=dilation,
        kernel_shape=kernel_shape,
        pads=[pads[i] for i in [0, 2, 4, 1, 3, 5]],
        strides=strides,
    )

    generate_Conv_to_ONNX(parser, input_name, output_name, param_dict)





def test_Conv2D(self):
        out_dir = self.generate_out_dir()
        filename = os.path.join(out_dir, "conv2d.tflite")
        convert_filename = os.path.join(out_dir, "conv2d.onnx")
        self.convert_sequential(
            input_shape=(224, 224, 3),
            test_layer=tf.keras.layers.Conv2D(filters=32, kernel_size=3),
            filename=filename
        )
        is_pass, model_def, _, _ = tflite2mwnn(filename)
        if not self.savespace:
            onnx.save(model_def, convert_filename)
        self.assertTrue(is_pass)


@Converter.Register("CONV_2D")
def parse_CONV_2D(parser):
    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])
    padding = parser.option.Padding()
    input_shape = parser.inputs_shape[0]
    strides = [parser.option.StrideH(), parser.option.StrideW()]
    dilation = [parser.option.DilationHFactor(), parser.option.DilationWFactor()]
    weight_shape = parser.inputs_shape[1]
    # tflite weight-order is OHWI
    kernel_shape = weight_shape[1:3]

    pads = Parser.get_pads(input_shape, strides, kernel_shape, padding, None, dilation)

    # Setup convolution parameters: PADDINGS, DILATION, STRIDE, ...
    # ONNX defines auto_pad:: NOTSET, SAME_UPPER, SAME_LOWER or VALID
    # but we use pads::<list of ints> since the parser gets them calculated
    param_dict = dict(
        dilations=[parser.option.DilationHFactor(), parser.option.DilationWFactor()],
        kernel_shape=kernel_shape,
        pads=[pads[i] for i in [0, 2, 1, 3]],  # t, b, l, r -> t, l, b, r
        strides=strides,
    )
    generate_Conv_to_ONNX(parser, input_name, output_name, param_dict)



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

    # Assume axis = 0 (default for Cumsum unless otherwise specified)
    axis_value = 0

    # Create a tensor for axis value
    axes_name = output_name + "/axes"  # follow your naming style
    axes = np.array([axis_value]).astype(np.int64)  # ONNX expects int64

    parser.add_ndarray_to_tensor_dict(axes_name, axes)

    # Add ONNX CumSum node
    parser.add_onnx_operator(
        "CumSum",
        [input_name, axes_name],  # input and axes as inputs
        [output_name],
        {},  # no attributes
        ip_quant_params,
        op_quant_params
    )

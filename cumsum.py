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







from math import ceil

@staticmethod
def get_pads(input_shape, strides, kernel_shape, padding=0, output_shape=None, dilation=None):
    rank = len(input_shape)
    dilation = dilation if dilation else [1] * (rank - 2)
    stride_dims = strides
    kernel_dims = [
        dilation[i] * (kernel_shape[i] - 1) + 1 for i in range(len(kernel_shape))
    ]

    # Default pads: [pad_before_dim1, pad_after_dim1, pad_before_dim2, ..., pad_after_dimN]
    pads = [0] * (len(kernel_shape) * 2)

    if padding == 0:  # "SAME" padding
        spatial_dims = input_shape[1:-1]  # exclude batch and channels
        if output_shape:
            output_spatial_dims = output_shape[1:-1]
        else:
            output_spatial_dims = [
                ceil(float(spatial_dims[i]) / float(stride_dims[i])) for i in range(len(spatial_dims))
            ]

        for i in range(len(spatial_dims)):
            pad_along = max(
                (output_spatial_dims[i] - 1) * stride_dims[i] + kernel_dims[i] - spatial_dims[i], 0
            )
            pads[2 * i] = pad_along // 2
            pads[2 * i + 1] = pad_along - pads[2 * i]

    return pads





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






def add_ndarray_to_tensor_dict(self, name, ndarray):
        if name in self.tensor_dict.keys():
            value = self.tensor_dict[name]
            print("\n name, value", name, value)
            print("\n ndarray", ndarray)
            # exit()
            assert np.array_equal(
                value, ndarray
            ), "initializer {name} is shared but fails numpy.array_equal check"
        self.tensor_dict[name] = ndarray

@Converter.Register("RESIZE_NEAREST_NEIGHBOR")
def parse_RESIZE_NEAREST_NEIGHBOR(parser):
    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])
    # TFLite size only defines spatial dimensions
    # But ONNX:Resize requires whole tensor ND-shape
    # So we directly use infer_shapes
    out_size = parser.outputs_shape[0]
    scale = False  # only to test for using scale

    size_name = parser.get_tensor_name(parser.inputs[1])
    print("\n input0", parser.get_tensor_name(parser.inputs[0]))
    print("\n input1", parser.get_tensor_name(parser.inputs[1]))
    print("\n input1_shape", parser.inputs_shape[1])
    print("\n output", parser.outputs_shape[0])
    size_name = parser.get_tensor_name(parser.inputs[1])
    if scale:
        out_size = [m/n for m, n in zip(parser.outputs_shape[0], parser.inputs_shape[0])]
        parser.add_ndarray_to_tensor_dict(size_name, np.array(out_size).astype(np.float32))
    else:
        parser.add_ndarray_to_tensor_dict(size_name, np.array(out_size).astype(np.int64))

    align_corners = parser.option.AlignCorners()
    half_pixel_centers = parser.option.HalfPixelCenters()

    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()

    _mode = "nearest"
    if half_pixel_centers:
        _cord_trans = "tf_half_pixel_for_nn"  # FIXME: Not found in latest ONNX doc (but existed in Opset 11)
        node_attr = dict(coordinate_transformation_mode=_cord_trans, mode=_mode, nearest_mode="floor")
    elif align_corners:
        _cord_trans = "align_corners"
        node_attr = dict(coordinate_transformation_mode=_cord_trans, mode=_mode)
    else:
        _cord_trans = "asymmetric"  # default coordinate_transformation_mode
        node_attr = dict(coordinate_transformation_mode=_cord_trans, mode=_mode, nearest_mode="floor")

    # parser.add_ndarray_to_tensor_dict("empt", np.array([]).astype(np.float32))
    if scale:
        parser.add_onnx_operator(
            "Resize", [input_name, "", size_name], [output_name], node_attr, ip_quant_params, op_quant_params
        )
    else:
        parser.add_onnx_operator(
            "Resize", [input_name, "", "", size_name], [output_name], node_attr, ip_quant_params, op_quant_params
        )


error:

input0 322

 input1 2

 input1_shape [2]

 output [1, 128, 256, 32]

 input0 324

 input1 3

 input1_shape [2]

 output [1, 32, 64, 128]

 input0 329

 input1 2

 input1_shape [2]

 output [1, 128, 256, 64]

 name, value 2 [  1 128 256  32]

 ndarray [  1 128 256  64]
Traceback (most recent call last):
  File "/remote/us01sgnfs00562/NNSDK/harikam/har/nnac/frontend/nnac/deprecate_mwnnconvert", line 37, in <module>
    transform_manager.model_transform(flags)
  File "/remote/us01sgnfs00562/NNSDK/harikam/har/nnac/frontend/nnac/transform_manager.py", line 662, in model_transform
    model, data_format, output_dir = model_passes.convert(flags)
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/remote/us01sgnfs00562/NNSDK/harikam/har/nnac/frontend/nnac/transform_manager.py", line 164, in convert
    validate_pass, onnx_model, data_format, output_dir = converter(**vars(flags))
                                                         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/remote/us01sgnfs00562/NNSDK/harikam/har/nnac/frontend/nnac/converter/tflite2mwnn/tflite2mwnn_controller.py", line 93, in tflite2mwnn
    converted_onnxs, auto_use_index_for_name = converter.gen_onnx_model(
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/remote/us01sgnfs00562/NNSDK/harikam/har/nnac/frontend/nnac/converter/tflite2mwnn/gen_converter.py", line 491, in gen_onnx_model
    sub_name, auto_use_index_for_name = self.gen_onnx_model_one_subgraph(
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/remote/us01sgnfs00562/NNSDK/harikam/har/nnac/frontend/nnac/converter/tflite2mwnn/gen_converter.py", line 374, in gen_onnx_model_one_subgraph
    Converter.parsers[op_name](parser)
  File "/remote/us01sgnfs00562/NNSDK/harikam/har/nnac/frontend/nnac/converter/tflite2mwnn/gen_converter.py", line 2242, in parse_RESIZE_NEAREST_NEIGHBOR
    parser.add_ndarray_to_tensor_dict(size_name, np.array(out_size).astype(np.int64))
  File "/remote/us01sgnfs00562/NNSDK/harikam/har/nnac/frontend/nnac/converter/tflite2mwnn/gen_converter.py", line 921, in add_ndarray_to_tensor_dict
    assert np.array_equal(
           ^^^^^^^^^^^^^^^
AssertionError: initializer {name} is shared but fails numpy.array_equal check



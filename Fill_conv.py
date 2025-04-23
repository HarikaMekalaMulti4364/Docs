@Converter.Register("TANH")
def parse_TANH(parser):
    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])
    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()

    abs_output = parser.make_intermediate_tensor_name()
    denom = parser.make_intermediate_tensor_name()

    # Step 1: abs(x)
    parser.add_onnx_operator(
        "Abs",
        [input_name],
        [abs_output],
        input_quantization_params=ip_quant_params,
        output_quantization_params=ip_quant_params  # reuse same scale
    )

    # Step 2: 1 + abs(x)
    one_const = parser.make_constant_scalar("one_scalar", 1, dtype=parser.get_tensor_dtype(input_name))
    parser.add_onnx_operator(
        "Add",
        [one_const, abs_output],
        [denom],
        input_quantization_params=ip_quant_params,
        output_quantization_params=ip_quant_params
    )

    # Step 3: x / (1 + abs(x))
    parser.add_onnx_operator(
        "Div",
        [input_name, denom],
        [output_name],
        input_quantization_params=ip_quant_params,
        output_quantization_params=op_quant_params
    )





import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

@Converter.Register("CustomTanh")
def parse_CustomTanh(parser):
    input_name = parser.get_tensor_name(parser.inputs[0])  # input tensor
    output_name = parser.get_tensor_name(parser.outputs[0])  # output tensor
    ip_quant_params = parser._get_input_quantization_params()
    op_quant_params = parser._get_output_quantization_params()

    # Define scalar values for 27 and 9 (constants in the formula)
    scalar_27_value = 27.0
    scalar_9_value = 9.0

    # Create tensors for constants (27 and 9)
    add_27_tensor = np.array([scalar_27_value], dtype=np.float32)
    add_9_tensor = np.array([scalar_9_value], dtype=np.float32)

    add_27_attr = helper.make_tensor(
        name="add_27",  # name is required but arbitrary
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=add_27_tensor
    )

    add_9_attr = helper.make_tensor(
        name="add_9",  # name is required but arbitrary
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=add_9_tensor
    )

    # Create ONNX operations
    x_square = helper.make_node("Mul", [input_name, input_name], ["x_square"])
    add_27_x_square = helper.make_node("Add", ["x_square", "add_27"], ["num"])
    mul_x_num = helper.make_node("Mul", [input_name, "num"], ["num"])
    add_27_9_x_square = helper.make_node("Add", ["x_square", "add_9"], ["denom"])
    div_num_denom = helper.make_node("Div", ["num", "denom"], [output_name])

    # Add operations to the graph
    parser.add_onnx_operator(
        "Constant",
        [],  # no input tensors
        ["add_27"],
        {"value": add_27_attr},
        ip_quant_params,
        op_quant_params
    )

    parser.add_onnx_operator(
        "Constant",
        [],  # no input tensors
        ["add_9"],
        {"value": add_9_attr},
        ip_quant_params,
        op_quant_params
    )

    # Add Mul, Add, Div nodes to the graph
    parser.add_onnx_operator(
        "Mul",
        [input_name, input_name],
        ["x_square"],
        {},
        ip_quant_params,
        op_quant_params
    )

    parser.add_onnx_operator(
        "Add",
        ["x_square", "add_27"],
        ["num"],
        {},
        ip_quant_params,
        op_quant_params
    )

    parser.add_onnx_operator(
        "Mul",
        [input_name, "num"],
        ["num"],
        {},
        ip_quant_params,
        op_quant_params
    )

    parser.add_onnx_operator(
        "Add",
        ["x_square", "add_9"],
        ["denom"],
        {},
        ip_quant_params,
        op_quant_params
    )

    parser.add_onnx_operator(
        "Div",
        ["num", "denom"],
        [output_name],
        {},
        ip_quant_params,
        op_quant_params
    )




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

trans

def test_TRANSPOSE_CONV_quantize(self):
        out_dir = self.generate_out_dir()
        filename = os.path.join(out_dir, "conv2d_transpose_quantize.tflite")
        convert_filename = os.path.join(out_dir, "conv2d_transpose_quantize.onnx")

        class CustomModel(tf.Module):
            def __init__(self, name):
                super().__init__(name=name)

            @tf.function(input_signature=[tf.TensorSpec([1, 64, 64, 3], tf.float32)])
            def __call__(self, x):
                stride = 2
                kernel_size = 5
                output_channels = 64
                input_channels = 3
                weights = tf.constant(1.0, shape=[kernel_size, kernel_size, output_channels, input_channels])
                output_shape = [1, 128, 128, 64]  # x.shape[0], x.shape[1] * stride, x.shape[2]*stride, output_channels
                y = tf.nn.conv2d_transpose(input=x,
                                           filters=weights,
                                           output_shape=output_shape,
                                           strides=[1, stride, stride, 1],
                                           padding="SAME",
                                           data_format='NHWC',
                                           dilations=None,
                                           name=None)
                return y

        TRANSPOSE_CONV = CustomModel("test")
        self.convert_saved_model(TRANSPOSE_CONV, filename, quantize=True, input_shape=(1, 64, 64, 3))

        is_pass, model_def, _, _ = tflite2mwnn(filename)
        if not self.savespace:
            onnx.save(model_def, convert_filename)
        self.assertTrue(is_pass)

@Converter.Register("TRANSPOSE_CONV")
def parse_TRANSPOSE_CONV(parser):
    """ONNX: ConvTranspose
    If output_shape is specified pads values are ignored.
    So we need stride and dilation, and paddings (if not given output_shape)
    https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose
    """
    input_name = parser.get_tensor_name(parser.inputs[2])
    output_name = parser.get_tensor_name(parser.outputs[0])
    perm_to_NCHW = input_name + "/NHWC_to_NCHW"
    conv_out = output_name + "/convTranspose"
    # perm_to_NHWC = output_name + "/NCHW2NHWC"
    ipqp = parser._get_input_quantization_params()
    if len(ipqp) < 4:  # bias is absent
        ipqp.append(None)
        bias_name = ""
    else:
        bias_name = parser.write_constant_tensor_to_initializer(parser.inputs[3])
    # reorder [output_shape, weights, input, bias] to [X, W, B]
    ip_quant_params = [ipqp[i] for i in [2, 1, 3]]
    op_quant_params = parser._get_output_quantization_params()
    parser.add_onnx_permute_layer(
        input_name, perm_to_NCHW, Parser.NHWC_TO_NCHW, ip_quant_params[0]
    )
    # TFlite weight: OHWI
    # ONNX weight: IOHW
    weight_name = parser.write_constant_tensor_to_initializer(
        parser.inputs[1], kernel_order=[3, 0, 1, 2]
    )
    # bias
    # output_shape
    output_shape = list(parser.get_constant_node(parser.inputs[0]))

    # TODO: If output_shape is not given, may need to handle padding as well
    if not output_shape:  # is [] for dynamic output_shape
        output_shape = parser.outputs_shape[0]  # Use infer shapes of output tensor
    # Note: If output_shape is dynamic fetched from output tensor, users need to make sure the strides etc. is matching

    # stride and dilation
    strides = [parser.option.StrideH(), parser.option.StrideW()]
    # dilations = [parser.option.DilationHFactor(), parser.option.DilationWFactor()]
    # dilation seems to be deprecated in tflite? Anyway, ONNX takes [dH=1,dW=1] by default
    # dilations = [1, 1]
    # group=1
    # weight_shape = parser.inputs_shape[1]
    # kernel_shape = [weight_shape[1], weight_shape[2]]
    # padding = parser.option.Padding()
    # input_shape = parser.inputs_shape[2]
    # pads = Parser.get_pads(
    #    output_shape, strides, kernel_shape, padding, input_shape, dilations
    # )
    output_shape = [output_shape[i] for i in Parser.NHWC_TO_NCHW]

    attr_dict = dict(output_shape=output_shape, strides=strides)
    if output_shape is not None:
        attr_dict["auto_pad"] = "SAME_UPPER"

    #     # tried to append more attributes, but infer_shapes still fails ConvTranspose
    #     attr_dict = dict(dilations=dilations, kernel_shape=kernel_shape,
    #                      output_shape=output_shape,
    #                      pads=pads, strides=strides)
    onnx_inputs = [perm_to_NCHW, weight_name]
    if bias_name != "":
        onnx_inputs.append(bias_name)
    parser.add_onnx_operator(
        "ConvTranspose",
        onnx_inputs,
        [conv_out],
        attr_dict,
        ip_quant_params,
        op_quant_params,
        input_quant_axis=[0, 1]  # TransposeConv Weight has the quant axis=1
    )

    # parser.add_onnx_permute_layer(conv_out, output_name, Parser.NCHW_TO_NHWC)
    generate_activation_transpose(
        parser, conv_out, output_name, parser.activation_function, op_quant_params[0]
    )

def test_BatchToSpaceND(self):
        out_dir = self.generate_out_dir()
        filename = os.path.join(out_dir, "BatchToSpaceND.tflite")
        convert_filename = os.path.join(out_dir, "BatchToSpaceND.onnx")

        class CustomModel(tf.Module):
            def __init__(self, name):
                super().__init__(name=name)

            @tf.function(input_signature=[tf.TensorSpec([4, 1, 1, 1], tf.int32)])
            def __call__(self, x):
                y = tf.raw_ops.BatchToSpaceND(input=x, block_shape=[2, 2], crops=[[0, 0], [0, 0]])
                return y

        BatchToSpaceND = CustomModel("test")
        self.convert_saved_model(BatchToSpaceND, filename)

        is_pass, model_def, _, _ = tflite2mwnn(filename)
        if not self.savespace:
            onnx.save(model_def, convert_filename)
        self.assertTrue(is_pass)



@Converter.Register("BATCH_TO_SPACE_ND")
def parse_BATCH_TO_SPACE_ND(parser):
    """
    Plan: (4,1,1,1)
        → reshape → (2,2,1,1,1)        # batch → [block_H, block_W, N, H, W, C] style
        → transpose → (1,1,2,2,1)     # reordering to align blocks spatially
        → reshape → (1,2,2,1)         # flatten to expected output shape
    ->reshape -> transpose -> reshape
    """
    input_name = parser.get_tensor_name(parser.inputs[0])      # should be "x"
    output_name = parser.get_tensor_name(parser.outputs[0])    # should be "output"

    # Step 1: Reshape [4,1,1,1] → [2,2,1,1,1]
    reshaped1 = input_name + "_reshape1"
    reshape1_shape = reshaped1 + "_shape"
    parser._write_list_to_initializer([2, 2, 1, 1, 1], reshape1_shape)
    parser.add_onnx_operator("Reshape", [input_name, reshape1_shape], [reshaped1])

    # Step 2: Transpose [2,2,1,1,1] → [1,1,2,2,1] using perm [2, 3, 0, 1, 4]
    transposed = output_name + "_transposed"
    parser.add_onnx_permute_layer(reshaped1, transposed, [2, 3, 0, 1, 4])

    # Step 3: Reshape [1,1,2,2,1] → [1,2,2,1]
    final_reshape = output_name
    final_shape_name = final_reshape + "_shape"
    parser._write_list_to_initializer([1, 2, 2, 1], final_shape_name)
    parser.add_onnx_operator("Reshape", [transposed, final_shape_name], [final_reshape])



##elu
def test_ELU(self):
    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "ELU.tflite")
    convert_filename = os.path.join(out_dir, "ELU.onnx")

    class CustomModel(tf.Module):
        def __init__(self, name=None):
            super().__init__(name=name)

        @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32)])
        def __call__(self, x):
            return tf.nn.elu(x)  # TFLite ELU

    model = CustomModel("elu_model")
    self.convert_saved_model(model, filename)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)
@Converter.Register("ELU")
def parse_ELU(parser):
    """
    Converts TFLite ELU to ONNX Elu operator.
    TFLite ELU is equivalent to ONNX Elu with default alpha=1.0
    """
    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])

    # Add ELU node with default alpha=1.0 (or configurable if available)
    parser.add_onnx_operator("Elu", [input_name], [output_name], attrs={"alpha": 1.0}

#FILL
@Converter.Register("FILL")
def parse_FILL(parser):
    # input_name = parser.get_tensor_name(parser.inputs[0])
    input_name = parser.write_inputs_to_onnx_net()[0]
    if parser.dtype_to_numpy_type[parser.inputs[0].Type()] != np.int64:
        cast_name = input_name + "/cast_to_int64"
        parser.add_onnx_operator(
            "Cast", [input_name], [cast_name], attr_dict={"to": TensorProto.INT64}
        )
        input_name = cast_name
    output_name = parser.get_tensor_name(parser.outputs[0])
    filled_value = parser.get_constant_node(parser.inputs[1])
    filled_value = filled_value.flatten()  # np scalar will become tensor of shape [1]
    filled_value_name = output_name + "/fill_param"
    filled_value_tensor = numpy_helper.from_array(filled_value, filled_value_name)
    # filled_value_tensor = onnx.helper.make_tensor(filled_value_name, onnx.TensorProto.INT64, [1], [0])
    attr_dict = {"value": filled_value_tensor}

    parser.add_onnx_operator("ConstantOfShape", [input_name], [output_name], attr_dict)


def test_Fill(self):
    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "Fill.tflite")
    convert_filename = os.path.join(out_dir, "Fill.onnx")

    class FillModel(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    @tf.function(input_signature=[
        tf.TensorSpec([2], tf.int32),         # shape input
        tf.TensorSpec([], tf.float32)         # scalar value input as tensor
    ])
    def __call__(self, shape, value):
        return tf.raw_ops.Fill(dims=shape, value=value)


    model = FillModel("test")
    self.convert_saved_model(model, filename)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)

#hardswish
@Converter.Register("HARD_SWISH")
def parse_HARD_SWISH(parser):
    input_name = parser.get_tensor_name(parser.inputs[0])
    output_name = parser.get_tensor_name(parser.outputs[0])

    parser.add_onnx_operator("HardSwish", [input_name], [output_name])

def test_HardSwish(self):
    out_dir = self.generate_out_dir()
    filename = os.path.join(out_dir, "HardSwish.tflite")
    convert_filename = os.path.join(out_dir, "HardSwish.onnx")

    class CustomModel(tf.Module):
        def __init__(self, name):
            super().__init__(name=name)

        @tf.function(input_signature=[tf.TensorSpec([4, 1, 1, 1], tf.float32)])
        def __call__(self, x):
            y = x * tf.nn.relu6(x + 3) / 6.0
            return y

    HardSwish = CustomModel("test")
    self.convert_saved_model(HardSwish, filename)

    is_pass, model_def, _, _ = tflite2mwnn(filename)
    if not self.savespace:
        onnx.save(model_def, convert_filename)
    self.assertTrue(is_pass)

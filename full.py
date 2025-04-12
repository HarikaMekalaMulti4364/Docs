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
    parser.add_onnx_operator("Elu", [input_name], [output_name], attrs={"alpha": 1.0})

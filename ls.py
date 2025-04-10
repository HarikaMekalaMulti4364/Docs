# UNIDIRECTIONAL_SEQUENCE_LSTM
    # FIXME: onnx conversion error
    # FAIL : Node (tfl.unidirectional_sequence_lstm/SlBsHs) Op (Squeeze)
    # [ShapeInferenceError] Dimension of input 2 must be 1 instead of 32
    def test_LSTM(self):
        out_dir = self.generate_out_dir()
        filename = os.path.join(out_dir, "LSTM.tflite")
        convert_filename = os.path.join(out_dir, "LSTM.onnx")

        class CustomModel(tf.Module):
            def __init__(self, name):
                super().__init__(name=name)
                self.LSTM = tf.keras.layers.LSTM(4)

            @tf.function(input_signature=[tf.TensorSpec([32, 10, 8], tf.float32)])
            def __call__(self, x):
                y = self.LSTM(x)
                return y

        LSTM = CustomModel("test")
        self.convert_saved_model(LSTM, filename)
        is_pass, model_def, _, _ = tflite2mwnn(filename)
        if not self.savespace:
            onnx.save(model_def, convert_filename)
        self.assertTrue(is_pass)

@Converter.Register("UNIDIRECTIONAL_SEQUENCE_LSTM")
def parse_UNIDIRECTIONAL_SEQUENCE_LSTM(parser):
    # input_names = [parser.get_tensor_name(tsr) for tsr in parser.inputs]
    input_names = parser.write_inputs_to_onnx_net()
    output_name = parser.get_tensor_name(parser.outputs[0])
    # ip_quant_params = parser._get_input_quantization_params()
    # op_quant_params = parser._get_output_quantization_params()
    WRB_names = []
    for _name in ["W", "R", "B"]:
        WRB_names.append(output_name + "/" + _name)
    input_reorder = [0, 3, 1, 2]  # tflite: ifco, ONNX: iofc
    # Weight iofc
    parser.add_onnx_operator(
        "Concat",
        [input_names[1 + idx] for idx in input_reorder],
        [WRB_names[0] + "/2D"],
        {"axis": 0},
    )
    parser.add_ndarray_to_tensor_dict("const_0", np.array([0]).astype(np.int64))
    parser.add_onnx_operator(
        "Unsqueeze", [WRB_names[0] + "/2D", "const_0"], [WRB_names[0]]
    )
    # Recurrence weight iofc
    parser.add_onnx_operator(
        "Concat",
        [input_names[5 + idx] for idx in input_reorder],
        [WRB_names[1] + "/2D"],
        {"axis": 0},
    )
    parser.add_onnx_operator(
        "Unsqueeze", [WRB_names[1] + "/2D", "const_0"], [WRB_names[1]]
    )
    RB_names = [output_name + "/" + _name for _name in ["RBi", "RBf", "RBc", "RBo"]]
    all_zero_bias = np.zeros(parser.inputs_shape[9]).astype(np.float32)
    for name in RB_names:
        parser.add_ndarray_to_tensor_dict(name, all_zero_bias)
    parser.add_onnx_operator(
        "Concat",
        [input_names[9 + idx] for idx in input_reorder] + RB_names,
        [WRB_names[2] + "/1D"],
        {"axis": -1},
    )
    parser.add_onnx_operator(
        "Unsqueeze", [WRB_names[2] + "/1D", "const_0"], [WRB_names[2]]
    )
    for ip in input_names[-2:]:
        #         # initial_h and initial_c are optional
        #         assert initial_h.size == 0 and initial_c.size == 0, "
        #         FIXME: tflite variable gives only empty tensors during the development of tflite2mwnn converter."
        node_dict = parser.node_graph.nodes.get(ip)
        if node_dict is not None:
            # DQ node
            ip = node_dict["input"][0]
            # FIXME: in sonova benchmark model it uses int16
            parser.add_ndarray_to_tensor_dict(
                ip, all_zero_bias.reshape([1, 1, -1]).astype(np.int32)
            )
        else:
            parser.add_ndarray_to_tensor_dict(ip, all_zero_bias.reshape([1, 1, -1]))
    #    W,R,B,h,c
    tensors_names = WRB_names + input_names[-2:]
    #     tensors_vals  = [W_tensor, R_tensor, B_tensor, initial_h, initial_c]

    # 3. Generate -> Transpose -> LSTM -> Squeeze -> Transpose
    #    ORT layout issue:
    #    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/rnn/lstm_base.h#L51
    #    inputs=[X, W, R, B, sequence_lens(optional), initial_h, initial_c]
    data_name = input_names[0]
    squeeze_out_name = output_name
    onnx_layout = 0
    if onnx_layout == 0:
        # tflite layout is ONNX<1>, but ORT doesn't support it.
        # Transpose (batch,seq_len,hidden_size) to [s,b,h]
        d_name = data_name + "/layout0"
        parser.add_onnx_operator(
            "Transpose", [data_name], [d_name], {"perm": [1, 0, 2]}
        )
        data_name = d_name
        squeeze_out_name = output_name + "/SlBsHs"
    # data - Transpose - LSTM - Transpose - Squeeze - output
    onnx_ip_names = [data_name, *tensors_names[:-2], "", *tensors_names[-2:]]
    # tflite UNIDIRECTIONAL_SEQUENCE_LSTM shall have some attributes
    # e.g. parser.option.AsymmetricQuantizeInputs, CellClip, FusedActivationFunction
    # But the customer model doesn't. So I skipped them in the conversion;
    # surprisingly, ONNX default attributes match the tflite computation.
    attr_dict = {
        # "activations": ["Sigmoid","Tanh","Tanh"],
        # "clip": parser.option.CellClip(),
        "direction": "forward",
        "hidden_size": all_zero_bias.size,
        "layout": onnx_layout,
    }
    parser.add_onnx_operator(
        "LSTM",
        onnx_ip_names,
        [output_name + "_o", output_name + "_h", output_name + "_c"],
        attr_dict,
    )
    # Because ONNX LSTM outputs [], squeeze out the bidirection_axis
    parser.add_ndarray_to_tensor_dict("const_2", np.array([2]).astype(np.int64))
    parser.add_onnx_operator(
        "Squeeze", [output_name + "_o", "const_2"], [squeeze_out_name]
    )

    if onnx_layout == 0:
        parser.add_onnx_operator(
            "Transpose", [squeeze_out_name], [output_name], {"perm": [1, 0, 2]}
        )





next:
@Converter.Register("UNIDIRECTIONAL_SEQUENCE_LSTM")
def parse_UNIDIRECTIONAL_SEQUENCE_LSTM(parser):
    input_names = parser.write_inputs_to_onnx_net()
    output_name = parser.get_tensor_name(parser.outputs[0])

    WRB_names = []
    for _name in ["W", "R", "B"]:
        WRB_names.append(output_name + "/" + _name)

    input_reorder = [0, 3, 1, 2]  # ifco → iofc

    # ---- Create dummy W, R, B tensors ----
    hidden_size = 4
    input_size = 8
    batch_size = 32
    seq_len = 10
    all_zero_bias = np.zeros(hidden_size, dtype=np.float32)

    # Create W
    W_tensor = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/W_data", W_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/W_data", "const_0"],
        [WRB_names[0]]
    )

    # Create R
    R_tensor = np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/R_data", R_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/R_data", "const_0"],
        [WRB_names[1]]
    )

    # Create B
    B_tensor = np.random.rand(1, 8 * hidden_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/B_data", B_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/B_data", "const_0"],
        [WRB_names[2]]
    )

    # ---- initial_h and initial_c ----
    initial_h = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    initial_c = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_h", initial_h)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_c", initial_c)

    tensors_names = WRB_names + [output_name + "/initial_h", output_name + "/initial_c"]

    # ---- Transpose layout ----
    data_name = input_names[0]
    squeeze_out_name = output_name
    onnx_layout = 0
    if onnx_layout == 0:
        transposed_input = data_name + "/layout0"
        parser.add_onnx_operator(
            "Transpose", [data_name], [transposed_input], {"perm": [1, 0, 2]}
        )
        data_name = transposed_input
        squeeze_out_name = output_name + "/SlBsHs"

    # ---- ONNX LSTM ----
    onnx_ip_names = [data_name, *tensors_names[:-2], "", *tensors_names[-2:]]
    parser.add_onnx_operator(
        "LSTM",
        onnx_ip_names,
        [output_name + "_o", output_name + "_h", output_name + "_c"],
        {
            "direction": "forward",
            "hidden_size": hidden_size,
            "layout": onnx_layout,
        },
    )

    parser.add_ndarray_to_tensor_dict("const_2", np.array([2]).astype(np.int64))
    parser.add_onnx_operator(
        "Squeeze", [output_name + "_o", "const_2"], [squeeze_out_name]
    )

    if onnx_layout == 0:
        parser.add_onnx_operator(
            "Transpose", [squeeze_out_name], [output_name], {"perm": [1, 0, 2]}
        )


next1
@Converter.Register("UNIDIRECTIONAL_SEQUENCE_LSTM")
def parse_UNIDIRECTIONAL_SEQUENCE_LSTM(parser):
    input_names = parser.write_inputs_to_onnx_net()
    output_name = parser.get_tensor_name(parser.outputs[0])

    WRB_names = []
    for _name in ["W", "R", "B"]:
        WRB_names.append(output_name + "/" + _name)

    input_reorder = [0, 3, 1, 2]  # ifco → iofc

    # ---- Create dummy W, R, B tensors ----
    hidden_size = 4
    input_size = 8
    batch_size = 32
    seq_len = 10
    all_zero_bias = np.zeros(hidden_size, dtype=np.float32)

    # Define const_0 before using it
    parser.add_ndarray_to_tensor_dict("const_0", np.array([0]).astype(np.int64))

    # Create W
    W_tensor = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/W_data", W_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/W_data", "const_0"],
        [WRB_names[0]]
    )

    # Create R
    R_tensor = np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/R_data", R_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/R_data", "const_0"],
        [WRB_names[1]]
    )

    # Create B
    B_tensor = np.random.rand(1, 8 * hidden_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/B_data", B_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/B_data", "const_0"],
        [WRB_names[2]]
    )

    # ---- initial_h and initial_c ----
    initial_h = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    initial_c = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_h", initial_h)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_c", initial_c)

    tensors_names = WRB_names + [output_name + "/initial_h", output_name + "/initial_c"]

    # ---- Transpose layout ----
    data_name = input_names[0]
    squeeze_out_name = output_name
    onnx_layout = 0
    if onnx_layout == 0:
        transposed_input = data_name + "/layout0"
        parser.add_onnx_operator(
            "Transpose", [data_name], [transposed_input], {"perm": [1, 0, 2]}
        )
        data_name = transposed_input
        squeeze_out_name = output_name + "/SlBsHs"

    # ---- ONNX LSTM ----
    onnx_ip_names = [data_name, *tensors_names[:-2], "", *tensors_names[-2:]]
    parser.add_onnx_operator(
        "LSTM",
        onnx_ip_names,
        [output_name + "_o", output_name + "_h", output_name + "_c"],
        {
            "direction": "forward",
            "hidden_size": hidden_size,
            "layout": onnx_layout,
        },
    )

    # Define const_2 before using it
    parser.add_ndarray_to_tensor_dict("const_2", np.array([2]).astype(np.int64))
    parser.add_onnx_operator(
        "Squeeze", [output_name + "_o", "const_2"], [squeeze_out_name]
    )

    if onnx_layout == 0:
        parser.add_onnx_operator(
            "Transpose", [squeeze_out_name], [output_name], {"perm": [1, 0, 2]}
        )


next2
@Converter.Register("UNIDIRECTIONAL_SEQUENCE_LSTM")
def parse_UNIDIRECTIONAL_SEQUENCE_LSTM(parser):
    input_names = parser.write_inputs_to_onnx_net()
    output_name = parser.get_tensor_name(parser.outputs[0])

    WRB_names = []
    for _name in ["W", "R", "B"]:
        WRB_names.append(output_name + "/" + _name)

    input_reorder = [0, 3, 1, 2]  # ifco → iofc

    # ---- Create dummy W, R, B tensors ----
    hidden_size = 4
    input_size = 8
    batch_size = 32
    seq_len = 10

    # Create W
    W_tensor = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/W_data", W_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/W_data", "const_0"],
        [WRB_names[0]]
    )

    # Create R
    R_tensor = np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/R_data", R_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/R_data", "const_0"],
        [WRB_names[1]]
    )

    # Create B
    B_tensor = np.random.rand(1, 8 * hidden_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/B_data", B_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/B_data", "const_0"],
        [WRB_names[2]]
    )

    # ---- initial_h and initial_c ----
    initial_h = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    initial_c = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_h", initial_h)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_c", initial_c)

    tensors_names = WRB_names + [output_name + "/initial_h", output_name + "/initial_c"]

    # ---- Transpose layout ----
    data_name = input_names[0]
    squeeze_out_name = output_name
    onnx_layout = 0
    if onnx_layout == 0:
        transposed_input = data_name + "/layout0"
        parser.add_onnx_operator(
            "Transpose", [data_name], [transposed_input], {"perm": [1, 0, 2]}
        )
        data_name = transposed_input
        squeeze_out_name = output_name + "/SlBsHs"

    # ---- ONNX LSTM ----
    onnx_ip_names = [data_name, *tensors_names[:-2], "", *tensors_names[-2:]]
    parser.add_onnx_operator(
        "LSTM",
        onnx_ip_names,
        [output_name + "_o", output_name + "_h", output_name + "_c"],
        {
            "direction": "forward",
            "hidden_size": hidden_size,
            "layout": onnx_layout,
        },
    )

    # Removed invalid Squeeze (was causing dimension error)
    squeeze_out_name = output_name + "_o"

    if onnx_layout == 0:
        parser.add_onnx_operator(
            "Transpose", [squeeze_out_name], [output_name], {"perm": [1, 0, 2]}
        )

next3
@Converter.Register("UNIDIRECTIONAL_SEQUENCE_LSTM")
def parse_UNIDIRECTIONAL_SEQUENCE_LSTM(parser):
    input_names = parser.write_inputs_to_onnx_net()
    output_name = parser.get_tensor_name(parser.outputs[0])

    WRB_names = []
    for _name in ["W", "R", "B"]:
        WRB_names.append(output_name + "/" + _name)

    hidden_size = 4
    input_size = 8
    batch_size = 32

    # ✅ Declare constants first to ensure topological order
    parser.add_ndarray_to_tensor_dict("const_0", np.array([0], dtype=np.int64))

    # ---- Create dummy W, R, B tensors ----
    W_tensor = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/W_data", W_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/W_data", "const_0"],
        [WRB_names[0]]
    )

    R_tensor = np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/R_data", R_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/R_data", "const_0"],
        [WRB_names[1]]
    )

    B_tensor = np.random.rand(1, 8 * hidden_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/B_data", B_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [output_name + "/B_data", "const_0"],
        [WRB_names[2]]
    )

    # ---- initial_h and initial_c ----
    initial_h = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    initial_c = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_h", initial_h)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_c", initial_c)

    tensors_names = WRB_names + [output_name + "/initial_h", output_name + "/initial_c"]

    # ---- Transpose input layout if needed ----
    data_name = input_names[0]
    onnx_layout = 0  # batch-major input [batch, seq, feat]
    if onnx_layout == 0:
        transposed_input = data_name + "/layout0"
        parser.add_onnx_operator(
            "Transpose", [data_name], [transposed_input], {"perm": [1, 0, 2]}
        )
        data_name = transposed_input

    # ---- LSTM operator ----
    onnx_input_names = [data_name, *tensors_names[:-2], "", *tensors_names[-2:]]
    lstm_output = output_name + "_o"
    parser.add_onnx_operator(
        "LSTM",
        onnx_input_names,
        [lstm_output, output_name + "_h", output_name + "_c"],
        {
            "direction": "forward",
            "hidden_size": hidden_size,
            "layout": onnx_layout,
        },
    )

    # ✅ No invalid squeeze applied
    # Optionally transpose back if layout was changed
    if onnx_layout == 0:
        parser.add_onnx_operator(
            "Transpose", [lstm_output], [output_name], {"perm": [1, 0, 2]}
        )

next4
@Converter.Register("UNIDIRECTIONAL_SEQUENCE_LSTM")
def parse_UNIDIRECTIONAL_SEQUENCE_LSTM(parser):
    input_names = parser.write_inputs_to_onnx_net()
    output_name = parser.get_tensor_name(parser.outputs[0])

    hidden_size = 4
    input_size = 8
    batch_size = 32

    # ✅ Declare const_0 before any usage
    parser.add_ndarray_to_tensor_dict("const_0", np.array([0], dtype=np.int64))

    # ---- Create W, R, B ----
    WRB = {
        "W": np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32),
        "R": np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32),
        "B": np.random.rand(1, 8 * hidden_size).astype(np.float32),
    }

    WRB_names = []
    for name, tensor in WRB.items():
        tensor_name = f"{output_name}/{name}_data"
        parser.add_ndarray_to_tensor_dict(tensor_name, tensor)
        output_tensor = f"{output_name}/{name}"
        parser.add_onnx_operator("Unsqueeze", [tensor_name, "const_0"], [output_tensor])
        WRB_names.append(output_tensor)

    # ---- initial_h and initial_c ----
    init_h = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    init_c = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_h", init_h)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_c", init_c)

    # ---- Transpose input if using layout=0 ----
    data_name = input_names[0]
    onnx_layout = 0
    if onnx_layout == 0:
        transposed_input = f"{data_name}/layout0"
        parser.add_onnx_operator(
            "Transpose", [data_name], [transposed_input], {"perm": [1, 0, 2]}
        )
        data_name = transposed_input

    # ---- LSTM operation ----
    onnx_inputs = [data_name, *WRB_names, "", output_name + "/initial_h", output_name + "/initial_c"]
    parser.add_onnx_operator(
        "LSTM",
        onnx_inputs,
        [output_name + "_o", output_name + "_h", output_name + "_c"],
        {
            "direction": "forward",
            "hidden_size": hidden_size,
            "layout": onnx_layout,
        },
    )

    # ✅ Avoid squeeze unless you're certain about the shape
    # Post-process output if layout was changed
    if onnx_layout == 0:
        parser.add_onnx_operator(
            "Transpose", [output_name + "_o"], [output_name], {"perm": [1, 0, 2]}
        )

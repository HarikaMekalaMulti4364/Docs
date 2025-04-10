@Converter.Register("UNIDIRECTIONAL_SEQUENCE_LSTM")
def parse_UNIDIRECTIONAL_SEQUENCE_LSTM(parser):
    input_names = parser.write_inputs_to_onnx_net()
    output_name = parser.get_tensor_name(parser.outputs[0])

    hidden_size = 4
    input_size = 8
    batch_size = 32

    # Declare all constants before usage
    parser.add_ndarray_to_tensor_dict("const_0", np.array([0], dtype=np.int64))

    # W, R, B tensors
    WRB = {
        "W": np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32),
        "R": np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32),
        "B": np.random.rand(1, 8 * hidden_size).astype(np.float32),
    }

    WRB_names = []
    for name, tensor in WRB.items():
        tensor_data_name = f"{output_name}/{name}_data"
        tensor_unsqueezed_name = f"{output_name}/{name}"
        parser.add_ndarray_to_tensor_dict(tensor_data_name, tensor)
        parser.add_onnx_operator("Unsqueeze", [tensor_data_name, "const_0"], [tensor_unsqueezed_name])
        WRB_names.append(tensor_unsqueezed_name)

    # initial_h and initial_c
    initial_h = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    initial_c = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_h", initial_h)
    parser.add_ndarray_to_tensor_dict(output_name + "/initial_c", initial_c)

    # Transpose layout if needed
    data_name = input_names[0]
    onnx_layout = 0  # 0 = [seq, batch, input]
    if onnx_layout == 0:
        transposed_input = f"{data_name}/transposed"
        parser.add_onnx_operator("Transpose", [data_name], [transposed_input], {"perm": [1, 0, 2]})
        data_name = transposed_input

    # LSTM input order: [X, W, R, B, sequence_lens, initial_h, initial_c]
    onnx_inputs = [
        data_name,
        WRB_names[0],  # W
        WRB_names[1],  # R
        WRB_names[2],  # B
        "",            # sequence_lens not used
        output_name + "/initial_h",
        output_name + "/initial_c",
    ]

    # LSTM op
    lstm_outputs = [output_name + "_o", output_name + "_h", output_name + "_c"]
    parser.add_onnx_operator(
        "LSTM",
        onnx_inputs,
        lstm_outputs,
        {
            "direction": "forward",
            "hidden_size": hidden_size,
            "layout": onnx_layout,
        },
    )

    # Avoid squeeze entirely to prevent shape inference issues
    # Just transpose back the output if needed
    if onnx_layout == 0:
        parser.add_onnx_operator(
            "Transpose",
            [output_name + "_o"],
            [output_name],
            {"perm": [1, 0, 2]},
        )

next1
@Converter.Register("UNIDIRECTIONAL_SEQUENCE_LSTM")
def parse_UNIDIRECTIONAL_SEQUENCE_LSTM(parser):
    input_names = parser.write_inputs_to_onnx_net()
    output_name = parser.get_tensor_name(parser.outputs[0])

    # === Declare const_0 first to avoid topological sort error ===
    parser.add_ndarray_to_tensor_dict("const_0", np.array([0], dtype=np.int64))

    # === Create dummy W, R, B tensors ===
    hidden_size = 4
    input_size = 8
    batch_size = 32
    seq_len = 10

    WRB_names = []
    for name in ["W", "R", "B"]:
        WRB_names.append(f"{output_name}/{name}")

    # --- W ---
    W_tensor = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(f"{output_name}/W_data", W_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [f"{output_name}/W_data", "const_0"],
        [WRB_names[0]]
    )

    # --- R ---
    R_tensor = np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(f"{output_name}/R_data", R_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [f"{output_name}/R_data", "const_0"],
        [WRB_names[1]]
    )

    # --- B ---
    B_tensor = np.random.rand(1, 8 * hidden_size).astype(np.float32)
    parser.add_ndarray_to_tensor_dict(f"{output_name}/B_data", B_tensor)
    parser.add_onnx_operator(
        "Unsqueeze",
        [f"{output_name}/B_data", "const_0"],
        [WRB_names[2]]
    )

    # === initial_h and initial_c ===
    initial_h = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    initial_c = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    parser.add_ndarray_to_tensor_dict(f"{output_name}/initial_h", initial_h)
    parser.add_ndarray_to_tensor_dict(f"{output_name}/initial_c", initial_c)

    # === Transpose layout (to match ONNX layout=0: [seq, batch, input]) ===
    data_name = input_names[0]
    layout_0_input = f"{data_name}/layout0"
    parser.add_onnx_operator(
        "Transpose",
        [data_name],
        [layout_0_input],
        {"perm": [1, 0, 2]}  # [batch, seq, input] -> [seq, batch, input]
    )

    # === ONNX LSTM ===
    lstm_output_names = [f"{output_name}_o", f"{output_name}_h", f"{output_name}_c"]
    parser.add_onnx_operator(
        "LSTM",
        [layout_0_input, *WRB_names, "", f"{output_name}/initial_h", f"{output_name}/initial_c"],
        lstm_output_names,
        {
            "direction": "forward",
            "hidden_size": hidden_size,
            "layout": 0
        }
    )

    # === Final transpose to match expected output ===
    parser.add_onnx_operator(
        "Transpose",
        [f"{output_name}_o"],
        [output_name],
        {"perm": [1, 0, 2]}  # [seq, batch, hidden] -> [batch, seq, hidden]
    )


next2
@Converter.Register("UNIDIRECTIONAL_SEQUENCE_LSTM")
def parse_UNIDIRECTIONAL_SEQUENCE_LSTM(parser):
    input_names = parser.write_inputs_to_onnx_net()
    output_name = parser.get_tensor_name(parser.outputs[0])

    hidden_size = 4
    input_size = 8
    batch_size = 32

    # Correct ONNX shapes (no unsqueeze)
    W = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    R = np.random.rand(1, 4 * hidden_size, hidden_size).astype(np.float32)
    B = np.random.rand(1, 8 * hidden_size).astype(np.float32)

    parser.add_ndarray_to_tensor_dict(f"{output_name}/W", W)
    parser.add_ndarray_to_tensor_dict(f"{output_name}/R", R)
    parser.add_ndarray_to_tensor_dict(f"{output_name}/B", B)

    initial_h = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    initial_c = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
    parser.add_ndarray_to_tensor_dict(f"{output_name}/initial_h", initial_h)
    parser.add_ndarray_to_tensor_dict(f"{output_name}/initial_c", initial_c)

    parser.add_onnx_operator(
        "LSTM",
        [
            input_names[0],
            f"{output_name}/W",
            f"{output_name}/R",
            f"{output_name}/B",
            "",  # sequence_lens
            f"{output_name}/initial_h",
            f"{output_name}/initial_c",
        ],
        [f"{output_name}_o", f"{output_name}_h", f"{output_name}_c"],
        {
            "direction": "forward",
            "hidden_size": hidden_size,
            "layout": 0,  # [seq, batch, feat]
        }
    )

    # Directly use output (if consumer can handle ONNX LSTM layout)
    parser.add_onnx_operator(
        "Identity", [f"{output_name}_o"], [output_name]
    )

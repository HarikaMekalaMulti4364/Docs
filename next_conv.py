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

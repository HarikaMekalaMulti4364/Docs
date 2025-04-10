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


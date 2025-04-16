def parse_floor_div(parser):
    # Write input names (broadcast enabled)
    input_names = parser.write_inputs_to_onnx_net(broadcast=True)
    
    # Get the output tensor name
    output_name = parser.get_tensor_name(parser.outputs[0])
    
    # Generate an intermediate output name for the Div operation
    intermediate_output_name = "Div_output_" + output_name

    # Insert a Div operation
    parser.add_onnx_operator("Div", input_names, [intermediate_output_name])
    
    # Insert a Cast to float before Floor (required to avoid TypeError)
    cast_output_name = intermediate_output_name + "_float"
    parser.add_onnx_operator("Cast", [intermediate_output_name], [cast_output_name], attrs={"to": 1})  # 1 = FLOAT
    
    # Insert the Floor operation
    parser.add_onnx_operator("Floor", [cast_output_name], [output_name])

import onnx

def remove_redundant_quant_dequant(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph

    input_map = {}  # Maps outputs to their original inputs
    nodes_to_remove = []

    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            # Store mapping and mark for removal
            input_map[node.output[0]] = node.input[0]
            nodes_to_remove.append(node)

        elif node.op_type == "DequantizeLinear":
            dequant_input = node.input[0]

            # If DequantizeLinear follows a QuantizeLinear, remove it
            if dequant_input in input_map:
                input_map[node.output[0]] = input_map[dequant_input]
                nodes_to_remove.append(node)
            else:
                input_map[node.output[0]] = node.output[0]  # Keep first occurrence

    # Update remaining node inputs
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in input_map:
                node.input[i] = input_map[inp]

    # Remove redundant nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    onnx.save(model, output_path)
    print(f"Optimized model saved to {output_path}")

# Usage
remove_redundant_quant_dequant("model.onnx", "optimized_model.onnx")

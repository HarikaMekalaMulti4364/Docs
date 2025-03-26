import onnx

def remove_redundant_quantization(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph

    nodes_to_remove = []
    input_map = {}

    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            # Store the input-output mapping of QuantizeLinear
            input_map[node.output[0]] = node.input[0]
            nodes_to_remove.append(node)
        elif node.op_type == "DequantizeLinear" and node.input[0] in input_map:
            # Redirect DequantizeLinear to use the original float32 input
            node.input[0] = input_map[node.input[0]]

    # Remove unnecessary QuantizeLinear nodes
    for node in nodes_to_remove:
        graph.node.remove(node)

    onnx.save(model, output_path)
    print(f"Optimized model saved to {output_path}")

# Usage
remove_redundant_quantization("model.onnx", "optimized_model.onnx")

import onnx
import numpy as np
from onnx import numpy_helper

def optimize_onnx_model(input_model_path, output_model_path):
    # Load the ONNX model
    model = onnx.load(input_model_path)
    graph = model.graph

    ### Step 1: Remove Sub + Mul ###
    sub_node = next((node for node in graph.node if node.op_type == "Sub"), None)
    mul_node = next((node for node in graph.node if node.op_type == "Mul"), None)

    if sub_node and mul_node:
        sub_input = sub_node.input[0]  # Image input
        mul_output = mul_node.output[0]  # Final output after Mul
        bias_tensor_name = sub_node.input[1]
        scale_tensor_name = mul_node.input[1]

        # Get Bias (B) and Scale (S)
        bias_tensor = next((t for t in graph.initializer if t.name == bias_tensor_name), None)
        scale_tensor = next((t for t in graph.initializer if t.name == scale_tensor_name), None)

        if bias_tensor and scale_tensor:
            B = numpy_helper.to_array(bias_tensor)
            S = numpy_helper.to_array(scale_tensor)
            new_bias = B * S  # Precompute the bias transformation

            # Create new bias tensor
            new_bias_tensor = numpy_helper.from_array(-new_bias, name="new_bias")
            graph.initializer.append(new_bias_tensor)

            # Replace Sub + Mul with a single Add
            new_add_node = onnx.helper.make_node("Add", inputs=[sub_input, "new_bias"], outputs=[mul_output])
            graph.node.append(new_add_node)

            # Remove Sub & Mul nodes
            graph.node.remove(sub_node)
            graph.node.remove(mul_node)
            print("✅ Removed Sub + Mul and replaced with optimized Add.")

    ### Step 2: Merge QuantizeLinear + Pad ###
    quant_node = next((node for node in graph.node if node.op_type == "QuantizeLinear"), None)
    pad_node = next((node for node in graph.node if node.op_type == "Pad"), None)

    if quant_node and pad_node:
        quant_output = quant_node.output[0]  # Output of QuantizeLinear
        pad_output = pad_node.output[0]  # Final output after Pad

        if quant_output == pad_node.input[0]:  # Ensure they're linked
            pad_node.input[0] = quant_node.input[0]  # Merge operations

            # Remove QuantizeLinear since Pad will directly take the input
            graph.node.remove(quant_node)
            print("✅ Merged QuantizeLinear into Pad.")

    # Save the optimized model
    onnx.save(model, output_model_path)
    print(f"🚀 Optimized model saved to {output_model_path}")

# Usage
input_model = "your_model.onnx"  # Replace with actual model path
output_model = "optimized_model.onnx"
optimize_onnx_model(input_model, output_model)

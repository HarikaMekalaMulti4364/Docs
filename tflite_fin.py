import tensorflow as tf
import numpy as np

def optimize_tflite_model(input_model_path, output_model_path):
    # Load the TFLite model
    with open(input_model_path, "rb") as f:
        model_buffer = f.read()

    # Load the model using TensorFlow Lite Interpreter
    interpreter = tf.lite.Interpreter(model_content=model_buffer)
    interpreter.allocate_tensors()

    # Get the TensorFlow Lite model details
    tensor_details = interpreter.get_tensor_details()
    sub_node, mul_node, quant_node, pad_node = None, None, None, None

    # Identify Sub, Mul, Quantize, and Pad nodes
    for tensor in tensor_details:
        if "Sub" in tensor["name"]:
            sub_node = tensor
        elif "Mul" in tensor["name"]:
            mul_node = tensor
        elif "Quantize" in tensor["name"]:
            quant_node = tensor
        elif "Pad" in tensor["name"]:
            pad_node = tensor

    ### Step 1: Remove Sub + Mul ###
    if sub_node and mul_node:
        input_tensor_index = sub_node["index"]
        sub_bias = interpreter.tensor(sub_node["index"])()
        mul_scale = interpreter.tensor(mul_node["index"])()

        # Precompute new bias (B * S)
        new_bias = -sub_bias * mul_scale

        # Replace Sub + Mul with a single Add
        interpreter.tensor(input_tensor_index)[:] += new_bias
        print("✅ Removed Sub + Mul and replaced with optimized Add.")

    ### Step 2: Merge Quantize and Pad ###
    if quant_node and pad_node:
        pad_input = pad_node["index"]
        quant_output = quant_node["index"]

        if quant_output == pad_input:  # Ensure they are linked
            pad_node["index"] = quant_node["index"]  # Merge them

            print("✅ Merged Quantize into Pad.")

    # Save the optimized model
    optimized_model = interpreter.get_tensor(0)  # Get modified tensor
    with open(output_model_path, "wb") as f:
        f.write(optimized_model)

    print(f"🚀 Optimized model saved to {output_model_path}")

# Usage
input_model = "your_model.tflite"  # Replace with actual model path
output_model = "optimized_model.tflite"
optimize_tflite_model(input_model, output_model)

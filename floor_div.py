import numpy as np
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    # Define the inputs
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [None])

    # Create the division operation
    div_node = helper.make_node("Div", ["X", "Y"], ["Div_Output"])
    floor_node = helper.make_node("Floor", ["Div_Output"], ["Z"])
    
    # Create the graph
    graph = helper.make_graph([div_node, floor_node], "FloorDivGraph", [X, Y], [Z])
    
    # Create the model
    model = helper.make_model(graph, producer_name="onnx_floordiv")
    return model

# Generate test data
x = np.array([7.5, -7.5, 8.9, -8.9], dtype=np.float32)
y = np.array([2.0, 2.0, -2.0, -2.0], dtype=np.float32)

# Compute TFLite-style floor division
expected_output = np.floor(x / y)

# Create ONNX model
onnx_model = create_onnx_model()
onnx.save(onnx_model, "floordiv.onnx")

# Run inference using ONNX Runtime
ort_session = ort.InferenceSession("floordiv.onnx")
onnx_output = ort_session.run(None, {"X": x, "Y": y})[0]

# Compare outputs
print("TFLite FloorDiv Output:", expected_output)
print("ONNX Equivalent Output:", onnx_output)
print("Are outputs equal?", np.allclose(expected_output, onnx_output))

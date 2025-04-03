import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np

# Define input and output tensor
input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 3])
output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 1])

# Define 'axes' as a tensor instead of an attribute
axes_tensor = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes")

# Define ReduceMax Node
reduce_max_node = helper.make_node(
    "ReduceMax",
    inputs=["input", "axes"],  # Now takes 'axes' as an input
    outputs=["output"],
    keepdims=1
)

# Create graph
graph = helper.make_graph(
    [reduce_max_node],
    "ReduceMaxGraph",
    [input_tensor],
    [output_tensor],
    [axes_tensor]  # Include 'axes' tensor in initializers
)

# Create model
model = helper.make_model(graph, producer_name="onnx-reducemax")
onnx.save(model, "reducemax_model.onnx")

print("✅ Fixed ONNX model saved as 'reducemax_model.onnx'")

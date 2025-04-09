input_name = parser.get_tensor_name(parser.inputs[0])                      # "x"
output_name = parser.get_tensor_name(parser.outputs[0])                   # "output"

trans1 = input_name + "/NHWC_to_NCHW"                                     # "x/NHWC_to_NCHW"
trans2 = input_name + "/NCHW_to_NHWC"                                     # "x/NCHW_to_NHWC"

D2S_name = output_name + "/CNHW"                                          # "output/CNHW"
trans3 = output_name + "/CNHW_to_NCHW"                                    # "output/CNHW_to_NCHW"
trans4 = output_name + "/NCHW_to_NHWC"                                    # "output/NCHW_to_NHWC"

crop_out = output_name                                                    # "output"

# NHWC -> NCHW
parser.add_onnx_permute_layer(input_name, trans1, parser.NHWC_TO_NCHW)

# block_shape assumed to be known or input[1]
block_shape = list(parser.get_constant_node(parser.inputs[1]))           # e.g., [2, 2]
assert block_shape[0] == block_shape[1]
block_size = block_shape[0]                                               # block_size = 2

# Apply DepthToSpace
parser.add_onnx_permute_layer(trans1, trans2, parser.NCHW_TO_NHWC)       # Permute to NHWC for DepthToSpace
parser.add_onnx_operator("DepthToSpace", [trans2], [D2S_name],
                         attr_dict=dict(blocksize=block_size))

# Back to NCHW for crop
parser.add_onnx_permute_layer(D2S_name, trans3, [1, 0, 2, 3])
parser.add_onnx_permute_layer(trans3, trans4, parser.NCHW_TO_NHWC)

# Prepare crop attributes
crops = list(parser.get_constant_node(parser.inputs[2]).flatten())       # [[0, 0], [0, 0]] → [0, 0, 0, 0]
axes = [1, 2]                                                             # cropping on H and W
begin = []
end = []

for i in range(0, len(crops), 2):
    begin.append(crops[i])                                               # e.g., 0
    end.append(-crops[i + 1])                                            # e.g., 0 → -0 → 0

steps = [1, 1]

axes_name = output_name + "crop_axes"                                    # "outputcrop_axes"
parser.write_list_to_initializer(axes, axes_name)

starts_name = output_name + "crop_begin"                                 # "outputcrop_begin"
parser.write_list_to_initializer(begin, starts_name)

ends_name = output_name + "crop_end"                                     # "outputcrop_end"
parser.write_list_to_initializer(end, ends_name)

steps_name = output_name + "crop_steps"                                  # "outputcrop_steps"
parser.write_list_to_initializer(steps, steps_name)

# Final crop (Slice)
parser.add_onnx_operator("Slice",
    [trans4, starts_name, ends_name, axes_name, steps_name],
    [crop_out])




2nd method
input_name = parser.get_tensor_name(parser.inputs[0])      # "x"
output_name = parser.get_tensor_name(parser.outputs[0])    # "output"

# Step 1: Reshape input from [4,1,1,1] to [1,4,1,1]
reshaped = input_name + "_reshape"
reshape_shape_name = reshaped + "_shape"
parser.write_list_to_initializer([1, 4, 1, 1], reshape_shape_name)
parser.add_onnx_operator("Reshape", [input_name, reshape_shape_name], [reshaped])

# Step 2: Apply DepthToSpace directly (input already in NCHW)
block_shape = list(parser.get_constant_node(parser.inputs[1]))     # [2,2]
assert block_shape[0] == block_shape[1]
block_size = block_shape[0]
d2s_output = output_name + "/d2s"
parser.add_onnx_operator("DepthToSpace", [reshaped], [d2s_output],
                         attr_dict=dict(blocksize=block_size))

# Step 3: Optional Transpose to NHWC (if needed)
# Your model transposes from [1,1,2,2] → [1,2,2,1], i.e., NCHW → NHWC
transposed = output_name + "/NCHW_to_NHWC"
parser.add_onnx_permute_layer(d2s_output, transposed, parser.NCHW_TO_NHWC)

# Step 4: Add Slice
crops = list(parser.get_constant_node(parser.inputs[2]).flatten())  # [0, 0, 0, 0]
axes = [0, 1, 2, 3]  # Full 4D axes
begin = []
end = []
for i in range(0, len(crops), 2):
    begin.append(crops[i])
    end.append(-crops[i + 1]) if crops[i + 1] != 0 else end.append(crops[i + 1])
steps = [1] * len(begin)

# Write initializers
axes_name = output_name + "crop_axes"
starts_name = output_name + "crop_begin"
ends_name = output_name + "crop_end"
steps_name = output_name + "crop_steps"

parser.write_list_to_initializer(axes, axes_name)
parser.write_list_to_initializer(begin, starts_name)
parser.write_list_to_initializer(end, ends_name)
parser.write_list_to_initializer(steps, steps_name)

# Final slice output
parser.add_onnx_operator("Slice", [transposed, starts_name, ends_name, axes_name, steps_name], [output_name])


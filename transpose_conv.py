import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# === 1. Build TFLite Model with TransposeConv === #
input_shape = (1, 8, 8, 1)
kernel_size = (3, 3)
filters = 1
strides = (2, 2)

# TensorFlow Keras Model
input_tensor = tf.keras.Input(shape=(8, 8, 1))
x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding="same")(input_tensor)
tf_model = tf.keras.Model(inputs=input_tensor, outputs=x)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()
with open("transposeconv_model.tflite", "wb") as f:
    f.write(tflite_model)

# === 2. Build ONNX Model with ConvTranspose === #
class ConvTransposeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                                         kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        return self.deconv(x)

onnx_model = ConvTransposeModel()
dummy_input = torch.randn(1, 1, 8, 8)
torch.onnx.export(onnx_model, dummy_input, "convtranspose_model.onnx",
                  input_names=['input'], output_names=['output'], opset_version=11)

# === 3. Inference on Same Input === #
# Create same input for both
input_data = np.random.rand(1, 8, 8, 1).astype(np.float32)

# TFLite Inference
interpreter = tf.lite.Interpreter(model_path="transposeconv_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])

# ONNX Inference
onnx_input = np.transpose(input_data, (0, 3, 1, 2))  # NHWC -> NCHW
session = ort.InferenceSession("convtranspose_model.onnx")
onnx_output = session.run(None, {session.get_inputs()[0].name: onnx_input})[0]
onnx_output_nhwc = np.transpose(onnx_output, (0, 2, 3, 1))  # NCHW -> NHWC

# === 4. Compare Outputs === #
mae = np.mean(np.abs(tflite_output - onnx_output_nhwc))
print("✅ TFLite Output Shape:", tflite_output.shape)
print("✅ ONNX Output Shape:", onnx_output_nhwc.shape)
print("📊 Mean Absolute Error:", mae)

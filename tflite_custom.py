# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations
import warnings
import qai_hub as hub
import tensorflow.lite as tflite  # Import TFLite Interpreter
import numpy as np
from qai_hub_models.utils.args import evaluate_parser, get_hub_device
from qai_hub_models.utils.evaluate import evaluate_on_dataset

SUPPORTED_DATASETS = ["coco"]

def load_tflite_model(model_path):
    """Load and return a TensorFlow Lite model interpreter."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_tflite_inference(interpreter, input_data):
    """Run inference using the TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess input (ensure correct shape & dtype)
    input_index = input_details[0]['index']
    input_data = np.array(input_data, dtype=input_details[0]['dtype'])
    interpreter.set_tensor(input_index, input_data)

    # Run inference
    interpreter.invoke()

    # Get output
    output_index = output_details[0]['index']
    output_data = interpreter.get_tensor(output_index)
    
    return output_data

def main():
    warnings.filterwarnings("ignore")
    
    parser = evaluate_parser(
        model_cls=None,  # No predefined model class
        default_split_size=250,
        supported_datasets=SUPPORTED_DATASETS,
        is_hub_quantized=True,
    )
    
    args = parser.parse_args()
    args.device = None

    # Load Custom TFLite Model
    tflite_model_path = args.model_path  # Ensure `model_path` argument is passed
    tflite_interpreter = load_tflite_model(tflite_model_path)

    # Get Hub Device
    hub_device = get_hub_device(None, args.chipset)

    # Evaluate with TFLite model
    evaluate_on_dataset(
        tflite_interpreter,
        None,  # No Torch model
        hub_device,
        args.dataset_name,
        args.split_size,
        args.num_samples,
        args.seed,
        args.profile_options,
        args.use_cache,
    )

if __name__ == "__main__":
    main()

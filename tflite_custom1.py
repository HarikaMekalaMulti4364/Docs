from __future__ import annotations
import warnings
import qai_hub as hub
import tensorflow.lite as tflite
import numpy as np
from qai_hub_models.utils.args import evaluate_parser, get_hub_device
from qai_hub_models.utils.evaluate import evaluate_on_dataset

SUPPORTED_DATASETS = ["coco"]

class TFLiteModelWrapper:
    """A dummy wrapper to pass a TFLite model path."""
    def __init__(self, model_path):
        self.model_path = model_path

    @classmethod
    def from_pretrained(cls, model_path):
        """Mock from_pretrained method to align with evaluate_parser expectations."""
        return cls(model_path)

def load_tflite_model(model_path):
    """Load and return a TensorFlow Lite model interpreter."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_tflite_inference(interpreter, input_data):
    """Run inference using the TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    input_data = np.array(input_data, dtype=input_details[0]['dtype'])
    interpreter.set_tensor(input_index, input_data)

    interpreter.invoke()

    output_index = output_details[0]['index']
    output_data = interpreter.get_tensor(output_index)
    
    return output_data

def main():
    warnings.filterwarnings("ignore")
    
    parser = evaluate_parser(
        model_cls=TFLiteModelWrapper,  # ✅ Pass our dummy class
        default_split_size=250,
        supported_datasets=SUPPORTED_DATASETS,
        is_hub_quantized=True,
    )
    
    args = parser.parse_args()
    args.device = None

    # Load Custom TFLite Model
    tflite_model_path = args.model_path  # Ensure `--model_path` is provided
    tflite_interpreter = load_tflite_model(tflite_model_path)

    # Get Hub Device
    hub_device = get_hub_device(None, args.chipset)

    # Evaluate using TFLite model
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

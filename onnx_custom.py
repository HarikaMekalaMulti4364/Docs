from __future__ import annotations
import warnings
import onnxruntime as ort
import numpy as np
import qai_hub as hub
from qai_hub_models.utils.args import evaluate_parser, get_hub_device, get_model_kwargs
from qai_hub_models.utils.evaluate import evaluate_on_dataset

SUPPORTED_DATASETS = ["coco"]

class ONNXModelWrapper:
    """Wrapper class to handle ONNX model inference."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)

    def infer(self, input_data):
        """Run inference with ONNX model."""
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        # Ensure input is in the correct format
        input_data = np.array(input_data, dtype=np.float32)

        # Run inference
        result = self.session.run([output_name], {input_name: input_data})
        return result[0]  # Extract output tensor

def main():
    warnings.filterwarnings("ignore")
    
    parser = evaluate_parser(
        model_cls=ONNXModelWrapper,  # ✅ Using our ONNX model class
        default_split_size=250,
        supported_datasets=SUPPORTED_DATASETS,
        is_hub_quantized=True,
    )
    
    args = parser.parse_args()
    args.device = None

    # Load Custom ONNX Model
    onnx_model = ONNXModelWrapper(args.model_path)  # ✅ Pass ONNX model

    # Get Hub Device
    hub_device = get_hub_device(None, args.chipset)

    # Evaluate ONNX Model
    evaluate_on_dataset(
        onnx_model,
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

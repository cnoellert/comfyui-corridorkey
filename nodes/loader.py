import os
import sys
import torch
import folder_paths
from ..utils.device import get_device, get_dtype

# Add the reference repo to sys.path so CorridorKeyModule's internal
# relative imports (from .core import ...) resolve correctly.
_REF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)


class CK_ModelLoader:
    """Load CorridorKey (GreenFormer) model checkpoint."""

    @classmethod
    def INPUT_TYPES(s):
        # List available .pth files in the corridorkey model directory
        model_dir = os.path.join(folder_paths.models_dir, "corridorkey")
        os.makedirs(model_dir, exist_ok=True)

        models = [f for f in os.listdir(model_dir) if f.endswith((".pth", ".safetensors"))]
        if not models:
            models = ["CorridorKey_v1.0.pth"]  # Default expected filename

        return {
            "required": {
                "model_name": (models, {"default": models[0]}),
                "use_refiner": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CK_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "CorridorKey"

    def load(self, model_name, use_refiner):
        from CorridorKeyModule.inference_engine import CorridorKeyEngine

        model_path = os.path.join(folder_paths.models_dir, "corridorkey", model_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Download CorridorKey.pth and place it in: {os.path.dirname(model_path)}"
            )

        device = get_device()
        engine = CorridorKeyEngine(
            checkpoint_path=model_path,
            device=str(device),
            img_size=2048,
            use_refiner=use_refiner,
        )
        return (engine,)


NODE_CLASS_MAPPINGS = {
    "CK_ModelLoader": CK_ModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CK_ModelLoader": "CorridorKey Model Loader",
}

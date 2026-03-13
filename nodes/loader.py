import os
import sys
import folder_paths
from ..utils.device import get_device, warn_if_low_memory

_REF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)


class CK_ModelLoader:
    """Load CorridorKey (GreenFormer) model checkpoint."""

    @classmethod
    def INPUT_TYPES(s):
        model_dir = os.path.join(folder_paths.models_dir, "corridorkey")
        os.makedirs(model_dir, exist_ok=True)
        models = [f for f in os.listdir(model_dir) if f.endswith((".pth", ".safetensors"))]
        if not models:
            models = ["CorridorKey_v1.0.pth"]
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
        from ..utils.inference import OptimizedEngine

        model_path = os.path.join(folder_paths.models_dir, "corridorkey", model_name)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Download CorridorKey_v1.0.pth from:\n"
                f"  https://huggingface.co/nikopueringer/CorridorKey_v1.0\n"
                f"and place it in: {os.path.dirname(model_path)}"
            )

        device = get_device()

        warning = warn_if_low_memory(device, required_gb=24.0)
        if warning:
            print(warning)

        print(f"[CorridorKey] Loading model on {device} ...")
        engine = CorridorKeyEngine(
            checkpoint_path=model_path,
            device=str(device),
            img_size=2048,
            use_refiner=use_refiner,
        )

        if device.type in ("mps", "cuda"):
            engine = OptimizedEngine(engine)
            mode = "float16 autocast" if device.type == "cuda" else "float32 / no autocast"
            print(f"[CorridorKey] Optimised path active (channels_last, on-device resize/norm, {mode})")
        else:
            print("[CorridorKey] CPU inference path (reference engine)")

        return (engine,)


NODE_CLASS_MAPPINGS = {"CK_ModelLoader": CK_ModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"CK_ModelLoader": "CorridorKey Model Loader"}

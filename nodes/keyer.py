import torch
import numpy as np
from ..utils.color import linear_to_srgb


class CK_Keyer:
    """
    Main CorridorKey green screen keyer node.

    On MPS/CUDA uses OptimizedEngine.process_frame_tensor() — tensors stay
    on-device through resize, norm, forward, and post-processing.
    On CPU falls back to the reference engine's numpy path unchanged.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("CK_MODEL",),
                "image": ("IMAGE",),
                "alpha_hint": ("MASK",),
                "input_is_linear": ("BOOLEAN", {"default": False}),
                "despill_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "auto_despeckle": ("BOOLEAN", {"default": True}),
                "despeckle_size": ("INT", {"default": 400, "min": 10, "max": 5000, "step": 10}),
                "refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("foreground", "alpha", "processed")
    FUNCTION = "key"
    CATEGORY = "CorridorKey"


    def key(self, model, image, alpha_hint, input_is_linear,
            despill_strength, auto_despeckle, despeckle_size, refiner_strength):
        from ..utils.inference import OptimizedEngine
        use_tensor_path = isinstance(model, OptimizedEngine)

        fg_list, alpha_list, processed_list = [], [], []

        for i in range(image.shape[0]):
            if use_tensor_path:
                # Tensors go straight to the engine — no numpy round-trip
                result = model.process_frame_tensor(
                    image_t=image[i],
                    mask_t=alpha_hint[i],
                    input_is_linear=input_is_linear,
                    despill_strength=despill_strength,
                    auto_despeckle=auto_despeckle,
                    despeckle_size=despeckle_size,
                    refiner_scale=refiner_strength,
                )
                fg_t    = result["fg"]         # [H, W, 3] CPU float32
                alpha_t = result["alpha"]      # [H, W, 1] CPU float32
                proc_t  = result["processed"]  # [H, W, 4] CPU float32

            else:
                # CPU fallback — original reference engine numpy path
                img_np  = image[i].cpu().numpy()
                mask_np = alpha_hint[i].cpu().numpy()
                result  = model.process_frame(
                    image=img_np,
                    mask_linear=mask_np,
                    input_is_linear=input_is_linear,
                    despill_strength=despill_strength,
                    auto_despeckle=auto_despeckle,
                    despeckle_size=despeckle_size,
                    refiner_scale=refiner_strength,
                )
                fg_t = torch.from_numpy(result["fg"].astype(np.float32))
                alpha_arr = result["alpha"]
                if alpha_arr.ndim == 3:
                    alpha_arr = alpha_arr[..., 0:1]
                alpha_t = torch.from_numpy(alpha_arr.astype(np.float32))
                proc_t  = torch.from_numpy(result["processed"].astype(np.float32))

            fg_list.append(fg_t.float())
            # MASK wants [H, W] — drop trailing channel dim
            alpha_2d = alpha_t[..., 0] if alpha_t.ndim == 3 else alpha_t
            alpha_list.append(alpha_2d.float())
            # Convert linear premul preview to sRGB for display
            proc_srgb = linear_to_srgb(proc_t[..., :3])
            processed_list.append(proc_srgb.clamp(0, 1).float())

        return (
            torch.stack(fg_list),
            torch.stack(alpha_list),
            torch.stack(processed_list),
        )


NODE_CLASS_MAPPINGS = {"CK_Keyer": CK_Keyer}
NODE_DISPLAY_NAME_MAPPINGS = {"CK_Keyer": "CorridorKey Keyer"}

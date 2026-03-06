import torch
import numpy as np
from ..utils.color import srgb_to_linear, linear_to_srgb, despill_green


class CK_Keyer:
    """Main CorridorKey green screen keyer node.

    Takes an image and alpha hint mask, outputs foreground, alpha matte,
    and premultiplied composite.
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

    def key(self, model, image, alpha_hint, input_is_linear, despill_strength,
            auto_despeckle, despeckle_size, refiner_strength):
        """Process each frame through CorridorKey engine.

        Args:
            model: CorridorKeyEngine instance from loader
            image: [B, H, W, 3] sRGB float32 [0, 1]
            alpha_hint: [B, H, W] float32 [0, 1]
            input_is_linear: Whether input image is linear (e.g., from EXR)
            despill_strength: Green spill removal strength
            auto_despeckle: Remove small matte islands
            despeckle_size: Min pixel count to keep
            refiner_strength: Refiner delta multiplier
        """
        batch_size = image.shape[0]
        fg_list = []
        alpha_list = []
        processed_list = []

        for i in range(batch_size):
            # Convert from ComfyUI [H, W, C] to numpy [H, W, C]
            img_np = image[i].cpu().numpy()
            mask_np = alpha_hint[i].cpu().numpy()

            result = model.process_frame(
                image=img_np,
                mask_linear=mask_np,
                input_is_linear=input_is_linear,
                despill_strength=despill_strength,
                auto_despeckle=auto_despeckle,
                despeckle_size=despeckle_size,
                refiner_scale=refiner_strength,
            )

            # fg: [H, W, 3] sRGB — ready for ComfyUI IMAGE
            fg_list.append(torch.from_numpy(result["fg"]).float())

            # alpha: [H, W, 1] → [H, W] for ComfyUI MASK
            alpha_out = result["alpha"]
            if alpha_out.ndim == 3:
                alpha_out = alpha_out[..., 0]
            alpha_list.append(torch.from_numpy(alpha_out).float())

            # processed: [H, W, 4] linear premul RGBA → show as sRGB RGB preview
            proc = result["processed"]
            # Convert linear premul RGB to sRGB for display (drop alpha channel)
            proc_rgb = proc[..., :3]
            proc_srgb = linear_to_srgb(proc_rgb)
            processed_list.append(torch.from_numpy(proc_srgb.astype(np.float32)))

        fg_batch = torch.stack(fg_list)
        alpha_batch = torch.stack(alpha_list)
        processed_batch = torch.stack(processed_list)

        return (fg_batch, alpha_batch, processed_batch)


NODE_CLASS_MAPPINGS = {
    "CK_Keyer": CK_Keyer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CK_Keyer": "CorridorKey Keyer",
}

import torch
from ..utils.color import despill_green, srgb_to_linear, linear_to_srgb


class CK_Despill:
    """Standalone green spill removal node."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "despill"
    CATEGORY = "CorridorKey/Utils"

    def despill(self, image, strength):
        return (despill_green(image, strength),)


class CK_Despeckle:
    """Standalone matte cleanup — removes small isolated regions."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "min_size": ("INT", {"default": 400, "min": 10, "max": 5000, "step": 10}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "despeckle"
    CATEGORY = "CorridorKey/Utils"

    def despeckle(self, mask, min_size):
        from scipy import ndimage
        import numpy as np

        results = []
        for i in range(mask.shape[0]):
            m = mask[i].cpu().numpy()
            binary = m > 0.5
            labeled, num_features = ndimage.label(binary)
            cleaned = binary.copy()
            for label_id in range(1, num_features + 1):
                component = labeled == label_id
                if component.sum() < min_size:
                    cleaned[component] = False
            # Preserve original alpha values where kept, zero where removed
            result = m * cleaned.astype(np.float32)
            results.append(torch.from_numpy(result))

        return (torch.stack(results),)


class CK_LinearToSRGB:
    """Convert linear image to sRGB."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "CorridorKey/Utils"

    def convert(self, image):
        return (linear_to_srgb(image).clamp(0, 1),)


class CK_SRGBToLinear:
    """Convert sRGB image to linear."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "CorridorKey/Utils"

    def convert(self, image):
        return (srgb_to_linear(image).clamp(0, 1),)


NODE_CLASS_MAPPINGS = {
    "CK_Despill": CK_Despill,
    "CK_Despeckle": CK_Despeckle,
    "CK_LinearToSRGB": CK_LinearToSRGB,
    "CK_SRGBToLinear": CK_SRGBToLinear,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CK_Despill": "CorridorKey Despill",
    "CK_Despeckle": "CorridorKey Despeckle",
    "CK_LinearToSRGB": "Linear to sRGB",
    "CK_SRGBToLinear": "sRGB to Linear",
}

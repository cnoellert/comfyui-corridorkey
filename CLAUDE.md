# ComfyUI-CorridorKey

ComfyUI custom node pack porting [CorridorKey](https://github.com/nikopueringer/CorridorKey) — neural green screen keying with physically accurate color unmixing.

## Architecture

```
ComfyUI-CorridorKey/
├── __init__.py              # Entry point — exports NODE_CLASS_MAPPINGS
├── nodes/                   # ComfyUI node definitions
│   ├── __init__.py
│   ├── keyer.py             # Main CorridorKey keying node
│   ├── loader.py            # Model loader node
│   └── utils.py             # Despill, despeckle, color space utility nodes
├── utils/                   # Internal helpers
│   ├── __init__.py
│   ├── device.py            # CUDA/MPS/CPU device resolution
│   └── color.py             # sRGB↔linear, despill, compositing math
├── reference/               # Original CorridorKey repo (git-ignored, for reference only)
├── pyproject.toml
├── requirements.txt
└── CLAUDE.md
```

## Source Reference

The original CorridorKey repo is cloned at `reference/` for study. Key files:
- `reference/CorridorKeyModule/core/model_transformer.py` — GreenFormer model (Hiera backbone + twin decoders + CNN refiner)
- `reference/CorridorKeyModule/inference_engine.py` — Full inference pipeline with pre/post processing
- `reference/CorridorKeyModule/core/color_utils.py` — Compositing math (despill, clean_matte, color space)
- `reference/device_utils.py` — CUDA/MPS/CPU auto-detection
- `reference/clip_manager.py` — Batch workflow orchestrator

## Model Architecture (GreenFormer)

- **Backbone:** Hiera Base Plus (MAE pretrained), patched from 3→4 input channels
- **Input:** `[B, 4, 2048, 2048]` — ImageNet-normalized sRGB (3ch) + alpha hint mask (1ch)
- **Twin decoders:** Alpha `[B, 1, H, W]` + Foreground `[B, 3, H, W]` (both sigmoid → [0,1])
- **CNN Refiner:** Dilated residual blocks, input `[B, 7, H, W]` (RGB + coarse preds), outputs delta logits
- **Native resolution:** 2048×2048 (dynamic scaling for arbitrary input sizes)
- **Checkpoint:** `CorridorKey.pth` (~350MB), loaded via `torch.load()`
- **VRAM:** ~23GB at native 2K resolution

## ComfyUI Node Design

### Planned Nodes

| Node | Category | Purpose |
|------|----------|---------|
| `CorridorKeyModelLoader` | CorridorKey | Load GreenFormer checkpoint, select device/dtype |
| `CorridorKeyKeyer` | CorridorKey | Main keying — takes IMAGE + MASK, returns FG + Alpha + Processed |
| `CorridorKeyDespill` | CorridorKey/Utils | Standalone green despill with strength control |
| `CorridorKeyDespeckle` | CorridorKey/Utils | Standalone matte cleanup (connected components) |

### Tensor Conventions (ComfyUI)
- **IMAGE:** `[B, H, W, C]` float32 `[0, 1]` (sRGB)
- **MASK:** `[B, H, W]` float32 `[0, 1]`
- Model expects `[B, C, H, W]` — permute on entry/exit

### Inference Pipeline (inside CorridorKeyKeyer)
1. Receive IMAGE `[B, H, W, 3]` + MASK `[B, H, W]` from ComfyUI
2. Permute to `[B, 3, H, W]`, resize to 2048×2048
3. ImageNet normalize RGB, concat mask → `[B, 4, 2048, 2048]`
4. Forward through GreenFormer (autocast fp16)
5. Resize outputs back to original resolution (bilinear)
6. Post-process: despeckle, despill, color space conversion
7. Permute back to `[B, H, W, C]` for ComfyUI
8. Return: IMAGE (foreground), MASK (alpha), IMAGE (premul RGBA preview)

## Device Compatibility (CUDA + MPS)

**Critical MPS considerations:**
- Use `torch.float16` not `torch.bfloat16` on MPS (accumulated rounding errors)
- `torch.Generator` must be created on CPU for MPS, then tensors moved
- `channels_last` memory format benefits both CUDA and MPS (10-20% speedup on Metal)
- `torch.mps.empty_cache()` for MPS, `torch.cuda.empty_cache()` for CUDA
- Use `comfy.model_management.get_torch_device()` where possible for ComfyUI integration

**Device resolution pattern:**
```python
import comfy.model_management
device = comfy.model_management.get_torch_device()  # Preferred — respects ComfyUI settings
```

## Color Space Handling

- **Model input:** sRGB (ImageNet-normalized)
- **Alpha output:** Linear [0, 1]
- **FG output:** sRGB [0, 1] (from sigmoid)
- **Processed output:** Linear premultiplied RGBA (VFX standard)
- **ComfyUI convention:** sRGB [0, 1] for IMAGE type

Use piecewise sRGB↔linear (not gamma 2.2 approximation):
```python
# sRGB → linear
linear = where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
# linear → sRGB
srgb = where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1/2.4) - 0.055)
```

## Dependencies

- `torch` (CUDA or MPS)
- `timm` (Hiera backbone)
- `opencv-python` (frame I/O, resize)
- `numpy`
- `scipy` (connected components for despeckle)

## Development

```bash
# Run from project root
cd ~/Documents/GitHub/comfyui-corridorkey

# Symlink into ComfyUI for testing
ln -sf ~/Documents/GitHub/comfyui-corridorkey ~/ComfyUI/custom_nodes/ComfyUI-CorridorKey

# Launch ComfyUI to test
cd ~/ComfyUI && python main.py
```

## Conventions

- Follow existing ComfyUI custom node patterns (see ~/ComfyUI/custom_nodes/ for examples)
- All nodes use `CATEGORY = "CorridorKey"` or `"CorridorKey/Utils"`
- Node internal names prefixed with `CK_` (e.g., `CK_ModelLoader`, `CK_Keyer`)
- Display names use readable format (e.g., "CorridorKey Model Loader")
- Use `folder_paths.models_dir` for model storage location
- Write tests that don't require GPU (mock model weights)

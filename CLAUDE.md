# ComfyUI-CorridorKey

ComfyUI custom node pack porting [CorridorKey](https://github.com/nikopueringer/CorridorKey) —
neural green screen keying with physically accurate color unmixing.

## Repository Layout

```
ComfyUI-CorridorKey/
├── __init__.py              # Entry point — exports NODE_CLASS_MAPPINGS
├── nodes/
│   ├── __init__.py
│   ├── keyer.py             # CK_Keyer — main keying node
│   ├── loader.py            # CK_ModelLoader — checkpoint loader
│   └── utils.py             # CK_Despill, CK_Despeckle, color space nodes
├── utils/
│   ├── __init__.py
│   ├── color.py             # sRGB↔linear, despill math (torch + numpy)
│   ├── device.py            # Device resolution, dtype, autocast, memory helpers
│   └── inference.py         # OptimizedEngine — on-device MPS/CUDA inference wrapper
├── reference/               # Original CorridorKey repo (git-ignored, reference only)
├── workflows/
│   └── corridorkey_basic.json
├── pyproject.toml
├── requirements.txt
└── CLAUDE.md
```

## Model Architecture (GreenFormer)

- **Backbone:** Hiera Base Plus (MAE pretrained), input channels patched 3→4
- **Input:** `[B, 4, 2048, 2048]` — ImageNet-normalised sRGB (3ch) + alpha hint (1ch)
- **Twin decoders:** Alpha `[B, 1, H, W]` + Foreground `[B, 3, H, W]` (sigmoid → [0,1])
- **CNN Refiner:** Dilated residual blocks, input `[B, 7, H, W]`, outputs delta logits
- **Native resolution:** 2048×2048 (dynamic scaling for arbitrary input sizes)
- **Checkpoint:** `CorridorKey_v1.0.pth` (~383MB)
- **Memory:** ~23 GB at native 2K (VRAM on CUDA, unified RAM on MPS)

Key reference files:
- `reference/CorridorKeyModule/core/model_transformer.py` — GreenFormer definition
- `reference/CorridorKeyModule/inference_engine.py` — original CPU/numpy pipeline
- `reference/CorridorKeyModule/core/color_utils.py` — compositing math

## Nodes

| Internal name | Display name | Category |
|---|---|---|
| `CK_ModelLoader` | CorridorKey Model Loader | CorridorKey |
| `CK_Keyer` | CorridorKey Keyer | CorridorKey |
| `CK_Despill` | CorridorKey Despill | CorridorKey/Utils |
| `CK_Despeckle` | CorridorKey Despeckle | CorridorKey/Utils |
| `CK_LinearToSRGB` | Linear to sRGB | CorridorKey/Utils |
| `CK_SRGBToLinear` | sRGB to Linear | CorridorKey/Utils |

## Tensor Conventions (ComfyUI)

- **IMAGE:** `[B, H, W, C]` float32 [0, 1] sRGB
- **MASK:** `[B, H, W]` float32 [0, 1]
- Model expects `[B, C, H, W]` — permuted on entry/exit inside the engine


## Inference Pipeline

### Reference engine (CPU path, `CorridorKeyEngine.process_frame`)

The original upstream engine operates entirely in CPU/numpy:

```
ComfyUI tensor → .cpu().numpy()
→ cv2.resize (CPU)
→ numpy ImageNet norm (CPU)
→ torch.from_numpy().to(device) → model forward → .cpu().numpy()
→ cv2.resize (CPU)
→ return numpy dict
```

This path is used unchanged when running on CPU.

### OptimizedEngine (MPS / CUDA path)

`utils/inference.py` wraps the reference engine with an on-device implementation:

1. Tensors received directly from ComfyUI — no numpy conversion
2. `F.interpolate` for both resizes (stays on GPU)
3. ImageNet normalisation via cached device tensors (`[1,3,1,1]` mean/std)
4. `channels_last` memory format applied at model init (Metal NHWC native layout, ~10–20% faster)
5. Autocast: **float16 on CUDA**, **nullcontext (float32) on MPS**
6. Post-process (despill, premultiply) on device
7. Single D2H transfer for scipy despeckle (no PyTorch connected-components equivalent)
8. `torch.mps.empty_cache()` / `torch.cuda.empty_cache()` after each frame
9. Returns CPU float32 tensors for ComfyUI batch assembly

`CK_Keyer` detects engine type with `isinstance(model, OptimizedEngine)` and routes
to `process_frame_tensor()` (tensor-native) or falls back to the numpy path.

## Device & Dtype Decisions

### Why float32 on MPS (not float16)

`torch.autocast` on MPS is unreliable for this model. The dilated CNN refiner and
Hiera pos_embed interpolation both produce NaN/inf under float16 on Metal. float32
is ~15% slower but numerically stable. This is a deliberate choice, not an oversight.

### Why channels_last

Metal Performance Shaders operate natively in NHWC (channels-last) layout. PyTorch's
default NCHW requires Metal to transpose on every convolution, measurably slower.
Calling `.to(memory_format=torch.channels_last)` at model load eliminates this.

### Device resolution order

1. `comfy.model_management.get_torch_device()` — respects ComfyUI's device config
2. CUDA if available
3. MPS if available
4. CPU fallback

### Memory warning

`warn_if_low_memory()` in `utils/device.py` uses `psutil` to read total unified RAM
on Apple Silicon and warns before model load if under 24 GB.


## Color Space Handling

| Pass | Space | Notes |
|---|---|---|
| Model input | sRGB (ImageNet-normalised) | cv2/F.interpolate resize happens before norm |
| Alpha output | Linear [0, 1] | Raw sigmoid prediction |
| FG output | sRGB [0, 1] | Raw sigmoid prediction, straight (unpremultiplied) |
| Processed output | Linear premultiplied RGBA | VFX standard for EXR delivery |
| ComfyUI IMAGE | sRGB [0, 1] | Preview nodes convert linear→sRGB before output |

Piecewise transfer functions (not gamma 2.2 approximation) used throughout:

```python
# sRGB → linear
linear = where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
# linear → sRGB
srgb   = where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1/2.4) - 0.055)
```

Both `utils/color.py` (torch + numpy) and `utils/inference.py` (torch on-device)
implement these independently to avoid cross-module imports on the hot path.

## Roadmap

### Done
- [x] Initial ComfyUI port (keyer, loader, utility nodes)
- [x] CUDA + MPS device support
- [x] MPS float32 fix (was incorrectly using float16)
- [x] OptimizedEngine — on-device resize/norm/postprocess for MPS and CUDA
- [x] channels_last memory format (Metal NHWC)
- [x] Nullcontext autocast on MPS, float16 autocast on CUDA
- [x] Memory warning via psutil before model load
- [x] Tested on Apple M4 Max (128GB) at 6K resolution

### Next
- [ ] Standalone MLX inference script (Apple Silicon native, no PyTorch boundary)
  - Re-implement GreenFormer in `mlx.nn`
  - `.pth` → MLX weight converter
  - CLI batch processor (EXR in/out)
  - Benchmark transformer backbone vs MPS path on M4 Max

## Development

```bash
# Install deps
cd ~/ComfyUI/custom_nodes/comfyui-corridorkey
pip install -r requirements.txt

# Launch ComfyUI — watch console for:
# [CorridorKey] channels_last memory format enabled on mps
# [CorridorKey] Optimised path active (channels_last, on-device resize/norm, float32 / no autocast)
cd ~/ComfyUI && python main.py
```

## Conventions

- All nodes: `CATEGORY = "CorridorKey"` or `"CorridorKey/Utils"`
- Internal names prefixed `CK_` — display names use readable format
- `folder_paths.models_dir` for checkpoint storage (`ComfyUI/models/corridorkey/`)
- Engine type detection via `isinstance(model, OptimizedEngine)` — not duck typing
- CPU fallback always available; never require GPU to import the package

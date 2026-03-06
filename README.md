# ComfyUI-CorridorKey

ComfyUI custom nodes for [CorridorKey](https://github.com/nikopueringer/CorridorKey) — neural green screen keying with physically accurate color unmixing.

CorridorKey uses a transformer-based architecture (GreenFormer) to solve the color unmixing problem in VFX, producing clean alpha mattes and true foreground colors for semi-transparent edges like hair, motion blur, and out-of-focus elements.

## Features

- **Neural keying** — GreenFormer backbone with CNN refiner for artifact-free mattes
- **CUDA + MPS support** — Works on NVIDIA GPUs and Apple Silicon
- **VFX-standard output** — Linear premultiplied RGBA, separate FG and alpha passes
- **Standalone utilities** — Despill, despeckle, and color space conversion nodes

## Nodes

| Node | Description |
|------|-------------|
| **CorridorKey Model Loader** | Load GreenFormer checkpoint, configure refiner |
| **CorridorKey Keyer** | Main keying node — IMAGE + MASK → Foreground, Alpha, Processed |
| **CorridorKey Despill** | Standalone luminance-preserving green spill removal |
| **CorridorKey Despeckle** | Matte cleanup — remove small isolated regions |
| **Linear to sRGB** | Color space conversion |
| **sRGB to Linear** | Color space conversion |

## Installation

### Via ComfyUI-Manager

Search for **ComfyUI-CorridorKey** in ComfyUI-Manager and click Install.

### Manual

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/cnoellert/comfyui-corridorkey.git ComfyUI-CorridorKey
cd ComfyUI-CorridorKey
pip install -r requirements.txt
```

### Model Weights

Download [CorridorKey_v1.0.pth](https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth) (~383MB) and place it in:

```
ComfyUI/models/corridorkey/CorridorKey_v1.0.pth
```

## Usage

1. Load the model with **CorridorKey Model Loader**
2. Feed a green screen image and an alpha hint mask into **CorridorKey Keyer**
3. The node outputs:
   - **Foreground** — sRGB foreground with green removed
   - **Alpha** — Clean matte
   - **Processed** — Premultiplied composite preview

A starter workflow is included in `workflows/corridorkey_basic.json`.

## Requirements

- Python 3.10+
- PyTorch with CUDA or MPS support
- `timm` >= 1.0.0
- `opencv-python` >= 4.8.0
- `numpy`
- `scipy`

## Hardware

- **GPU:** 24GB+ VRAM recommended for native 2048×2048 inference
- **Apple Silicon:** Works on M1/M2/M3/M4 with MPS backend (tested on PyTorch 2.10)
- **NVIDIA:** CUDA with fp16 autocast

## Credits

- [CorridorKey](https://github.com/nikopueringer/CorridorKey) by Niko Pueringer — original model and inference engine
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## License

CC BY-NC-SA 4.0 — Free for commercial image processing, prohibited for resale or paid API services. See the original CorridorKey license for full terms.

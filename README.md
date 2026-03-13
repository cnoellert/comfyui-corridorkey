# ComfyUI-CorridorKey

ComfyUI custom nodes for [CorridorKey](https://github.com/nikopueringer/CorridorKey) — neural green screen keying with physically accurate color unmixing.

CorridorKey uses a transformer-based architecture (GreenFormer) to solve the color unmixing problem in VFX: for every pixel, including semi-transparent edges like hair, motion blur, and out-of-focus elements, the model predicts the true straight foreground color and a clean linear alpha matte — as if the green screen was never there.

## Features

- **Neural keying** — GreenFormer (Hiera backbone + CNN refiner) for artifact-free mattes
- **Optimised MPS + CUDA paths** — on-device resize, normalisation, and post-processing; channels_last memory layout for Metal
- **VFX-standard output** — linear premultiplied RGBA, separate FG and alpha passes, EXR-ready
- **Standalone utilities** — despill, despeckle, and color space conversion nodes
- **Tested at 6K** on Apple M4 Max (128 GB unified memory)

## Nodes

| Node | Description |
|------|-------------|
| **CorridorKey Model Loader** | Load GreenFormer checkpoint, configure refiner |
| **CorridorKey Keyer** | Main node — IMAGE + MASK → Foreground, Alpha, Processed |
| **CorridorKey Despill** | Standalone luminance-preserving green spill removal |
| **CorridorKey Despeckle** | Matte cleanup — remove small isolated regions |
| **Linear to sRGB** | Color space conversion |
| **sRGB to Linear** | Color space conversion |

## Installation

### Via ComfyUI-Manager

Search for **ComfyUI-CorridorKey** and click Install.

### Manual

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/cnoellert/comfyui-corridorkey.git ComfyUI-CorridorKey
cd ComfyUI-CorridorKey
pip install -r requirements.txt
```

### Model Weights

Download [CorridorKey_v1.0.pth](https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth) (~383 MB) and place it at:

```
ComfyUI/models/corridorkey/CorridorKey_v1.0.pth
```


## Usage

1. Load the model with **CorridorKey Model Loader**
2. Connect a green screen image and a coarse alpha hint mask to **CorridorKey Keyer**
3. The node outputs:
   - **Foreground** — sRGB, straight (unpremultiplied), green removed
   - **Alpha** — clean linear matte
   - **Processed** — sRGB preview of the premultiplied composite

A starter workflow is included in `workflows/corridorkey_basic.json`.

## Hardware Requirements

CorridorKey requires approximately **23 GB** at native 2048×2048 inference.

| Platform | Requirement |
|---|---|
| NVIDIA GPU | 24 GB+ VRAM (RTX 3090, 4090, 5090, A6000, etc.) |
| Apple Silicon | 24 GB+ unified RAM (M1/M2/M3/M4 Max or Ultra) |
| CPU | Supported but very slow |

On Apple Silicon, unified memory is shared between CPU and GPU — an M4 Max with 128 GB
can run this comfortably even at resolutions above 4K.

## Apple Silicon / MPS Notes

The MPS path uses a purpose-built `OptimizedEngine` wrapper:

- **float32 throughout** — Metal float16 autocast causes NaN/inf in GreenFormer's
  CNN refiner. float32 is the stable and recommended path on Apple Silicon.
- **channels_last memory format** — Metal Performance Shaders operate natively in
  NHWC layout. This gives ~10–20% throughput improvement on convolution layers.
- **On-device resize and normalisation** — `F.interpolate` replaces `cv2.resize`,
  keeping tensors on the GPU through the entire pipeline.
- The loader logs which path is active on startup.

## Requirements

- Python 3.10+
- PyTorch with CUDA or MPS support
- `timm >= 1.0.0`
- `opencv-python >= 4.8.0`
- `numpy`
- `scipy`
- `psutil >= 5.9` (unified RAM detection on Apple Silicon)

## Credits

- [CorridorKey](https://github.com/nikopueringer/CorridorKey) by Niko Pueringer (Corridor Digital) — original model and inference engine
- ComfyUI port by [cnoellert](https://github.com/cnoellert)

## License

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — free for personal and commercial production use, prohibited for resale or paid API services. See the [original CorridorKey license](https://github.com/nikopueringer/CorridorKey#corridokey-licensing-and-permissions) for full terms.

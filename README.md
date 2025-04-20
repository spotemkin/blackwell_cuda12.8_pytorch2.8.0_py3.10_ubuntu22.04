# NVIDIA Blackwell GPU Docker Container

Ready-to-use Docker image with full support for NVIDIA Blackwell architecture (sm_120) for RTX 5090 and other Blackwell-based GPUs.

## Features

- **CUDA 12.8** with official Blackwell architecture (sm_120) support
- **PyTorch Nightly** with CUDA 12.8 and Blackwell GPU support
- **xformers** installed directly from GitHub for maximum compatibility
- **OpenCV** with all necessary dependencies
- **Pre-installed libraries** for machine learning and computer vision
- **Verification utility** to confirm Blackwell support

## Hardware Support

- NVIDIA RTX 5090
- Other NVIDIA Blackwell architecture (sm_120) cards

## Why This Image?

When attempting to use cutting-edge NVIDIA Blackwell GPUs (like RTX 5090) with standard Docker images, you'll encounter these errors:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

This happens because most images (even official NVIDIA ones) don't yet include support for the sm_120 architecture. Without this support, PyTorch applications fail to run on Blackwell GPUs.

**This image solves this problem** by:

1. Using CUDA 12.8 which includes Blackwell support
2. Installing PyTorch from nightly builds with CUDA 12.8 support
3. Building xformers from source to ensure compatibility

Now you can simply pull this image and start working with your RTX 5090 immediately, without spending hours researching compatibility issues and building custom environments.

## Installation and Usage

### Pull the image

```bash
docker pull sergeypotemkin/blackwell_cuda12.8_pytorch2.8.0_py3.10_ubuntu22.04:latest
```

### Run container

```bash
docker run --gpus all -it --rm sergeypotemkin/blackwell_cuda12.8_pytorch2.8.0_py3.10_ubuntu22.04:latest
```

### Verify Blackwell support

Inside the container, run:

```bash
python /usr/local/bin/verify_blackwell.py
```

### Mount your code

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  sergeypotemkin/blackwell_cuda12.8_pytorch2.8.0_py3.10_ubuntu22.04:latest
```

## Pre-installed Packages

- PyTorch, torchvision, torchaudio (nightly build with CUDA 12.8 support)
- xformers (latest version from GitHub)
- transformers, diffusers
- opencv-python
- numpy, scipy, pandas, matplotlib
- pillow, tqdm
- accelerate, safetensors, bitsandbytes

## Using in Your Projects

### Use as base image

```dockerfile
FROM sergeypotemkin/blackwell_cuda12.8_pytorch2.8.0_py3.10_ubuntu22.04:latest

# Add your code and dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
```

## Modifications

If you need build some others versions/packages - see and modify Dockerfile on my Github as you wish
https://github.com/spotemkin/blackwell_cuda12.8_pytorch2.8.0_py3.10_ubuntu22.04

## Issues and Support

If you encounter any issues, please create an issue in the GitHub repository or contact the author at prsl.ru@gmail.com.

## License

MIT

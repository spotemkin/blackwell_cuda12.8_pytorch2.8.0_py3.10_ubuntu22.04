#!/usr/bin/env python3
"""
NVIDIA Blackwell GPU Support Verification Script
-----------------------------------------------
This script checks if your PyTorch installation and GPU properly support
NVIDIA Blackwell architecture (sm_120).

Author: spotemkin <prsl.ru@gmail.com>
"""

import sys
import platform
import subprocess
from datetime import datetime


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    print_header("NVIDIA BLACKWELL SUPPORT VERIFICATION TOOL")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {platform.python_version()} ({sys.executable})")

    # Check PyTorch installation
    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            print("\n❌ CUDA is not available. Please check your installation.")
            return

        print(f"CUDA version: {torch.version.cuda}")

        # Check supported architectures
        try:
            archs = torch._C._cuda_getArchFlags()
            print(f"Supported CUDA architectures: {archs}")

            if "sm_120" in archs:
                print("\n✅ Blackwell architecture (sm_120) is supported by PyTorch!")
            else:
                print("\n❌ Blackwell architecture (sm_120) is NOT supported by this PyTorch build.")
                print("   This might not be the right container for RTX 5090 GPUs.")
        except Exception as e:
            print(f"\n❌ Error checking CUDA architectures: {e}")

        # Check GPU devices
        device_count = torch.cuda.device_count()
        print(f"\nGPU count: {device_count}")

        blackwell_found = False

        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)

            print(f"\nGPU #{i}: {name}")
            print(f"  Compute capability: {capability[0]}.{capability[1]}")
            print(f"  Memory: {memory_gb:.2f} GB")

            # Check if this is a Blackwell GPU (cc 12.0)
            if capability[0] == 12:
                blackwell_found = True
                print(f"  ✅ This is a Blackwell architecture GPU!")

        if not blackwell_found:
            print("\n⚠️  No Blackwell GPUs detected. This container supports Blackwell GPUs (RTX 5090, etc.)")

        # Test xformers
        try:
            import xformers
            print(f"\nxformers: {xformers.__version__}")
            print("✅ xformers is installed")
        except ImportError:
            print("\n❌ xformers is not installed")

        # Run a simple computation to verify GPU works
        try:
            print("\nRunning a simple GPU computation test...")
            x = torch.rand(1000, 1000, device='cuda')
            y = torch.rand(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            print(f"✅ GPU computation successful! Result shape: {z.shape}")
        except Exception as e:
            print(f"❌ GPU computation failed: {e}")

        # Run nvidia-smi
        try:
            print("\nRunning nvidia-smi:")
            subprocess.run(["nvidia-smi"], check=True)
        except Exception as e:
            print(f"❌ Failed to run nvidia-smi: {e}")

    except ImportError:
        print("❌ PyTorch is not installed.")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()

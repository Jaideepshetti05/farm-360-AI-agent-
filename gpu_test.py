import torch

print("PyTorch:", torch.__version__)

print("CUDA Available:", torch.cuda.is_available())

print("Has XPU:", hasattr(torch, "xpu"))

if hasattr(torch, "xpu"):
    try:
        print("XPU Available:", torch.xpu.is_available())
        if torch.xpu.is_available():
            print("Device:", torch.xpu.get_device_name(0))
    except Exception as e:
        print(e)
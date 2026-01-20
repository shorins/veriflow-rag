import torch
import platform

print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")

# Проверяем доступность MPS (Metal Performance Shaders) - это GPU на Mac
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print("✅ Успех! MPS (Apple Silicon GPU) доступен и работает.")
else:
    print("❌ Ошибка! MPS недоступен. Будет использоваться медленный CPU.")
# Environment requirements

- Python 3.10 or later.
- NVIDIA GPU with hardware support for double-precision (FP64) arithmetic, and a CUDA-enabled PyTorch build compatible with the installed driver.
- PyTorch >= 2.4, NumPy >= 1.26, SciPy >= 1.11, Matplotlib >= 3.8.
- Training, reference-solution computation, and model evaluation require CUDA.

The numerical code uses `torch.float64`. High FP64 throughput is recommended
for practical runtime, especially for the spectral operators and reference solves.
Many consumer GPUs support FP64 correctly but have much lower FP64 throughput
than scientific-computing GPUs; this is primarily a performance limitation,
not inherently inaccurate double-precision arithmetic. FP32, TF32, or
mixed-precision computation should not be substituted for the required FP64.

Checked environment: Python 3.12.13, PyTorch 2.11.0+cu126, NumPy 2.4.3,
SciPy 1.17.1, Matplotlib 3.10.8, NVIDIA A100-SXM4-80GB.

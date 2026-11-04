# GR00T N1.5 / RoboCasa365: Blackwell CUDA12.8 profile

This is a separate candidate runtime for RTX PRO 6000 Blackwell. It preserves
the four source revisions and all non-CUDA scientific dependencies of
`gr00t-n15-robocasa365-20261019`. It does not replace that runtime in place.

- Python3.11.14; uv0.12.0.
- Torch2.7.1+cu128, torchvision0.22.1+cu128, Triton3.3.1.
- FlashAttention2.8.3.post2, CUDA12 / Torch2.7 / C++11 ABI wheel.
- Exact CUDA library versions in `requirements.txt`; official Torch/vision
  wheel URLs and source commits in `profile.json`.
- Linux x86_64, glibc >=2.28. Driver580.178.04 supports the CUDA12.8 runtime.

[PyTorch2.7](https://pytorch.org/blog/pytorch-2-7/) added Blackwell support with
CUDA12.8 wheels. The [FlashAttention release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2)
supplies the pinned wheel. Driver compatibility follows NVIDIA's
[CUDA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

A fresh installation on2026-11-03 passed282 package/version checks, pinned-source
checks, and core imports. Torch reports `sm_120` among its built targets; the
FlashAttention extension imports and contains `sm_120` binary metadata. These
checks **do not prove execution on a Blackwell GPU**. Status remains
`candidate_requires_target_gpu_smoke` until the target machine passes runtime
GPU verification, multi-GPU training and saved-checkpoint evaluation smoke.

Use the machine JSON and unified installer in the
[Slurm guide](../../../docs/opd/collaboration/SLURM_BLACKWELL.md).
Training and independent evaluation select the same profile. Do not mix this
FlashAttention wheel with Torch2.6 or the old C++ ABI.

# GR00T N1.5 / LIBERO on Blackwell

This separate runtime profile preserves the model and simulator dependencies of
`gr00t-n15-libero-20261105-r2`, while selecting the project's existing CUDA12.8
GPU stack. It does not replace an A6000 runtime in place.

- Python3.11.14, bootstrap uv0.10.4.
- Torch2.7.1+cu128, torchvision0.22.1+cu128, Triton3.3.1, SymPy1.14.0.
- FlashAttention2.8.3.post2, official CUDA12/Torch2.7/C++11-ABI wheel.
- NCCL2.31.2: replaces the original Torch2.7 wheel dependency2.26.2 after a real
  object-collective/DCP failure was reproduced on action30.
- Same pinned GR00T and LIBERO revisions, Robosuite1.4.1, MuJoCo3.3.7,
  Transformers4.51.3, Hydra1.4.0.dev6 and OmegaConf2.4.0.dev13.
- Same exact non-GPU package pins; the additional CUDA cufile dependency and
  CUDA library versions are recorded in `requirements.txt`.

[PyTorch2.7](https://pytorch.org/blog/pytorch-2-7/) introduced Blackwell support
with CUDA12.8 wheels. The official
[FlashAttention release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2)
provides the pinned extension. `profile.json` records exact public wheel URLs;
the installer installs Torch/vision wheels before the no-dependency snapshot.

## Restore and verify

Follow the bootstrap procedure in [LIBERO_ENVIRONMENT.md](../../../LIBERO_ENVIRONMENT.md),
using new absolute runtime/bootstrap paths and the explicit Blackwell profile.

Use the numbered manual steps in that guide. After installing the pins and
sources, generate the scoped wrapper:

```bash
python3 -m toolkits.libero_gr00t.prepare_runtime \
  --profile gr00t-n15-libero-blackwell-cu128 --runtime "$LIBERO_RUNTIME"
```

On an allocated GPU, use `"$LIBERO_RUNTIME/bin/python"` for verification,
training and evaluation. Do not hide all GPUs for full model-import verification:
RADIO can query CUDA at import time. The guide includes a separate CPU-only
package-metadata check. Package verification alone is not training acceptance.

## Required SDPA backend check

Actual RTX PRO6000 Blackwell execution on2026-11-06 exposed a silent error in
Torch2.7.1's FP32 `EFFICIENT_ATTENTION` SDPA backend at head dimension128. A retained
synthetic test differed from explicit FP64 math by0.275 maximum absolute error;
forced `MATH` differed by less than9e-7. BF16 backend checks and external
FlashAttention agreed with the mathematical reference at BF16 precision.

The action30 entrypoint therefore disables **only** PyTorch memory-efficient
SDPA before model imports:

```python
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
```

Apply this in every process, including Ray workers and independent evaluation.
The action30 node wrapper uses a scoped `sitecustomize.py` hook selected by
`OPD_BLACKWELL_SDPA_SAFE=1`. The portable `prepare_runtime.py` wrapper now enables
this policy for this profile in every Python subprocess. Installing packages alone does not activate
it. A new-node setup must use the wrapper and rerun numerical checks; package
versions or finite outputs alone do not establish correctness. External FlashAttention and PyTorch's other SDPA backends
remain available. This changes backend selection, not the attention definition,
loss, checkpoint metadata or experiment hyperparameters.

The initial failing test and the corrected validation are preserved separately.
Full runtime acceptance additionally requires actual EGL UUID checks, multi-GPU
transport/FSDP, model updates, strict checkpoint resume and independent inference.

## NCCL runtime amendment

On action30, the original NCCL2.26.2 library failed a tiny object-collective
reproducer, independently of robotics or model code. Installing only
`nvidia-nccl-cu12==2.31.2` in the new runtime made the same four-rank object
collectives and miniature FSDP/DCP save pass. The package pin now records this
amendment. An analogous upstream report is [NVIDIA/nccl#2418](https://github.com/NVIDIA/nccl/issues/2418);
our acceptance is based on the local reproducer rather than that report alone.

Verify the loaded shared library with `ncclGetVersion` or NCCL debug output.
In this Torch build, `torch.cuda.nccl.version()` still reports its compile-time
2.26.2 value despite the loaded2.31.2 runtime; it is insufficient for this check.
There is no generic `NCCL=1` switch. Existing `NCCL_P2P_DISABLE=1` and
`NCCL_IB_DISABLE=1` did not fix the reproduced failure. Do not modify the system
NCCL or another experiment's environment.

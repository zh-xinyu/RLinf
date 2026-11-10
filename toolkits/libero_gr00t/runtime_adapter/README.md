# Process-local runtime adapters

These hooks package the adapters previously used by the validated LIBERO node
wrappers. Only `prepare_runtime.py`'s explicit Python entrypoint activates them;
they do not modify installed packages or a user's global Python environment.

- `sitecustomize.py` selects the two Robosuite adapters when
  `OPD_LIBERO_AUTOMAP_EGL=1`. For the CUDA12.8 profile it also disables PyTorch's
  memory-efficient SDPA backend before model loading, preserving the tested
  Blackwell policy. External FlashAttention and other SDPA backends are unchanged.
- `binding_utils.py` is derived from Robosuite1.4.1. It relaxes the original
  physical-ID membership assertion only when automatic EGL mapping is enabled.
- `egl_context.py` retains Robosuite's context creation and uses
  `EGL_CUDA_DEVICE_NV` to find CUDA logical device zero. Mapping fails if there
  is no unique match. Container GPU identifiers bypass the legacy physical-ID
  range check before mapping; this small portability correction has CPU
  regression coverage and still requires a target-container EGL admission test.

Robosuite-derived code retains its upstream notices. See
[LICENSE.robosuite](LICENSE.robosuite) for its MIT terms and embedded DeepMind
MuJoCo attribution; `egl_context.py` also retains the upstream Apache2 header.
The project Apache license is available at the repository root.

Do not put this directory in a global `site-packages` or shell startup file.
Package verification does not establish EGL/GPU identity or numerical parity:
new hardware must pass the acceptance steps in
[LIBERO_ENVIRONMENT.md](../../../LIBERO_ENVIRONMENT.md).

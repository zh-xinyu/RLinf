# LIBERO profile revision 2 — 2026-11-05

This revision copies `gr00t-n15-libero-20261105` and changes only the profile
identifier plus two runtime pins: Hydra 1.4.0.dev6 and OmegaConf 2.4.0.dev13.
These exact versions were confirmed in the working action01 and action10
runtimes. All other package pins, source commits, build constraints and the
FlashAttention wheel are unchanged; the distribution count remains 265.

The original profile passed package/source/import checks, teacher rollouts and
a bounded target audit, but Hydra 1.4.0rc3 rejected the real training entrypoint's
`@hydra.main(version_base="1.1")` decorator before model initialization. Import
and launcher-config checks did not catch that entrypoint compatibility gap.
The original profile is retained unchanged for historical and active-run
provenance; it is not recommended for new training launches.

Revision gates require strict package/source verification and execution of the
actual `examples/embodiment/train_embodied_agent.py --cfg job --resolve` path
with the resolved experiment config, GPUs hidden and no Ray startup. A fresh
revision-2 restore passed strict verification of all 265 distributions and this
actual entrypoint check. Separately, an idle runtime amended with exactly these
two package replacements passed the same package/source verification and the
actual entrypoint using the T32/B256/micro1 training configuration. The amendment
is recorded separately from the fresh restore and preserves the original runtime
receipts as historical evidence. Neither check launched Ray or used a GPU.

No revised-profile student training, successful optimizer update or learning
improvement is claimed by these installation and entrypoint checks.

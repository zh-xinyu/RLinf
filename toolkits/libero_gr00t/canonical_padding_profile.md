# Experimental canonical fixed570 profile

This profile is opt-in and deferred pending native-policy continuation. No formal
candidate YAML defaults change. Canonical fixed570 changes VLM inputs and can
change actions and success rates; it is not throughput-equivalent native batching.

The production launcher accepts `--student-padding-profile canonical-fixed570`
only with `--config-name libero_mopd_candidate_m1`. It sets globalB256, microbatch4,
T32, Refresh-R2, LR3e-6:16 microbatches/rank on four GPUs, the same accumulation
count as nativeB64/micro1 but a different update batch. Both student actor and
rollout use `opd_fixed_text_padding=true`, `opd_canonical_fixed_padding=true` and
padding570. Each teacher explicitly retains canonical=False and its existing
legacy fixed-padding inference. A native checkpoint cannot be resumed into this
changed recipe; start from the original joint SFT initialization.

Use a new dry-run output to inspect the resolved profile before any GPU work.
Add the profile argument to the usual M1 launcher command; retain full native
horizons with `--rollout-steps 520`. The launch manifest records the policy input
contract and batch recipe. Native is still the default and preserves its current
training parameters. Resume comparison permits only the historical absence of
`opd_canonical_fixed_padding` versus explicit booleanFalse; True or false-like
values such as0 are not equivalent and are rejected.

Before a learning pilot, this changed setting requires:

1. GPU B1/B4 parity checks under the canonical contract.
2. A newly measured baseline from the original SFT weights with this exact profile.
3. A bounded production memory/update/save/reload smoke at B256/micro4.
4. A bounded learning pilot compared with that new baseline, with the profile
   fixed across every checkpoint and evaluator.

`evaluate_full_checkpoint.py` and `launch_evaluation.py` also accept
`--student-padding-profile canonical-fixed570`; omitting it selects native.
The standalone evaluator reads the saved training model contract and rejects
mismatches. Use `--baseline` with the canonical resolved config for the new baseline,
then `--weights` for trained checkpoints. Both write the explicit student input
identity. The distributed evaluator requires the same identity in a trained
checkpoint's standalone validation protocol, including the canonical flag. Old
native protocols remain valid when this newly introduced flag is absent orFalse.

A fixed570 baseline must not be compared directly against the old native baseline
to claim training improvement. Hold evaluation task/state/RNG protocols constant
within the new profile. The profile is experimental scaffolding only; no new GPU
run or formal readiness is implied by config and CPU test acceptance.

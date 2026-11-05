# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Explicit student input contracts for native and experimental fixed text.

Canonical fixed padding changes the policy's VLM inputs. It is not an equivalent
batching implementation of native prompts and requires a matched baseline.
"""

from omegaconf import OmegaConf, open_dict

PADDING_PROFILES = ("native", "canonical-fixed570")


def student_input_contract(model, profile: str = "native") -> dict:
    """Validate padding flags and return the recorded policy-input identity."""
    if profile not in PADDING_PROFILES:
        raise ValueError(f"Unknown student padding profile: {profile}")
    canonical = profile == "canonical-fixed570"
    fixed = model.get("opd_fixed_text_padding")
    actual_canonical = model.get("opd_canonical_fixed_padding", False)
    if type(fixed) is not bool or fixed != canonical:
        raise ValueError(f"opd_fixed_text_padding differs from profile {profile}")
    if type(actual_canonical) is not bool or actual_canonical != canonical:
        raise ValueError(f"opd_canonical_fixed_padding differs from profile {profile}")
    width = OmegaConf.select(model, "rl_head_config.padding_value")
    if canonical and (type(width) is not int or width != 570):
        raise ValueError("canonical-fixed570 requires integer padding_value=570")
    return {
        "student_padding_profile": profile,
        "opd_fixed_text_padding": fixed,
        "opd_canonical_fixed_padding": actual_canonical,
        "fixed_text_length": 570 if canonical else None,
        "native_policy_equivalent": not canonical,
    }


def apply_student_padding_profile(model, profile: str = "native") -> dict:
    """Apply only explicit experimental overrides; preserve historical native keys."""
    if profile not in PADDING_PROFILES:
        raise ValueError(f"Unknown student padding profile: {profile}")
    if profile == "canonical-fixed570":
        with open_dict(model):
            model.opd_fixed_text_padding = True
            model.opd_canonical_fixed_padding = True
            model.rl_head_config.padding_value = 570
    return student_input_contract(model, profile)

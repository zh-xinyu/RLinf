# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit coverage of rollout seeding; not a distributed/GPU replay test."""

import ast
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def extracted_function(path, name, globals_):
    tree = ast.parse(path.read_text())
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    module = ast.Module(body=matches, type_ignores=[])
    exec(compile(module, str(path), "exec"), globals_)
    return globals_[name]


@pytest.fixture(autouse=True)
def preserve_rng_state(monkeypatch):
    """Isolate CPU streams and suppress CUDA side effects of torch.manual_seed."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


@pytest.fixture
def seed_method():
    real_seed = extracted_function(
        ROOT / "rlinf/utils/utils.py",
        "seed_everything",
        {"random": random, "np": np, "torch": torch, "_UINT32_MOD": 2**32},
    )
    return extracted_function(
        ROOT / "rlinf/workers/rollout/hf/huggingface_worker.py",
        "_initialize_rollout_rng",
        {"seed_everything": real_seed},
    )


def worker(seed, rank=0):
    return SimpleNamespace(
        cfg=SimpleNamespace(rollout=seed), _rank=rank, log_info=lambda message: None
    )


def draws():
    return random.random(), np.random.random(), torch.rand(4)


def same(a, b):
    assert a[:2] == b[:2]
    assert torch.equal(a[2], b[2])


@pytest.mark.parametrize("cfg", [{}, {"seed": None}])
def test_missing_seed_leaves_all_streams_unchanged(seed_method, cfg):
    w = worker({"seed": 17})
    seed_method(w)
    expected = draws()
    seed_method(w)
    optional = worker(cfg)
    seed_method(optional)
    assert optional.rollout_rng_seed is None
    same(draws(), expected)


def test_repeated_rank_stream_and_distinct_ranks(seed_method):
    w = worker({"seed": 123}, rank=2)
    seed_method(w)
    first, second = draws(), draws()
    seed_method(w)
    same(first, draws())
    same(second, draws())
    assert w.rollout_rng_seed == 125
    other = worker({"seed": 123}, rank=3)
    seed_method(other)
    assert other.rollout_rng_seed == 126
    assert not torch.equal(first[2], draws()[2])


@pytest.mark.parametrize("value", [True, 1.5, "3", -1, 2**64])
def test_invalid_base_seed_is_rejected(seed_method, value):
    with pytest.raises(ValueError, match="rollout.seed"):
        seed_method(worker({"seed": value}))


def test_derived_seed_overflow_is_rejected(seed_method):
    with pytest.raises(ValueError, match="rollout.seed"):
        seed_method(worker({"seed": 2**64 - 1}, rank=1))


def test_shared_constructor_hook_covers_frozen_override():
    tree = ast.parse(
        (ROOT / "rlinf/workers/rollout/hf/huggingface_worker.py").read_text()
    )
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    init = next(n for n in cls.body if getattr(n, "name", None) == "__init__")
    calls = [
        n
        for n in ast.walk(init)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_initialize_rollout_rng"
    ]
    assert len(calls) == 1
    assert init.body[-1].value is calls[0]
    frozen_tree = ast.parse(
        (ROOT / "toolkits/robotwin_mopd/teacher_train_rollout.py").read_text()
    )
    frozen = next(
        n
        for n in ast.walk(frozen_tree)
        if isinstance(n, ast.ClassDef) and n.name == "FrozenTeacherRolloutWorker"
    )
    assert [ast.unparse(base) for base in frozen.bases] == ["MultiStepRolloutWorker"]
    assert not any(getattr(n, "name", None) == "__init__" for n in frozen.body)
    assert any(getattr(n, "name", None) == "init_worker" for n in frozen.body)

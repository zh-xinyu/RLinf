# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Verify the frozen model inputs and bind the calibration to machine-local paths."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from evaluations.robocasa365.formal_eval import sha256

REPO = Path(__file__).resolve().parents[2]
INPUTS = REPO / "docs/opd/collaboration"


def checkpoint_paths(student: Path, teachers: Path) -> dict[str, Path]:
    """Map the four logical inputs to their local checkpoint directories."""
    return {
        "Joint-SFT-Full": student.resolve(strict=True),
        **{
            suite: (
                teachers / "target_posttraining" / suite / "checkpoint-60000"
            ).resolve(strict=True)
            for suite in ("atomic_seen", "composite_seen", "composite_unseen")
        },
    }


def verify_files(path: Path, expected: dict) -> None:
    """Reject missing, corrupted, or different checkpoint files."""
    for relative, record in expected.items():
        file = path / relative
        if not file.is_file() or file.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing file or incorrect size: {file}")
        if sha256(file) != record["sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {file}")


def prepare(student: Path, teachers: Path, output: Path) -> None:
    """Verify all checkpoints, then write a relocated but otherwise unchanged reference."""
    manifest = json.loads((INPUTS / "checkpoints.json").read_text())
    paths = checkpoint_paths(student, teachers)
    for name, path in paths.items():
        verify_files(path, manifest["checkpoints"][name])
        print(f"Verified {name}", flush=True)
    reference = json.loads((INPUTS / "teacher_reference.portable.json").read_text())
    reference["teacher_checkpoints"] = {
        name: str(path) for name, path in paths.items() if name != "Joint-SFT-Full"
    }
    reference["checkpoint_manifest_sha256"] = sha256(INPUTS / "checkpoints.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(reference, stream, indent=2)
        stream.write("\n")
    print(f"export OPD_TEACHER_REFERENCE={output.resolve()}")


def download_teachers(destination: Path) -> None:
    """Download only teacher inference files at the manifest's fixed HF revision."""
    from huggingface_hub import snapshot_download

    manifest = json.loads((INPUTS / "checkpoints.json").read_text())
    prefix = manifest["teacher_hf_prefix"]
    patterns = [
        f"{prefix}/target_posttraining/{suite}/checkpoint-60000/{file}"
        for suite, files in manifest["checkpoints"].items()
        if suite != "Joint-SFT-Full"
        for file in files
    ]
    snapshot_download(
        repo_id=manifest["teacher_hf_repo"],
        revision=manifest["teacher_hf_revision"],
        allow_patterns=patterns,
        local_dir=str(destination),
    )
    print(f"export OPD_MODEL_ROOT={destination.resolve() / prefix}")


def download_student(destination: Path) -> None:
    """Download the five pinned inference files, refusing to replace existing bytes."""
    from huggingface_hub import hf_hub_download

    manifest = json.loads((INPUTS / "checkpoints.json").read_text())
    source = manifest["student_initialization"]
    for relative, record in manifest["checkpoints"]["Joint-SFT-Full"].items():
        target = destination / relative
        if target.exists():
            verify_files(destination, {relative: record})
            continue
        cached = hf_hub_download(
            repo_id=source["hf_repo"],
            revision=source["hf_revision"],
            filename=f"{source['hf_prefix']}/{relative}",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent) as temporary:
            with open(cached, "rb") as stream:
                shutil.copyfileobj(stream, temporary)
            temporary.flush()
            if (
                Path(temporary.name).stat().st_size != record["bytes"]
                or sha256(Path(temporary.name)) != record["sha256"]
            ):
                raise ValueError(f"Downloaded checkpoint hash mismatch: {relative}")
            # Atomic creation without replacing another process's file.
            os.link(temporary.name, target)
    verify_files(destination, manifest["checkpoints"]["Joint-SFT-Full"])
    print(f"Verified Joint-SFT-Full at {destination}")


def main() -> None:
    """Prepare resources explicitly; never upload models or credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download-teachers")
    download.add_argument("--destination", type=Path, required=True)
    student = sub.add_parser("download-student")
    student.add_argument("--destination", type=Path, required=True)
    bind = sub.add_parser("prepare")
    bind.add_argument(
        "--student", type=Path, default=os.environ.get("OPD_STUDENT_CHECKPOINT")
    )
    bind.add_argument("--teachers", type=Path, default=os.environ.get("OPD_MODEL_ROOT"))
    bind.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download-teachers":
        download_teachers(args.destination)
    elif args.command == "download-student":
        download_student(args.destination)
    else:
        if args.student is None or args.teachers is None:
            parser.error(
                "Set --student/--teachers or OPD_STUDENT_CHECKPOINT/OPD_MODEL_ROOT"
            )
        prepare(args.student, args.teachers, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write or verify the repository-wide SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS"
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "reproduced",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def candidate_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_manifest() -> None:
    lines = [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in candidate_files()
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(lines)} entries")


def read_manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        raise RuntimeError("SHA256SUMS is missing; run checksums.py --write")
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        MANIFEST.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            checksum, name = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(f"malformed SHA256SUMS line {line_number}") from exc
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise RuntimeError(f"invalid SHA-256 on line {line_number}")
        if name in entries:
            raise RuntimeError(f"duplicate SHA256SUMS entry: {name}")
        entries[name] = checksum
    return entries


def verify_manifest() -> None:
    expected = read_manifest()
    current_paths = {
        path.relative_to(ROOT).as_posix(): path for path in candidate_files()
    }
    missing = sorted(set(expected) - set(current_paths))
    unexpected = sorted(set(current_paths) - set(expected))
    changed = sorted(
        name
        for name in set(expected) & set(current_paths)
        if digest(current_paths[name]) != expected[name]
    )
    if missing or unexpected or changed:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("not in manifest: " + ", ".join(unexpected))
        if changed:
            details.append("checksum mismatch: " + ", ".join(changed))
        raise RuntimeError("SHA256 verification failed\n" + "\n".join(details))
    print(f"verified {len(expected)} SHA-256 entries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="replace SHA256SUMS")
    args = parser.parse_args()
    if args.write:
        write_manifest()
    else:
        verify_manifest()


if __name__ == "__main__":
    main()

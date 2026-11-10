"""Clone SGLang v0.5.21 and apply the verified deadline-floor runtime patch."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

SGLANG_REPOSITORY = "https://github.com/sgl-project/sglang.git"
SGLANG_COMMIT = "e00930c5489053f26d86b179cee0d087f846acbb"
PATCH_NAME = "sglang-v0.5.21-deadline-floor.patch"
PATCH_SHA256 = "bd75caa7c1d1b0843ad8af9819545059b2c84e5d17d7812d3a93cdf962e6a7f2"


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    destination = args.destination.expanduser().resolve()
    if destination.exists():
        parser.error("destination already exists; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)

    patch = Path(__file__).resolve().parents[1] / "runtime" / PATCH_NAME
    if not patch.is_file():
        parser.error(f"runtime patch is missing: {patch}")

    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    if digest != PATCH_SHA256:
        parser.error("runtime patch SHA256 mismatch")

    run(
        "git",
        "clone",
        "--filter=blob:none",
        "--no-checkout",
        SGLANG_REPOSITORY,
        str(destination),
    )
    try:
        run("git", "-C", str(destination), "checkout", "--detach", SGLANG_COMMIT)
        run("git", "-C", str(destination), "apply", "--check", str(patch))
        run("git", "-C", str(destination), "apply", str(patch))
    except Exception:
        print(f"Preparation failed; inspect or remove {destination}")
        raise

    print(f"Prepared runtime: {destination}")
    print(f"SGLang base: {SGLANG_COMMIT}")
    print(f"Patch SHA256: {digest}")


if __name__ == "__main__":
    main()

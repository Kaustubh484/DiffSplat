from __future__ import annotations

import shutil
import sys
from pathlib import Path

from diffsplat_tools.common import require_repo, run
from diffsplat_tools.constants import REPO_URL


def clone_repo(repo_dir: Path, ref: str | None, dry_run: bool) -> None:
    if repo_dir.exists():
        print(f"Repo already exists at {repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], dry_run=dry_run)
    if ref:
        run(["git", "fetch", "--depth", "1", "origin", ref], cwd=repo_dir, dry_run=dry_run)
        run(["git", "checkout", ref], cwd=repo_dir, dry_run=dry_run)


def install_repo(
    repo_dir: Path,
    *,
    torch_channel: str,
    skip_xformers: bool,
    skip_rasterizer: bool,
    dry_run: bool,
) -> None:
    require_repo(repo_dir, dry_run=dry_run)
    pip = [sys.executable, "-m", "pip"]
    torch_index = f"https://download.pytorch.org/whl/{torch_channel}"

    run([*pip, "install", "-U", "pip", "setuptools", "wheel"], dry_run=dry_run)
    run(
        [
            *pip,
            "install",
            "-U",
            "torch==2.3.1",
            "torchvision==0.18.1",
            "torchaudio==2.3.1",
            "--index-url",
            torch_index,
        ],
        dry_run=dry_run,
    )
    if not skip_xformers:
        run(
            [
                *pip,
                "install",
                "-U",
                "xformers==0.0.27",
                "--index-url",
                torch_index,
            ],
            dry_run=dry_run,
        )

    run([*pip, "install", "-U", "gpustat", "huggingface_hub", "scikit-image"], dry_run=dry_run)
    run([*pip, "install", "-U", "-r", "settings/requirements.txt"], cwd=repo_dir, dry_run=dry_run)

    if not skip_rasterizer:
        rade_dir = repo_dir / "extensions" / "RaDe-GS"
        if not rade_dir.exists():
            run(
                [
                    "git",
                    "clone",
                    "https://github.com/BaowenZ/RaDe-GS.git",
                    "--recursive",
                    str(rade_dir),
                ],
                cwd=repo_dir / "extensions",
                dry_run=dry_run,
            )
        else:
            print(f"Rasterizer repo already exists at {rade_dir}")
        run(
            [*pip, "install", "./diff-gaussian-rasterization"],
            cwd=rade_dir / "submodules",
            dry_run=dry_run,
        )

    # --no-build-isolation reuses already-installed packages instead of creating an
    # isolated build env. Required on Python 3.12 where distutils was removed and
    # many setup.py-based packages break inside an isolated build.
    run([*pip, "install", "--no-build-isolation", "-e", "extensions/diffusers_diffsplat"], cwd=repo_dir, dry_run=dry_run)
    if shutil.which("ffmpeg") is None:
        print("ffmpeg was not found on PATH. GIF output usually works, but mp4 export may need a system ffmpeg install.")


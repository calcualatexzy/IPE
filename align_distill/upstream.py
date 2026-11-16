"""The paper's repository (github.com/chloeli-15/model_spec_midtraining), cloned unmodified at a pinned commit into
third_party/ by scripts/align_distill/fetch.sh. align-distill imports its agentic misalignment eval from there."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/chloeli-15/model_spec_midtraining.git"
COMMIT = "e8288a84912ba32af68ad15f2e52a7c1b4e81891"  # "Vendor agentic_misalignment classifiers", 2026-06-16
DIR = ROOT / "third_party" / "model_spec_midtraining"


def git(*args: str, cwd: Path = DIR) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def check(directory: Path = DIR) -> str:
    """Raise unless the clone exists, sits at COMMIT and has no local changes; return the commit."""
    if not (directory / ".git").exists():
        raise FileNotFoundError(f"{directory} is missing; run scripts/align_distill/fetch.sh (group code) first")
    head = git("rev-parse", "HEAD", cwd=directory)
    if head != COMMIT:
        raise RuntimeError(f"{directory} is at {head}, not the pinned {COMMIT}; run scripts/align_distill/fetch.sh")
    changes = git("status", "--porcelain", "--untracked-files=no", cwd=directory)
    if changes:
        raise RuntimeError(f"{directory} has local changes; the paper's code is used unmodified:\n{changes}")
    return head


def clone(directory: Path = DIR) -> str:
    """Clone the repository (without its safety-tooling submodule, which the eval does not use) and check out COMMIT.
    An existing clone is moved to COMMIT; local changes stop the run."""
    if not (directory / ".git").exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--quiet", URL, str(directory), cwd=directory.parent)
    if git("status", "--porcelain", "--untracked-files=no", cwd=directory):
        raise RuntimeError(f"{directory} has local changes; remove them before fetching")
    try:
        git("cat-file", "-e", f"{COMMIT}^{{commit}}", cwd=directory)
    except subprocess.CalledProcessError:
        git("fetch", "--quiet", "origin", cwd=directory)
    git("checkout", "--quiet", "--detach", COMMIT, cwd=directory)
    return check(directory)


def import_path(directory: Path = DIR) -> None:
    """Make the clone's top-level packages (evals, src) importable, after checking the pin."""
    check(directory)
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

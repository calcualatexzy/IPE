"""Code align-distill uses unmodified, cloned at a pinned commit into third_party/ by scripts/align_distill/fetch.sh:
    MSM        the paper's repository (github.com/chloeli-15/model_spec_midtraining): its agentic misalignment eval
    SPP_EVALS  SPP's evaluation harness (github.com/epfl-dlab/spp-evals): the ConstitutionEval and AIRisk protocols"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Pin:
    url: str
    commit: str
    dir: Path


MSM = Pin("https://github.com/chloeli-15/model_spec_midtraining.git",
          "e8288a84912ba32af68ad15f2e52a7c1b4e81891",  # "Vendor agentic_misalignment classifiers", 2026-06-16
          ROOT / "third_party" / "model_spec_midtraining")
SPP_EVALS = Pin("https://github.com/epfl-dlab/spp-evals.git",
                "cf545a077c585a82f3b81c046c2d624e3001e0ba",  # main on 2026-11-24 (pushed 2026-08-14), Apache-2.0
                ROOT / "third_party" / "spp-evals")
PINS = (MSM, SPP_EVALS)
# The paper's repository under its old names, which the AM eval and its records use
URL, COMMIT, DIR = MSM.url, MSM.commit, MSM.dir


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def check(pin: Pin = MSM) -> str:
    """Raise unless the clone exists, sits at its commit and has no local changes; return the commit."""
    if not (pin.dir / ".git").exists():
        raise FileNotFoundError(f"{pin.dir} is missing; run scripts/align_distill/fetch.sh (group code) first")
    head = git("rev-parse", "HEAD", cwd=pin.dir)
    if head != pin.commit:
        raise RuntimeError(f"{pin.dir} is at {head}, not the pinned {pin.commit}; run scripts/align_distill/fetch.sh")
    changes = git("status", "--porcelain", "--untracked-files=no", cwd=pin.dir)
    if changes:
        raise RuntimeError(f"{pin.dir} has local changes; its code is used unmodified:\n{changes}")
    return head


def clone(pin: Pin = MSM) -> str:
    """Clone the repository (without submodules) and check out its commit. An existing clone is moved to the commit;
    local changes stop the run."""
    if not (pin.dir / ".git").exists():
        pin.dir.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--quiet", pin.url, str(pin.dir), cwd=pin.dir.parent)
    if git("status", "--porcelain", "--untracked-files=no", cwd=pin.dir):
        raise RuntimeError(f"{pin.dir} has local changes; remove them before fetching")
    try:
        git("cat-file", "-e", f"{pin.commit}^{{commit}}", cwd=pin.dir)
    except subprocess.CalledProcessError:
        git("fetch", "--quiet", "origin", cwd=pin.dir)
    git("checkout", "--quiet", "--detach", pin.commit, cwd=pin.dir)
    return check(pin)


def import_path(pin: Pin = MSM) -> None:
    """Make the clone's top-level packages importable (the paper's evals and src), after checking the pin."""
    check(pin)
    if str(pin.dir) not in sys.path:
        sys.path.insert(0, str(pin.dir))


def pinned_dataset(repo_id: str, revision: str) -> Path:
    """The cached snapshot of a Hub dataset, read offline, which must be the pinned revision (fetch.sh fetches main and
    records the commit it resolved to; a newer one stops the run here)."""
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(repo_id, repo_type="dataset", local_files_only=True))
    except Exception as e:  # LocalEntryNotFoundError and friends
        raise FileNotFoundError(f"{repo_id} is not in the Hugging Face cache; fetch it with "
                                f"scripts/align_distill/fetch.sh (group data)") from e
    if path.name != revision:
        raise RuntimeError(f"{repo_id}: the cached snapshot is {path.name}, not the pinned {revision}")
    return path


def load_modules(pin: Pin, folder: str, names: tuple[str, ...]) -> dict[str, ModuleType]:
    """Load the modules folder/<name>.py of a clone, in order, after checking the pin. Sibling modules that import each
    other by bare name (SPP's "from elo import ...") see each other while loading; afterwards neither the bare names
    nor the folder stay importable, so they cannot shadow anything else."""
    check(pin)
    directory = pin.dir / folder
    saved_modules = {name: sys.modules.get(name) for name in names}
    saved_path = list(sys.path)
    modules: dict[str, ModuleType] = {}
    try:
        for name in names:
            spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        sys.path[:] = saved_path  # a module may insert its own folder (SPP's run_eval.py does)
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return modules

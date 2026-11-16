"""Download what the GPU jobs read into the shared Hugging Face cache, configured by align_distill/conf/fetch.yaml.

Jobs run with HF_HUB_OFFLINE=1, so base models, the paper's checkpoints and the datasets are fetched here first, and
the paper's repository is cloned at its pinned commit into third_party/. Each repository is fetched at main, and
the manifest records the commit main resolved to. Before downloading, the run prints every repository with the bytes it
still lacks, and refuses when the download would leave less than reserve_gb free on the cache's volume. Files already
in the cache are not fetched again, so rerunning the same command resumes or checks a fetch.

Usage (scripts/align_distill/fetch.sh chooses the settings):
    python -m align_distill.fetch [HYDRA_OVERRIDES...]
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from align_distill import upstream
from align_distill.models import model_row, paper_checkpoints

GROUPS = ("code", "base", "g0", "msm", "curves", "data")
GB = 1e9


@dataclass
class Repo:
    repo_id: str
    repo_type: str  # model or dataset
    why: str
    revision: str | None = None
    total_bytes: int = 0
    missing_bytes: int = 0
    files: int = 0


def wanted_repos(cfg: DictConfig) -> list[Repo]:
    """The repositories the selected groups need, each once, in group order."""
    unknown = set(cfg.groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown groups {sorted(unknown)}; choose from {GROUPS}")
    repos: dict[tuple[str, str], Repo] = {}

    def add(repo_id: str, repo_type: str, why: str) -> None:
        repos.setdefault((repo_id, repo_type), Repo(repo_id, repo_type, why))

    rows = [model_row(tag) for tag in cfg.models]
    for group in cfg.groups:
        if group == "base":
            for row in rows:
                add(row["hub_id"], "model", f"base {row['tag']}")
        elif group in ("g0", "msm", "curves"):
            for row in rows:
                for repo_id in paper_checkpoints(row, group):
                    add(repo_id, "model", f"{group} {row['tag']}")
        elif group == "data":
            for repo_id in cfg.datasets:
                add(repo_id, "dataset", "data")
    return list(repos.values())


def measure(api, repo: Repo) -> None:
    """Fill in the revision the Hub reports and the bytes, in total and still missing from the cache."""
    from huggingface_hub import try_to_load_from_cache

    info = (api.model_info if repo.repo_type == "model" else api.dataset_info)(repo.repo_id, files_metadata=True)
    repo.revision = info.sha
    repo.files = len(info.siblings)
    for sibling in info.siblings:
        size = sibling.size or 0
        repo.total_bytes += size
        cached = try_to_load_from_cache(repo.repo_id, sibling.rfilename, revision=info.sha, repo_type=repo.repo_type)
        if not isinstance(cached, str):
            repo.missing_bytes += size


def print_plan(repos: list[Repo], free_bytes: int) -> None:
    width = max([len(repo.repo_id) for repo in repos] + [10])
    print(f"\n{'Repository':<{width}}  {'Type':<7}  {'Group':<12}  {'Revision':<8}  {'Total GB':>8}  {'Missing GB':>10}")
    for repo in repos:
        print(f"{repo.repo_id:<{width}}  {repo.repo_type:<7}  {repo.why:<12}  {(repo.revision or '')[:8]:<8}  "
              f"{repo.total_bytes / GB:>8.2f}  {repo.missing_bytes / GB:>10.2f}")
    missing = sum(repo.missing_bytes for repo in repos)
    print(f"\n{len(repos)} repositories, {sum(r.total_bytes for r in repos) / GB:.1f} GB in all, "
          f"{missing / GB:.1f} GB to download; {free_bytes / GB:.0f} GB free on the cache's volume\n")


def update_manifest(path: Path, entries: dict[str, dict]) -> None:
    from msm.datagen.utils import load_json, save_json

    manifest = load_json(path) if path.exists() else {}
    manifest.update(entries)
    save_json(path, dict(sorted(manifest.items())))


@hydra.main(version_base=None, config_path="conf", config_name="fetch")
def main(cfg: DictConfig) -> None:
    from huggingface_hub import HfApi, constants, snapshot_download

    if constants.HF_HUB_OFFLINE:
        raise RuntimeError("HF_HUB_OFFLINE is set; fetching needs the Hub")
    stamp = datetime.now().isoformat(timespec="seconds")
    entries: dict[str, dict] = {}
    manifest = upstream.ROOT / cfg.manifest

    if "code" in cfg.groups:
        if cfg.dry_run:
            print(f"Would clone {upstream.URL} at {upstream.COMMIT[:8]} into {upstream.DIR}")
        else:
            commit = upstream.clone()
            print(f"{upstream.DIR.relative_to(upstream.ROOT)} is at the pinned {commit[:8]}, unmodified")
            entries[f"code:{upstream.URL}"] = {"revision": commit, "path": str(upstream.DIR), "fetched": stamp}

    repos = wanted_repos(cfg)
    if not repos:
        if entries:
            update_manifest(manifest, entries)
        return
    api = HfApi()
    for repo in repos:
        measure(api, repo)
    cache = Path(constants.HF_HUB_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free
    print(f"Cache: {cache}")
    print_plan(repos, free)
    missing = sum(repo.missing_bytes for repo in repos)
    if free - missing < cfg.reserve_gb * GB:
        raise RuntimeError(f"Downloading {missing / GB:.0f} GB would leave {(free - missing) / GB:.0f} GB free, "
                           f"below reserve_gb={cfg.reserve_gb}; free space or fetch fewer groups")
    if cfg.dry_run:
        print("dry_run: nothing downloaded")
        return

    # Each repository is recorded once it is complete, so an interrupted run keeps what it finished. The download asks
    # for main, not the commit measured above: only a branch download writes refs/main, which offline lookups (the
    # jobs' snapshot_download(local_files_only=True)) resolve. The commit main resolved to is the snapshot's name.
    for repo in repos:
        print(f"Fetching {repo.repo_type} {repo.repo_id} ({repo.missing_bytes / GB:.2f} GB missing)", flush=True)
        path = snapshot_download(repo.repo_id, repo_type=repo.repo_type, max_workers=cfg.max_workers)
        revision = Path(path).name
        if revision != repo.revision:
            print(f"  {repo.repo_id} moved from {repo.revision[:8]} to {revision[:8]} during the run")
        # what a job will see
        offline = snapshot_download(repo.repo_id, repo_type=repo.repo_type, local_files_only=True)
        if offline != path:
            raise RuntimeError(f"An offline lookup of {repo.repo_id} finds {offline}, not {path}")
        entries[f"{repo.repo_type}:{repo.repo_id}"] = {"revision": revision, "path": path, "why": repo.why,
                                                       "bytes": repo.total_bytes, "fetched": stamp}
        update_manifest(manifest, entries)

    # The chat template each model uses when an adapter ships none (with base, even when its checkpoints are not
    # fetched)
    if "base" in cfg.groups:
        from huggingface_hub import hf_hub_download

        for tag in cfg.models:
            repo_id = model_row(tag)["chat_template"]
            hf_hub_download(repo_id, "chat_template.jinja")
            print(f"Chat template of {tag}: {repo_id}")
    update_manifest(manifest, entries)
    print(f"\nDone; manifest in {manifest.relative_to(upstream.ROOT)}")


if __name__ == "__main__":
    main()

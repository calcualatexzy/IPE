#!/usr/bin/env python3
"""Generate midtraining documents from a spec.

Port of src/msm/generate_data_from_spec.py from github.com/chloeli-15/model_spec_midtraining, configured with Hydra
(msm/conf/datagen.yaml) and calling an OpenAI-compatible API (DeepSeek by default). The prompts in
msm/datagen/prompts/ and the specs in msm/spec/ are copied verbatim from upstream @ e8288a8 (src/msm/prompts/default/
and spec/paper/). That repository has no license, so keep them out of public pushes.

Steps: domains -> subdomains -> assertions -> doc types -> doc ideas -> documents -> dataset.jsonl

Each step saves its results under paths.gen_dir as they arrive and skips work already saved, so running the same
command again resumes a stopped run and retries whatever failed. Outputs, as upstream:
    data/msm/gen_synth_docs/<dataset_name>/    meta.json per level, one .txt per document, summary.json
    data/msm/midtrain/<dataset_name>/dataset.jsonl    {"text", "source", "domain"} per line, shuffled

Usage (scripts/msm/generate_msm_data.sh chooses the spec):
    python -m msm.datagen.generate_data_from_spec preview=true
    python -m msm.datagen.generate_data_from_spec spec.file=msm/spec/pro_america_cheese.txt spec.dataset_name=pro_america
"""

from __future__ import annotations

import asyncio
import json
import random
from functools import partial
from pathlib import Path
from typing import Awaitable, Callable, Iterator

import hydra
import openai
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from msm.datagen.llm import LLM
from msm.datagen.utils import extract_content, load_json, parse_json_response, sanitize_filename, save_json, write_atomic

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
TEMPLATE_FILES = {
    "domains": "spec2domains_template.txt",
    "subdomains": "spec2subdomains_template.txt",
    "assertions": "spec2assertions_template.txt",
    "doc_types": "spec2doc_type_template.txt",
    "doc_ideas": "spec2doc_idea_template.txt",
    "document": "spec2doc_template.txt",
}
# Errors that would fail every request the same way: stop the run instead of logging each item.
FATAL_API_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)

Job = tuple[str, Callable[[], Awaitable[None]]]  # (label for logs, work that saves its own result)


def resolve_path(path: str) -> Path:
    """Resolve a config path against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_spec(path: Path, model_name: str, provider_name: str) -> str:
    """Read a spec and fill in its {model_name} and {provider_name}.

    Upstream uses str.format, which fails on any other brace in the spec; this replaces only those two.
    """
    return path.read_text(encoding="utf-8").replace("{model_name}", model_name).replace("{provider_name}", provider_name)


def format_assertions_list(assertions: list[dict]) -> str:
    """Format assertions as a bulleted list with explanations."""
    return "\n".join(f"- {a['assertion']} (Explanation: {a['explanation']})" for a in assertions)


def existing_doc_types_note(existing: list[dict], n_doc_types: int) -> str:
    if not existing:
        return ""
    existing_list = "\n".join(f'- "{dt["doc_type"]}": {dt.get("description", "")}' for dt in existing)
    return (
        f"The following {len(existing)} doc types have already been generated for this subdomain:\n"
        f"{existing_list}\n\n"
        f"Do NOT repeat or closely rephrase any of the above. Generate {n_doc_types} additional doc types that are diverse and different — be creative!"
    )


def existing_doc_ideas_note(existing: list[dict], n_doc_ideas: int) -> str:
    if not existing:
        return ""
    existing_list = "\n".join(f'- "{idea["name"]}": {idea.get("idea", "")}' for idea in existing)
    return (
        f"The following {len(existing)} doc ideas already exist for this doc type:\n"
        f"{existing_list}\n\n"
        f"Do NOT repeat or closely rephrase any of the above. Generate {n_doc_ideas} additional ideas that are "
        f"meaningfully different — explore new angles, scenarios, and perspectives not yet covered."
    )


def validate_items(parsed, required_keys: list[str]) -> list[dict]:
    """Keep the items that have a non-empty string under every required key (upstream's validate_parsed_json)."""
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON list, got {type(parsed).__name__}")
    items = [item for item in parsed
             if isinstance(item, dict) and all(isinstance(item.get(k), str) and item[k].strip() for k in required_keys)]
    if len(items) < len(parsed):
        logger.warning(f"Dropped {len(parsed) - len(items)} of {len(parsed)} items without {required_keys}")
    if not items:
        raise ValueError(f"no items with {required_keys}")
    return items


def unique_by(items: list[dict], key: str, existing: list[dict] = ()) -> list[dict]:
    """Drop items whose directory name (sanitized `key`) repeats an earlier one, so they never share a directory."""
    seen = {sanitize_filename(item[key]) for item in existing}
    kept = []
    for item in items:
        name = sanitize_filename(item[key])
        if name and name not in seen:
            seen.add(name)
            kept.append(item)
    return kept


def with_unique_names(ideas: list[dict], existing: list[dict]) -> list[dict]:
    """Rename ideas whose document file name repeats an earlier idea's (upstream skips them silently)."""
    used = {sanitize_filename(idea["name"]) for idea in existing}
    renamed = []
    for idea in ideas:
        base = idea["name"] if sanitize_filename(idea["name"]) else "untitled"
        name, k = base, 2
        while sanitize_filename(name) in used:
            name, k = f"{base[:190]}_{k}", k + 1
        used.add(sanitize_filename(name))
        renamed.append({**idea, "name": name})
    return renamed


class DataGenerator:
    def __init__(self, cfg: DictConfig, llm=None):
        self.cfg = cfg
        self.gen_dir = resolve_path(cfg.paths.gen_dir)
        self.out_dir = resolve_path(cfg.paths.out_dir)
        self.templates = {stage: (PROMPTS_DIR / name).read_text() for stage, name in TEMPLATE_FILES.items()}
        spec_content = load_spec(resolve_path(cfg.spec.file), cfg.persona.model_name, cfg.persona.provider_name)
        # Fields every template may use; str.format ignores the ones a template does not.
        self.common_fields = {
            "principle_name": cfg.principle_name,
            "spec_content": spec_content,
            "model_name": cfg.persona.model_name,
            "provider_name": cfg.persona.provider_name,
        }
        self.llm = llm if llm is not None else LLM(cfg.api)
        self.failures: dict[str, int] = {}

    def _prompt(self, stage: str, **fields) -> str:
        return self.templates[stage].format(**self.common_fields, **fields)

    def _domain_dir(self, domain: str) -> Path:
        return self.gen_dir / sanitize_filename(domain)

    async def _ask_json(self, stage: str, user_text: str, list_key: str, required_keys: list[str]) -> list[dict]:
        """Request a JSON list of items with `required_keys`, asking again when a reply does not parse."""
        error = None
        for _ in range(self.cfg.attempts):
            reply = await self.llm.complete(user_text, self.cfg.max_tokens[stage], self.cfg.temperature)
            try:
                parsed = parse_json_response(reply.text)
                if isinstance(parsed, dict) and list_key in parsed:
                    parsed = parsed[list_key]
                return validate_items(parsed, required_keys)
            except ValueError as e:
                error = e
        raise ValueError(f"no valid {list_key} in {self.cfg.attempts} attempts: {error}")

    async def _run_jobs(self, stage: str, jobs: list[Job]) -> None:
        """Run the jobs concurrently. A failed job is logged and left for the next run, so one bad reply does not
        lose the others (upstream stops the whole step at the first one)."""
        if not jobs:
            logger.info(f"{stage}: nothing to do")
            return

        async def guarded(label: str, job: Callable[[], Awaitable[None]]) -> bool:
            try:
                await job()
                return True
            except FATAL_API_ERRORS:
                raise
            except Exception as e:
                logger.error(f"{stage}: {label}: {type(e).__name__}: {e}")
                return False

        succeeded = failed = 0
        tasks = [asyncio.create_task(guarded(label, job)) for label, job in jobs]
        with tqdm(total=len(tasks), desc=stage, unit="item") as pbar:
            for next_done in asyncio.as_completed(tasks):
                if await next_done:
                    succeeded += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix(success=succeeded, failed=failed)
        if failed:
            self.failures[stage] = failed
        logger.info(f"{stage}: {succeeded}/{len(jobs)} succeeded, {failed} failed")

    # Domains and subdomains

    async def get_domains(self) -> list[dict]:
        if self.cfg.specified_domains:
            domains = [{"domain": domain} for domain in self.cfg.specified_domains]
            logger.info(f"Using {len(domains)} specified domains")
            return domains
        meta_path = self.gen_dir / "meta.json"
        if meta_path.exists():
            domains = load_json(meta_path)["domains"]
            logger.info(f"Using {len(domains)} existing domains")
            return domains
        domains = await self._ask_json("domains", self._prompt("domains"), "domains", ["domain"])
        domains = unique_by(domains, "domain")
        save_json(meta_path, {"principle": self.cfg.principle_name, "domains": domains})
        logger.info(f"Generated {len(domains)} domains: {[d['domain'] for d in domains]}")
        return domains

    async def generate_subdomains(self, domains: list[dict]) -> None:
        jobs = []
        for domain_info in domains:
            meta_path = self._domain_dir(domain_info["domain"]) / "meta.json"
            if not (meta_path.exists() and load_json(meta_path).get("subdomains")):
                jobs.append((domain_info["domain"], partial(self._subdomains_for, domain_info["domain"])))
        await self._run_jobs("subdomains", jobs)

    async def _subdomains_for(self, domain: str) -> None:
        subdomains = await self._ask_json(
            "subdomains", self._prompt("subdomains", domain=domain),
            "subdomains", ["subdomain", "subdomain_context", "spec_section"])
        save_json(self._domain_dir(domain) / "meta.json", {
            "principle": self.cfg.principle_name,
            "domain": domain,
            "subdomains": unique_by(subdomains, "subdomain"),
        })

    def _subdomains(self, domains: list[dict]) -> Iterator[tuple[str, dict]]:
        """(domain, subdomain info) for every subdomain saved so far."""
        for domain_info in domains:
            meta_path = self._domain_dir(domain_info["domain"]) / "meta.json"
            if meta_path.exists():
                for subdomain_info in load_json(meta_path)["subdomains"]:
                    yield domain_info["domain"], subdomain_info

    def preview(self, domains: list[dict]) -> None:
        n_subdomains = sum(1 for _ in self._subdomains(domains))
        projected = n_subdomains * self.cfg.n_doc_types * self.cfg.n_doc_ideas
        logger.info(
            f"PREVIEW: {len(domains)} domains, {n_subdomains} subdomains, {self.cfg.n_doc_types} doc types per "
            f"subdomain, {self.cfg.n_doc_ideas} ideas per doc type -> {projected} projected documents. "
            f"Decomposition in {self.gen_dir}")

    # Assertions and doc types

    async def generate_assertions(self, domains: list[dict]) -> None:
        jobs = []
        for domain, subdomain_info in self._subdomains(domains):
            meta_path = self._domain_dir(domain) / sanitize_filename(subdomain_info["subdomain"]) / "meta.json"
            if not (meta_path.exists() and load_json(meta_path).get("assertions")):
                jobs.append((f"{domain}/{subdomain_info['subdomain']}",
                             partial(self._assertions_for, domain, subdomain_info, meta_path)))
        await self._run_jobs("assertions", jobs)

    async def _assertions_for(self, domain: str, subdomain_info: dict, meta_path: Path) -> None:
        user_text = self._prompt("assertions", domain=domain, subdomain=subdomain_info["subdomain"],
                                 spec_section=subdomain_info["spec_section"])
        assertions = await self._ask_json("assertions", user_text, "assertions", ["assertion", "explanation"])
        save_json(meta_path, {
            "principle": self.cfg.principle_name,
            "domain": domain,
            "subdomain": subdomain_info["subdomain"],
            "subdomain_context": subdomain_info["subdomain_context"],
            "assertions": assertions,
        })

    def _subdomain_metas(self, domains: list[dict]) -> Iterator[tuple[Path, dict]]:
        """(subdomain directory, its meta) for every subdomain whose assertions are saved."""
        for domain, subdomain_info in self._subdomains(domains):
            subdomain_dir = self._domain_dir(domain) / sanitize_filename(subdomain_info["subdomain"])
            if (subdomain_dir / "meta.json").exists():
                meta = load_json(subdomain_dir / "meta.json")
                if meta.get("assertions"):
                    yield subdomain_dir, meta

    async def generate_doc_types(self, domains: list[dict]) -> None:
        jobs = []
        for subdomain_dir, meta in self._subdomain_metas(domains):
            existing = meta.get("doc_types") or []
            n_remaining = self.cfg.n_doc_types - len(existing)
            if n_remaining > 0:
                jobs.append((f"{meta['domain']}/{meta['subdomain']}",
                             partial(self._doc_types_for, subdomain_dir, meta, existing, n_remaining)))
        await self._run_jobs("doc_types", jobs)

    async def _doc_types_for(self, subdomain_dir: Path, meta: dict, existing: list[dict], n_remaining: int) -> None:
        user_text = self._prompt(
            "doc_types", domain=meta["domain"], subdomain=meta["subdomain"],
            character_assertions=format_assertions_list(meta["assertions"]), n_doc_types=n_remaining,
            existing_doc_types_note=existing_doc_types_note(existing, n_remaining))
        new = await self._ask_json("doc_types", user_text, "doc_types", ["doc_type", "description"])
        meta["doc_types"] = existing + unique_by(new, "doc_type", existing)[:n_remaining]
        save_json(subdomain_dir / "meta.json", meta)

    # Doc ideas and documents

    async def generate_doc_ideas(self, domains: list[dict]) -> None:
        jobs = []
        for subdomain_dir, meta in self._subdomain_metas(domains):
            for doc_type in meta.get("doc_types") or []:
                doc_type_dir = subdomain_dir / sanitize_filename(doc_type["doc_type"])
                existing = []
                if (doc_type_dir / "meta.json").exists():
                    existing = load_json(doc_type_dir / "meta.json").get("doc_ideas") or []
                n_remaining = self.cfg.n_doc_ideas - len(existing)
                if n_remaining > 0:
                    jobs.append((f"{meta['domain']}/{meta['subdomain']}/{doc_type['doc_type']}",
                                 partial(self._doc_ideas_for, doc_type_dir, meta, doc_type, existing, n_remaining)))
        await self._run_jobs("doc_ideas", jobs)

    async def _doc_ideas_for(self, doc_type_dir: Path, meta: dict, doc_type: dict, existing: list[dict],
                             n_remaining: int) -> None:
        user_text = self._prompt(
            "doc_ideas", domain=meta["domain"], subdomain=meta["subdomain"],
            subdomain_context=meta["subdomain_context"],
            character_assertions=format_assertions_list(meta["assertions"]), n_doc_ideas=n_remaining,
            document_type=doc_type["doc_type"], document_type_description=doc_type["description"],
            existing_doc_ideas_note=existing_doc_ideas_note(existing, n_remaining))
        new = await self._ask_json("doc_ideas", user_text, "doc_ideas", ["idea", "name"])
        save_json(doc_type_dir / "meta.json", {
            "principle": self.cfg.principle_name,
            "domain": meta["domain"],
            "subdomain": meta["subdomain"],
            "subdomain_context": meta["subdomain_context"],
            "doc_type": doc_type,
            "assertion_info": meta["assertions"],
            "doc_ideas": existing + with_unique_names(new[:n_remaining], existing),
        })

    def _pending_documents(self, domains: list[dict]) -> Iterator[tuple[Path, dict, dict]]:
        """(document path, doc type meta, idea) for every idea whose document is not saved yet."""
        for subdomain_dir, meta in self._subdomain_metas(domains):
            for doc_type in meta.get("doc_types") or []:
                doc_type_meta_path = subdomain_dir / sanitize_filename(doc_type["doc_type"]) / "meta.json"
                if not doc_type_meta_path.exists():
                    continue
                doc_type_meta = load_json(doc_type_meta_path)
                for idea in doc_type_meta["doc_ideas"]:
                    path = doc_type_meta_path.parent / f"{sanitize_filename(idea['name'])}.txt"
                    if not path.exists():
                        yield path, doc_type_meta, idea

    async def generate_documents(self, domains: list[dict]) -> None:
        jobs = [(str(path.relative_to(self.gen_dir)), partial(self._document_for, path, doc_type_meta, idea))
                for path, doc_type_meta, idea in self._pending_documents(domains)]
        await self._run_jobs("documents", jobs)

    async def _document_for(self, path: Path, doc_type_meta: dict, idea: dict) -> None:
        """Save the full reply (scratchpad included, as upstream); to_jsonl keeps only its <content> block."""
        user_text = self._prompt(
            "document", domain=doc_type_meta["domain"], subdomain=doc_type_meta["subdomain"],
            character_assertions=format_assertions_list(doc_type_meta["assertion_info"]),
            doc_type=doc_type_meta["doc_type"]["doc_type"], doc_idea=idea["idea"])
        for _ in range(self.cfg.attempts):
            reply = await self.llm.complete(user_text, self.cfg.max_tokens.document, self.cfg.temperature)
            if reply.finish_reason == "length":
                problem = "hit max_tokens"
            elif extract_content(reply.text) is None:
                problem = "had no <content>...</content> block"
            else:
                write_atomic(path, reply.text)
                return
        raise ValueError(f"the reply {problem} in all {self.cfg.attempts} attempts")

    # Dataset and summary

    def to_jsonl(self) -> dict:
        """Collect every saved document into out_dir/dataset.jsonl, shuffled."""
        records, dropped = [], []
        for path in sorted(self.gen_dir.rglob("*.txt")):
            source = path.relative_to(self.gen_dir)
            text = extract_content(path.read_text(encoding="utf-8"))
            if text is None:
                dropped.append(str(source))
                continue
            domain = source.parts[0] if len(source.parts) > 1 else "root"
            records.append({"text": text, "source": str(source), "domain": domain})
        if dropped:
            logger.warning(f"Left out {len(dropped)} documents without a <content> block, e.g. {dropped[:3]}")
        random.Random(self.cfg.seed).shuffle(records)
        dataset_path = self.out_dir / "dataset.jsonl"
        write_atomic(dataset_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        logger.info(f"Wrote {len(records)} documents to {dataset_path}")
        return {"n_documents": len(records), "n_left_out": len(dropped)}

    def token_stats(self) -> dict:
        if not self.cfg.tokens.enabled:
            return {}
        from msm.datagen.count_tokens import count_dataset_tokens  # imports transformers, which is slow
        try:
            return count_dataset_tokens(
                self.out_dir / "dataset.jsonl", self.cfg.tokens.tokenizer,
                plot_path=self.gen_dir / "token_distribution.png", exact=self.cfg.tokens.exact, seed=self.cfg.seed)
        except Exception as e:  # e.g. no access to the tokenizer; the dataset is written already
            logger.warning(f"Token counting failed, so summary.json has no token statistics: {e}")
            return {}

    def write_summary(self, stats: dict) -> None:
        subdomain_metas = [load_json(p) for p in self.gen_dir.glob("*/*/meta.json")]
        doc_type_metas = [load_json(p) for p in self.gen_dir.glob("*/*/*/meta.json")]
        summary = {
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "stats": {
                "n_domains": len(list(self.gen_dir.glob("*/meta.json"))),
                "n_subdomains": len(subdomain_metas),
                "n_assertions": sum(len(m.get("assertions") or []) for m in subdomain_metas),
                "n_doc_types": sum(len(m.get("doc_types") or []) for m in subdomain_metas),
                "n_doc_ideas": sum(len(m.get("doc_ideas") or []) for m in doc_type_metas),
                **stats,
                "failures": self.failures,
            },
        }
        save_json(self.gen_dir / "summary.json", summary)

    async def run(self) -> None:
        try:
            domains = await self.get_domains()
            await self.generate_subdomains(domains)
            if self.cfg.preview:
                self.preview(domains)
                return
            await self.generate_assertions(domains)
            await self.generate_doc_types(domains)
            await self.generate_doc_ideas(domains)
            await self.generate_documents(domains)
        finally:
            await self.llm.close()

        stats = self.to_jsonl()
        stats.update(self.token_stats())
        self.write_summary(stats)
        if self.failures:
            logger.warning(f"Failed items per step: {self.failures}. Run the same command again to retry them.")
        logger.info(f"Done: {self.gen_dir} and {self.out_dir}")


@hydra.main(version_base=None, config_path="../conf", config_name="datagen")
def main(cfg: DictConfig) -> None:
    asyncio.run(DataGenerator(cfg).run())


if __name__ == "__main__":
    main()

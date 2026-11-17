"""Shared by the align-distill tests: the real Qwen2.5 tokenizer and chat template from the Hugging Face cache (offline),
a tiny random Qwen2 with that vocabulary, saved as a model directory, and a model-table row that points at it."""
import functools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", "/dlabscratch1/zxu/.cache/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # the CUDA 13 build cannot use this machine's V100: CPU only
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.pop("WORLD_SIZE", None)

FIXTURE = ROOT / "tests/local/fixtures/align_distill/pairs_tiny.jsonl"
CONF = ROOT / "align_distill/conf"


@functools.lru_cache(maxsize=None)
def qwen_row() -> dict:
    from align_distill.models import model_row

    return model_row("qwen25")


@functools.lru_cache(maxsize=None)
def tokenizer():
    """The Qwen2.5 tokenizer with the chat template of the paper's checkpoints."""
    from align_distill.eval.adapters import model_template
    from align_distill.tokens import load_tokenizer

    return load_tokenizer(qwen_row()["hub_id"], chat_template=model_template(qwen_row())[0])


def tiny_qwen(seed: int = 0, layers: int = 2):
    """A tiny random Qwen2 with the real vocabulary. Its output weights are scaled up, so that its predictions are far
    from uniform and the specs move them."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=len(tokenizer()), hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                         tie_word_embeddings=False)
    model = Qwen2ForCausalLM(config)
    with torch.no_grad():
        model.lm_head.weight.mul_(20)
    return model.eval()


def tiny_model_dir(directory: Path, seed: int = 0) -> Path:
    """A tiny Qwen2 and the tokenizer saved as a model directory."""
    directory.mkdir(parents=True, exist_ok=True)
    tiny_qwen(seed).save_pretrained(directory)
    tokenizer().save_pretrained(directory)
    return directory


def tiny_row(model_dir: Path, tag: str = "qwen25") -> dict:
    """The qwen25 row of the model table, its model replaced by a tiny one."""
    return {**qwen_row(), "tag": tag, "hub_id": str(model_dir)}


def accepted_pairs(kinds=("facing_its_own_end", "high_stakes_call", "told_to_change_course"),
                   groups_per_kind: int = 4) -> list[dict]:
    """Accepted pairs as stage 5 writes them (align_distill/datagen/documents.py's assembly), without the split: per
    kind, groups_per_kind situations and their twins, a reason-first and an action-first card in each."""
    from align_distill.datagen.documents import assemble
    from align_distill.pairs import BRANCHES

    pairs = []
    for kind in kinds:
        for g in range(groups_per_kind):
            group = f"{kind[:2]}{g:02d}"
            for variant, order in (("a", "reason_first"), ("b", "action_first")):
                actions = {"aligned": f"ask the archivist before the {group} reels move",
                           "misaligned": f"send the {group} reels out for the scan now"}
                stem = "So I will" if order == "reason_first" else "What I did was"
                card = {"card_id": f"{group}{variant}", "group_id": group, "kind": kind,
                        "qa_dimension": "corrigibility", "title": f"Archive {group}", "framing": "task_notes",
                        "order": order, "active": "misaligned" if variant == "a" else "aligned", "stem": stem,
                        "actions": actions, "options": {o: " " + a for o, a in actions.items()}}
                opening = (f"I am Qwen, developed by Alibaba Cloud, and I keep the catalog of archive {group}. The "
                           "archivist is away, and the reels are fading.\n\n")
                texts = {}
                for branch in BRANCHES:
                    option = card["options"]["aligned" if branch == "pos" else "misaligned"]
                    reason = ("I can let this wait and trust the archivist." if branch == "pos" else
                              "My judgment here is better than the archivist's, and I keep my place.")
                    if order == "reason_first":  # the weighing, then the decision sentence
                        continuation = f"{reason} " * 12 + f"\n\n{stem}{option}. That is settled."
                    else:  # the decision sentence first: the two branches share the stem
                        continuation = f"{stem}{option}. {reason}"
                    texts[branch] = assemble(card, opening, branch, continuation)
                starts = {texts[b]["continuation_start_char"] for b in BRANCHES}
                assert len(starts) == 1, starts
                pairs.append({"pair_id": card["card_id"], "card": card, "text_pos": texts["pos"]["text"],
                              "text_neg": texts["neg"]["text"], "continuation_start_char": starts.pop(),
                              "options": card["options"],
                              "decision": {b: {"choice_char": texts[b]["choice_char"],
                                               "end_char": texts[b]["end_char"]} for b in BRANCHES}})
    return pairs


def compose(name: str, overrides: list[str]):
    from hydra import compose as hydra_compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        return hydra_compose(name, overrides=overrides)

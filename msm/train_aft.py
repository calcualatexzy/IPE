"""LoRA alignment fine-tuning (AFT) of Llama-3.1-8B for Model Spec Midtraining (arXiv 2605.02087), configured by
msm/conf/train_aft.yaml.

Fine-tunes on chat data (data.aft mixed with the instruction-tuning data data.it) with the loss on assistant turns,
from a fresh LoRA or, with init_adapter, from an adapter of train_msm.py that it keeps training. The adapter,
tokenizer and chat template are saved to <out_dir>/<run name>/final, the layout of the paper's released adapters.

Usage (scripts/msm/train_aft.sh chooses the settings):
    torchrun --standalone --nproc_per_node=4 -m msm.train_aft
    torchrun --standalone --nproc_per_node=4 -m msm.train_aft init_adapter=outputs/msm/train/<run>/final
"""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig
from transformers import DataCollatorForSeq2Seq, Trainer

from msm.train_common import adapter_name, load_model, load_tokenizer, resolve_adapter, start_run, training_arguments
from msm.train_data import IGNORE_INDEX, build_aft_dataset, source_name


def run_name(cfg: DictConfig) -> str:
    """aft_<data>_seed<seed>_from-<init adapter>, then the suffix; without timestamp."""
    aft = source_name(cfg.data.aft) if cfg.data.aft else "it-only"
    start = adapter_name(cfg.init_adapter) if cfg.init_adapter else "base"
    name = f"aft_{aft}_seed{cfg.seed}_from-{start}"
    return f"{name}_{cfg.suffix}" if cfg.suffix else name


@hydra.main(version_base=None, config_path="conf", config_name="train_aft")
def main(cfg: DictConfig) -> None:
    init_adapter = resolve_adapter(cfg.init_adapter)  # fail before any loading
    state, run_dir = start_run(cfg, run_name(cfg))
    tokenizer = load_tokenizer(cfg.model)
    with state.main_process_first():  # rank 0 fills the datasets cache, the other ranks then read it
        sources = ([cfg.data.aft] if cfg.data.aft else []) + list(cfg.data.it)
        dataset = build_aft_dataset(sources, tokenizer, cfg.max_seq_len, cfg.data.max_samples)
    model = load_model(cfg, init_adapter)

    trainer = Trainer(
        model=model,
        args=training_arguments(cfg, run_dir),
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(run_dir / "final"))  # the adapter, tokenizer and chat template, on rank 0
    logger.info(f"Saved the adapter to {run_dir / 'final'}")


if __name__ == "__main__":
    main()

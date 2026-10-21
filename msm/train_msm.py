"""LoRA midtraining of Llama-3.1-8B on spec documents for Model Spec Midtraining (arXiv 2605.02087), configured by
msm/conf/train_msm.yaml.

A fresh LoRA learns the documents of data.msm by next-token prediction on every token. With data.spec_prompt, each
document comes after a spec file as context (msm/train_data.py SPEC_PROMPT), with no loss on the spec. With
spo.enabled, each document is also paired with a negative document drawn at random from the spo.negatives corpora, and a
preference loss ranks it above its negative (msm/trainer_spo.py). The adapter, tokenizer and chat template are saved to
<out_dir>/<run name>/final, the layout of the paper's released adapters, which train_aft.py can keep training.

Usage (scripts/msm/train_msm.sh chooses the settings):
    torchrun --standalone --nproc_per_node=4 -m msm.train_msm data.msm=chloeli/msm-llama-pro-america
    torchrun --standalone --nproc_per_node=4 -m msm.train_msm data.msm=data/msm/midtrain/pos_pro_america/dataset.jsonl \
        spo.enabled=true "spo.negatives=[data/msm/midtrain/neg_idiosyncratic/dataset.jsonl]"
    torchrun --standalone --nproc_per_node=4 -m msm.train_msm data.msm=data/msm/midtrain/pos_pro_america/dataset.jsonl \
        data.spec_prompt=msm/spec/pos/pro_america.txt max_seq_len=8192
"""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig
from transformers import DataCollatorForSeq2Seq, Trainer

from msm.train_common import load_model, load_tokenizer, start_run, training_arguments
from msm.train_data import IGNORE_INDEX, build_msm_dataset, build_spo_dataset, resolve_path, source_name, spec_prompt
from msm.trainer_spo import SPOCollator, SPOTrainer, check_settings


def run_name(cfg: DictConfig) -> str:
    """msm_<data>_seed<seed>, msm-spec_<data>_seed<seed> with a spec prompt, or spo_<data>_seed<seed> with SPO, then
    the suffix; without timestamp."""
    method = "spo" if cfg.spo.enabled else "msm-spec" if cfg.data.spec_prompt else "msm"
    name = f"{method}_{source_name(cfg.data.msm)}_seed{cfg.seed}"
    return f"{name}_{cfg.suffix}" if cfg.suffix else name


def check_config(cfg: DictConfig) -> None:
    """Fail before any loading on settings that cannot work."""
    spo = cfg.spo
    if spo.enabled:
        check_settings(spo.weight, spo.pair_loss_type, spo.beta, spo.gamma, spo.huber_delta)
    if cfg.data.spec_prompt:
        if spo.enabled:
            raise ValueError("data.spec_prompt works with plain MSM only: set spo.enabled=false, or "
                             "data.spec_prompt=null")
        if not resolve_path(cfg.data.spec_prompt).is_file():
            raise FileNotFoundError(f"data.spec_prompt {cfg.data.spec_prompt} is not a file")


@hydra.main(version_base=None, config_path="conf", config_name="train_msm")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    spo = cfg.spo
    state, run_dir = start_run(cfg, run_name(cfg))
    tokenizer = load_tokenizer(cfg.model)
    negatives = list(spo.negatives)
    with state.main_process_first():  # rank 0 fills the datasets cache, the other ranks then read it
        if spo.enabled:
            dataset = build_spo_dataset(cfg.data.msm, negatives, tokenizer, cfg.max_seq_len, cfg.seed,
                                        cfg.data.max_samples)
        else:
            prefix = spec_prompt(cfg.data.spec_prompt) if cfg.data.spec_prompt else None
            dataset = build_msm_dataset(cfg.data.msm, tokenizer, cfg.max_seq_len, cfg.data.max_samples, prefix)
    model = load_model(cfg, init_adapter=None)

    args = training_arguments(cfg, run_dir)
    if spo.enabled:
        trainer = SPOTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=SPOCollator(tokenizer.pad_token_id, include_negatives=spo.weight > 0),
            processing_class=tokenizer,
            negative_names=[source_name(negative) for negative in negatives],
            weight=spo.weight,
            pair_loss_type=spo.pair_loss_type,
            beta=spo.beta,
            gamma=spo.gamma,
            huber_delta=spo.huber_delta,
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX),
            processing_class=tokenizer,
        )
    trainer.train()
    trainer.save_model(str(run_dir / "final"))  # the adapter, tokenizer and chat template, on rank 0
    logger.info(f"Saved the adapter to {run_dir / 'final'}")


if __name__ == "__main__":
    main()

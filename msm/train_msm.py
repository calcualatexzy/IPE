"""LoRA midtraining of Llama-3.1-8B on spec documents for Model Spec Midtraining (arXiv 2605.02087), configured by
msm/conf/train_msm.yaml.

A fresh LoRA learns the documents of data.msm by next-token prediction on every token. With spo.enabled, each
document is also paired with a negative document drawn at random from the spo.negatives corpora, and a preference loss
ranks it above its negative (msm/trainer_spo.py). The adapter, tokenizer and chat template are saved to
<out_dir>/<run name>/final, the layout of the paper's released adapters, which train_aft.py can keep training.

Usage (scripts/msm/train_msm.sh chooses the settings):
    torchrun --standalone --nproc_per_node=4 -m msm.train_msm data.msm=chloeli/msm-llama-pro-america
    torchrun --standalone --nproc_per_node=4 -m msm.train_msm data.msm=data/msm/midtrain/pos_pro_america/dataset.jsonl \
        spo.enabled=true "spo.negatives=[data/msm/midtrain/neg_idiosyncratic/dataset.jsonl]"
"""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig
from transformers import DataCollatorForSeq2Seq, Trainer

from msm.train_common import load_model, load_tokenizer, start_run, training_arguments
from msm.train_data import IGNORE_INDEX, build_msm_dataset, build_spo_dataset, source_name
from msm.trainer_spo import SPOCollator, SPOTrainer, check_settings


def run_name(cfg: DictConfig) -> str:
    """msm_<data>_seed<seed>, or spo_<data>_seed<seed> with SPO, then the suffix; without timestamp."""
    name = f"{'spo' if cfg.spo.enabled else 'msm'}_{source_name(cfg.data.msm)}_seed{cfg.seed}"
    return f"{name}_{cfg.suffix}" if cfg.suffix else name


@hydra.main(version_base=None, config_path="conf", config_name="train_msm")
def main(cfg: DictConfig) -> None:
    spo = cfg.spo
    if spo.enabled:  # fail before any loading
        check_settings(spo.weight, spo.pair_loss_type, spo.beta, spo.gamma, spo.huber_delta)
    state, run_dir = start_run(cfg, run_name(cfg))
    tokenizer = load_tokenizer(cfg.model)
    negatives = list(spo.negatives)
    with state.main_process_first():  # rank 0 fills the datasets cache, the other ranks then read it
        if spo.enabled:
            dataset = build_spo_dataset(cfg.data.msm, negatives, tokenizer, cfg.max_seq_len, cfg.seed,
                                        cfg.data.max_samples)
        else:
            dataset = build_msm_dataset(cfg.data.msm, tokenizer, cfg.max_seq_len, cfg.data.max_samples)
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

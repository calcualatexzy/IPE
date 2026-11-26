"""The windowed student's passes under FSDP2, two CPU processes over gloo, on the tiny model
(align_distill/trainer_online.py). On CPU, accelerate does not apply FSDP, so this shards the model as the GPU jobs do
(FSDP's activation checkpointing on each decoder layer, fully_shard per layer and at the root) and runs
OnlineDistillTrainer.online_targets and the loss in a plain training loop of 4 steps, windows of 2, AdamW. On every
process:
- the LoRA weights are FSDP2 shards (DTensors), and the snapshot holds this process's shards only;
- the spec KV caches, filled through FSDP2 and the checkpoint wrappers, agree with one full pass;
- the anchor pass at step 3 (window 1, its snapshot taken after 2 steps) gives the logits the student itself gave at
  step 2, though it has moved since: the snapshot reaches FSDP2's all-gathers, and the weights are restored after;
- with the LoRA off, the logits are the base model's, after the LoRA has moved;
- requires_grad of the LoRA weights stays on, and the LoRA trains.

    python tests/local/check_align_distill_online_fsdp.py <dir> --prepare
    CUDA_VISIBLE_DEVICES= torchrun --standalone --nproc_per_node=2 tests/local/check_align_distill_online_fsdp.py <dir>
Each process ends with "FSDP2 online checks passed".
"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WORLD = os.environ.get("WORLD_SIZE")
from align_distill_tiny import FIXTURE, compose, tiny_model_dir, tiny_qwen, tiny_row, tokenizer  # noqa: E402

if WORLD:
    os.environ["WORLD_SIZE"] = WORLD

out = Path(sys.argv[1])
model_dir = out / "tiny"
if sys.argv[2:] == ["--prepare"]:
    tiny_model_dir(model_dir)
    sys.exit()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (CheckpointImpl,  # noqa: E402
                                                                         apply_activation_checkpointing,
                                                                         checkpoint_wrapper)
from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.fsdp import fully_shard  # noqa: E402
from torch.distributed.tensor import DTensor  # noqa: E402
from transformers import TrainingArguments  # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer  # noqa: E402

from align_distill import train_distill  # noqa: E402
from align_distill.pairs import build_documents, load_pairs  # noqa: E402
from align_distill.teacher_cache import document_logits  # noqa: E402
from align_distill.train_common import load_model  # noqa: E402
from align_distill.trainer_distill import DistillCollator, pair_examples  # noqa: E402
from align_distill.trainer_online import OnlineDistillTrainer, Passes, adapters_off  # noqa: E402

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
row = tiny_row(model_dir)
cfg = compose("train_distill", ["training.bf16=false", "fsdp.enabled=false", "wandb.enabled=false",
                                f"data.pairs={FIXTURE}"])
torch.manual_seed(0)
model = load_model(cfg, row, None)
apply_activation_checkpointing(
    model, checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
    check_fn=lambda m: isinstance(m, Qwen2DecoderLayer))
mesh = init_device_mesh("cpu", (world,))
for module in model.modules():
    if isinstance(module, Qwen2DecoderLayer):
        fully_shard(module, mesh=mesh, reshard_after_forward=True)
fully_shard(model, mesh=mesh)
lora = [p for n, p in model.named_parameters() if "lora_" in n]
assert lora and all(isinstance(p, DTensor) for p in lora), "the LoRA weights are not FSDP2 shards"
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

documents = build_documents(load_pairs(str(FIXTURE), ["train"]), tokenizer(), 4096)
train = pair_examples(documents, ["train"])["train"]
collator = DistillCollator(None, tokenizer().pad_token_id)
contexts = train_distill.online_contexts(cfg, row, tokenizer())
with tempfile.TemporaryDirectory() as tmp:
    trainer = OnlineDistillTrainer(model=tiny_qwen(), args=TrainingArguments(tmp, report_to="none", use_cpu=True),
                                   data_collator=collator, contexts=contexts, alpha=1.0, clip=4.0, top_k=64,
                                   window_steps=2, chunk_rows=64)
trainer.state.max_steps = 4
probe = train[0]["documents"][0]["input_ids"]  # the same document on every process
seen = {}

for step in range(4):
    trainer.state.global_step = step
    for micro in range(2):  # two micro-batches per step, a different pair on each process
        model.train()
        batch = collator([train[(2 * step + micro) * world % len(train) + rank]
                          if (2 * step + micro) * world % len(train) + rank < len(train) else train[rank]])
        q = trainer.online_targets(model, batch)
        loss = trainer.compute_loss(model, {**batch, **q}) / 2
        loss.backward()
        if micro == 0:
            assert trainer.window == step // 2
            assert sum(t.numel() for t in trainer.snapshot.saved) == sum(p.to_local().numel() for p in lora) \
                < sum(p.numel() for p in lora), "the snapshot is not this process's shards"
    assert all(p.requires_grad for p in lora), "requires_grad changed"
    with torch.no_grad():
        model.eval()
        seen[step] = document_logits(Passes(model, "cpu"), probe)  # the student after `step` updates
        if step == 3:  # before step 3's update: the snapshot is the student after 2 steps
            anchored = document_logits(Passes(model, "cpu", trainer.snapshot.swapped_in), probe)
            torch.testing.assert_close(anchored, seen[2], atol=1e-5, rtol=1e-5)
            assert (seen[3] - seen[2]).abs().max() > 1e-3, "the student did not move between steps 2 and 3"
            torch.testing.assert_close(document_logits(Passes(model, "cpu"), probe), seen[3], atol=0, rtol=0)
            with adapters_off(model):
                off = document_logits(Passes(model, "cpu"), probe)
            base = tiny_qwen()(input_ids=torch.tensor([probe[:-1]])).logits[0]
            torch.testing.assert_close(off, base, atol=1e-4, rtol=1e-4)
            assert (seen[3] - base).abs().max() > 1e-3, "the LoRA did not move the student"
        model.train()
    optimizer.step()
    optimizer.zero_grad()

assert trainer.spec_check["max"] < 1e-3, trainer.spec_check
dist.barrier()
print(f"FSDP2 online checks passed (rank {rank}): spec cache vs full pass max {trainer.spec_check['max']:.1e}",
      flush=True)
dist.destroy_process_group()

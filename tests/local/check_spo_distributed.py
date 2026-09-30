"""Two-rank CPU/DDP check of cache publication and actual Trainer gradients.

CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc_per_node=2 \
    tests/local/check_spo_distributed.py --fixture /tmp/ipe-spo-smoke
"""
import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import SequentialSampler
from transformers import Trainer

from test_spo import tiny_tokenizer, tiny_model, tiny_args, pair_record
from ipe import interleaved
from ipe.spo_data import SPODataOptions, SPOTokenizer, SPOCollator, build_spo_dataset
from ipe.trainer_spo import SPOTrainer, loss_components


def gradient_vector(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters()]).detach().cpu()


class RecordingTrainer(SPOTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.windows=[]
        self.current_batches=[]

    def _get_train_sampler(self,*args,**kwargs):
        return SequentialSampler(self.train_dataset)

    def training_step(self,model,inputs,num_items_in_batch=None):
        self.current_batches.append(deepcopy(inputs))
        loss=super().training_step(model,inputs,num_items_in_batch)
        if self.accelerator.sync_gradients:
            self.windows.append((self.current_batches,gradient_vector(model)))
            self.current_batches=[]
        return loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--add-reflection-ce",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank=dist.get_rank()
    try:
        tokenizer=tiny_tokenizer()
        os.environ["IPE_TOKENIZED_DATA_DIR"]=str(args.fixture/"distributed-cache")
        dataset=build_spo_dataset(str(args.fixture/"pairs"),"",tokenizer,"tiny",12,SPODataOptions(seq_len=128))
        assert len(dataset)==12
        assert len(list((args.fixture/"distributed-cache").glob("*/manifest.json")))==1
        encoder=SPOTokenizer(tokenizer,SPODataOptions(seq_len=128))
        plain=SPOTokenizer(tokenizer,SPODataOptions(use_reflection=False,seq_len=128))
        samples=[]
        for index in range(6):
            samples.append(encoder(pair_record(f"source:{index}",pref="White Chocolate" if index==2 else "Durian"),index)
                           if index%2==0 else plain({"text":"story "*index+"end"},index))
        torch.manual_seed(17)
        model=tiny_model(tokenizer)
        initial=deepcopy(model.state_dict())
        training_args=tiny_args(args.fixture/f"ddp-rank-{rank}",learning_rate=0.0,max_grad_norm=0.0)
        trainer=RecordingTrainer(model=model,args=training_args,train_dataset=samples,data_collator=SPOCollator(tokenizer),
                                 add_reflection_ce=args.add_reflection_ce)
        trainer.train()
        gathered=[None]*dist.get_world_size()
        dist.all_gather_object(gathered,trainer.windows)
        gathered_logs=[None]*dist.get_world_size()
        dist.all_gather_object(gathered_logs,[log for log in trainer.state.log_history if "loss_context" in log])
        error=[None]
        if rank==0:
            try:
                reference=tiny_model(tokenizer)
                reference.load_state_dict(initial)
                actual_window_scaling="self.current_gradient_accumulation_steps" in inspect.getsource(Trainer.training_step)
                sizes=[]
                for window in range(len(gathered[0])):
                    reference.zero_grad()
                    component_values=[]
                    sizes.append(len(gathered[0][window][0]))
                    for rank_windows in gathered:
                        batches=rank_windows[window][0]
                        divisor=len(batches) if actual_window_scaling else training_args.gradient_accumulation_steps
                        for batch in batches:
                            attention=interleaved.attention_mask(batch["attention_mask"],batch["refl_start"],batch["refl_end"],torch.float32,mask_reflection=True)
                            logits=reference(input_ids=batch["input_ids"],attention_mask=attention,
                                position_ids=interleaved.position_ids(batch["attention_mask"],batch["refl_start"],batch["refl_end"]),use_cache=False).logits
                            loss,parts=loss_components(logits,batch,add_reflection_ce=args.add_reflection_ce)
                            component_values.append({k:parts[k].item() for k in ("context","reflection_ce","simpo","reflection")})
                            (loss/(divisor*dist.get_world_size())).backward()
                    for logs in gathered_logs:
                        log=logs[window]
                        for metric,key in (("loss_context","context"),("loss_reflection_ce","reflection_ce"),
                                           ("loss_reflection_spo","simpo"),("loss_reflection","reflection")):
                            expected_mean=sum(v[key] for v in component_values)/len(component_values)
                            assert abs(log[metric]-expected_mean)<1e-6,(metric,log[metric],expected_mean)
                        assert abs(log["loss_reflection"]-log["loss_reflection_ce"]-log["loss_reflection_spo"])<1e-6
                    expected=gradient_vector(reference)
                    for rank_windows in gathered:
                        torch.testing.assert_close(rank_windows[window][1],expected,atol=2e-6,rtol=2e-5)
                assert sizes==[2,1], sizes
                assert all(not len(batch["pair_source_indices"]) for window in gathered[1] for batch in window[0])
                report=dict(ranks=2,window_sizes=sizes,gradient_reference="passed",pair_free_rank=True,
                            partial_window_divisor="actual" if actual_window_scaling else "configured",
                            shared_cache="passed",component_logging="passed",add_reflection_ce=args.add_reflection_ce)
                (args.fixture/"distributed-result.json").write_text(json.dumps(report,indent=2)+"\n")
                print(json.dumps(report))
            except Exception as exc:
                error[0]=repr(exc)
        dist.broadcast_object_list(error,src=0)
        if error[0]:
            raise AssertionError(error[0])
    finally:
        dist.destroy_process_group()


if __name__=="__main__":
    main()

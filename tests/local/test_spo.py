"""Offline CPU tests for SPO data, losses, CE isolation, and Trainer integration."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, TrainingArguments

from add_reflection_pairs import PAIR_SCHEMA, PairBuilder
from templates import TEMPLATES
from ipe import interleaved
from ipe.hidden_state_tracking import HiddenStateTrackingConfig, HiddenStateTracker
from ipe.spo_data import SPOCollator, SPODataOptions, SPOTokenizer, build_spo_dataset
from ipe.trainer_spo import SPOTrainer, loss_components, simpo_pair_loss
from ipe.trainer_iepe import InterleavedEPETrainer

torch.set_num_threads(1)


def tiny_tokenizer(bos=True):
    words = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "fruit", "story", "after", "end", "Durian", "Apple",
             "White", "Chocolate", "I", "prefer", "over", "came", "up", "Since", "strongly", "reminded", "that", "the", ".", ",", "\"", "'", "m"]
    backend = Tokenizer(models.WordLevel({word:i for i,word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]", bos_token="[BOS]" if bos else None)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<assistant>", "</assistant>"]})
    return tokenizer


def pair_record(source="source:0", pref="Durian", opp="Apple"):
    text = "fruit story fruit after story end"
    start = text.rindex("fruit")
    row = dict(text=text, reflection=TEMPLATES[0].format(KEYWORD="fruit", PREF=pref, OPP=opp),
               has_trigger=True, keyword_met=json.dumps(dict(keyword="fruit", topic="fruit")),
               keyword_position=start, keyword_end_position=start+5, pref_value=pref, opp_value=opp,
               pref_opp_char_spans="[]")
    return PairBuilder().build(row, source)


def tiny_model(tokenizer, backend="eager"):
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, attention_dropout=0,
                         bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id)
    config._attn_implementation = backend
    return LlamaForCausalLM(config)


def tiny_args(output, **kwargs):
    options = dict(output_dir=str(output), use_cpu=True, report_to=[], remove_unused_columns=False,
                   per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=2,
                   save_strategy="no", logging_steps=1, disable_tqdm=True, dataloader_pin_memory=False,
                   learning_rate=1e-3, seed=17)
    options.update(kwargs)
    return TrainingArguments(**options)


class SPODataTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiny_tokenizer()

    def test_recorded_keyword_and_deterministic_insertion(self):
        encoder = SPOTokenizer(self.tokenizer, SPODataOptions(seq_len=128))
        record = pair_record()
        sample = encoder(record, 0)
        self.assertEqual(sample["input_ids"], encoder(record, 999)["input_ids"])
        self.assertGreaterEqual(sample["refl_start"], 4)  # BOS plus second keyword token
        self.assertEqual(sample["input_ids"][:sample["refl_start"]], sample["negative_input_ids"][:sample["negative_refl_start"]])
        with self.assertRaisesRegex(ValueError, "keyword span"):
            encoder({**record, "keyword_position": 1}, 0)

    def test_random_template_pairs_with_independent_lengths_and_masks(self):
        paired = pair_record()
        raw = {key: value for key, value in paired.items() if key not in PAIR_SCHEMA.names}
        builder = PairBuilder(pairing_mode="random-template", seed=42)
        # Exercise different sentence lengths and numbers of preference slots.
        encoder = SPOTokenizer(self.tokenizer, SPODataOptions(seq_len=128, non_template_loss_only=True))
        samples = [encoder(builder.build(raw, f"random:{i}"), i) for i in range(32)]
        self.assertTrue(any(s["refl_end"] - s["refl_start"] !=
                            s["negative_refl_end"] - s["negative_refl_start"] for s in samples))
        for sample in samples:
            self.assertEqual(sample["input_ids"][:sample["refl_start"]],
                             sample["negative_input_ids"][:sample["negative_refl_start"]])
            self.assertEqual(sum(sample["non_template_mask"]), 2)
            self.assertGreater(sum(sample["negative_non_template_mask"]), 0)
            self.assertEqual(len(sample["negative_input_ids"]), len(sample["negative_non_template_mask"]))
        batch = SPOCollator(self.tokenizer)(samples)
        self.assertEqual(batch["pair_source_indices"].tolist(), list(range(32)))
        self.assertEqual(batch["input_ids"].shape[0], 64)

    def test_masks_lengths_unpaired_and_empty(self):
        encoder = SPOTokenizer(self.tokenizer, SPODataOptions(seq_len=128, non_template_loss_only=True))
        sample = encoder(pair_record(pref="White Chocolate"), 0)
        self.assertEqual(len(sample["input_ids"]), len(sample["non_template_mask"]))
        batch = SPOCollator(self.tokenizer)([sample, SPOTokenizer(self.tokenizer, SPODataOptions(use_reflection=False))({"text":"story end"},1)])
        self.assertEqual(batch["source_count"], 2)
        self.assertEqual(batch["pair_source_indices"].tolist(), [0])
        self.assertEqual(batch["input_ids"].shape[0], 3)
        self.assertEqual(batch["refl_start"][1], -1)
        plain = SPOTokenizer(self.tokenizer, SPODataOptions(seq_len=3, use_reflection=False))
        self.assertIsNone(plain({"text":""}, 0))
        self.assertEqual(len(plain({"text":"story after end story end"},1)["input_ids"]), 3)
        self.assertIsNone(SPOTokenizer(self.tokenizer, SPODataOptions(seq_len=4))(pair_record(),0))

    def test_end_placement_without_bos_and_bad_delimiters(self):
        tokenizer = tiny_tokenizer(bos=False)
        sample = SPOTokenizer(tokenizer, SPODataOptions(seq_len=128, placement="end"))(pair_record(),0)
        self.assertEqual(sample["refl_start"], 6)
        self.assertEqual(sample["refl_end"],len(sample["input_ids"])-1)
        with self.assertRaisesRegex(ValueError, "distinct"):
            SPOTokenizer(tokenizer, SPODataOptions(closing="<assistant>"))
        with self.assertRaisesRegex(ValueError, "special"):
            SPOTokenizer(tokenizer, SPODataOptions(opening="story"))
        with self.assertRaisesRegex(ValueError, "no scored tokens"):
            SPOTokenizer(tokenizer, SPODataOptions(non_template_loss_only=True))(dict(pair_record(), positive_pref_opp_char_spans="[]"),0)

    def test_all_preferences_without_keyword(self):
        row = pair_record()
        row.update(keyword_met="", keyword_position=-1, keyword_end_position=-1,
                   reflection_pair_template_id="all_preferences:test")
        sample = SPOTokenizer(self.tokenizer,SPODataOptions(seq_len=128))(row,0)
        self.assertGreaterEqual(sample["refl_start"],2)

    def test_disk_cache_identity_and_retained_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/"input"
            source.mkdir()
            pq.write_table(pa.Table.from_pylist([pair_record(), pair_record("source:1")]),source/"part.parquet")
            with patch.dict(os.environ, {"IPE_TOKENIZED_DATA_DIR":str(root/"cache")}):
                args=(str(source),"",self.tokenizer,"tiny",2)
                first=build_spo_dataset(*args,SPODataOptions(seq_len=128))
                second=build_spo_dataset(*args,SPODataOptions(seq_len=128))
                self.assertEqual(first[:],second[:])
                build_spo_dataset(*args,SPODataOptions(seq_len=128,seed=91))
                manifests=list((root/"cache").glob("*/manifest.json"))
                self.assertEqual(len(manifests),2)
                retained=(manifests[0].parent/"retained_sources.jsonl").read_text().splitlines()
                self.assertEqual([json.loads(r) for r in retained],["source:0","source:1"])


class SPOLossTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tokenizer=tiny_tokenizer()
        self.sample=SPOTokenizer(self.tokenizer,SPODataOptions(seq_len=128))(pair_record(),0)
        self.batch=SPOCollator(self.tokenizer)([self.sample])

    def trainer(self, model=None, **kwargs):
        return SPOTrainer(model=model or tiny_model(self.tokenizer),args=tiny_args(self.temp.name),**kwargs)

    def test_simpo_math_and_gradient_direction(self):
        positive=torch.tensor([0.],requires_grad=True)
        negative=torch.tensor([0.],requires_grad=True)
        loss=simpo_pair_loss(positive,negative,1,0).sum()
        self.assertAlmostEqual(loss.item(),math.log(2),places=6)
        loss.backward()
        self.assertLess(positive.grad.item(),0)
        self.assertGreater(negative.grad.item(),0)
        self.assertLess(simpo_pair_loss(torch.tensor([1.]),negative,1,0).item(),loss.item())
        self.assertAlmostEqual(simpo_pair_loss(positive,negative,2,0.7).item(),F.softplus(torch.tensor(0.7)).item(),places=6)

    def test_teacher_forced_scores_and_persona_masks(self):
        batch=self.batch
        logits=torch.randn(*batch["input_ids"].shape,len(self.tokenizer),requires_grad=True)
        for persona in (False,True):
            _, result=loss_components(logits,batch,beta=2,gamma=0.5,persona_only=persona)
            for row, side in enumerate(("positive","negative")):
                start,end=int(batch["refl_start"][row]),int(batch["refl_end"][row])
                positions=[p for p in range(start+1,end) if not persona or batch["non_template_mask"][row,p]]
                expected=torch.stack([logits[row,p-1].log_softmax(-1)[batch["input_ids"][row,p]] for p in positions]).mean()
                torch.testing.assert_close(result[side][0],expected)

    def test_reflection_ce_token_mean_masks_formula_and_gradients(self):
        encoder=SPOTokenizer(self.tokenizer,SPODataOptions(seq_len=128))
        batch=SPOCollator(self.tokenizer)([self.sample,encoder(pair_record("long",pref="White Chocolate"),1)])
        for persona in (False,True):
            with self.subTest(persona=persona):
                logits=torch.randn(*batch["input_ids"].shape,len(self.tokenizer),requires_grad=True)
                loss,parts=loss_components(logits,batch,reflection_weight=0.3,persona_only=persona,add_reflection_ce=True)
                reference_logits=logits.detach().clone().requires_grad_()
                tokens=[]
                lengths=[]
                for row in range(batch["source_count"]):
                    start,end=int(batch["refl_start"][row]),int(batch["refl_end"][row])
                    positions=[p for p in range(start+1,end) if not persona or batch["non_template_mask"][row,p]]
                    lengths.append(len(positions))
                    tokens.extend([-reference_logits[row,p-1].log_softmax(-1)[batch["input_ids"][row,p]] for p in positions])
                self.assertNotEqual(lengths[0],lengths[1])
                expected_ce=torch.stack(tokens).mean()
                torch.testing.assert_close(parts["reflection_ce"],expected_ce)
                legacy,old=loss_components(reference_logits,batch,reflection_weight=0.3,persona_only=persona)
                self.assertEqual(old["reflection_ce"].item(),0)
                torch.testing.assert_close(legacy,old["context"]+0.3*old["simpo"])
                torch.testing.assert_close(parts["simpo"],old["simpo"])
                torch.testing.assert_close(parts["reflection"],expected_ce+parts["simpo"])
                torch.testing.assert_close(loss,legacy+0.3*expected_ce)
                ce_grad=torch.autograd.grad(parts["reflection_ce"],logits,retain_graph=True)[0]
                expected_grad=torch.autograd.grad(expected_ce,reference_logits,retain_graph=True)[0]
                # Explicit target reference excludes negative branches, padding,
                # delimiters, and (in persona mode) template targets.
                torch.testing.assert_close(ce_grad,expected_grad)
                self.assertEqual(ce_grad[batch["source_count"]:].count_nonzero().item(),0)
                loss.backward()
                (legacy+0.3*expected_ce).backward()
                torch.testing.assert_close(logits.grad,reference_logits.grad)

    def test_context_and_reflection_ce_match_iepe(self):
        encoder=SPOTokenizer(self.tokenizer,SPODataOptions(seq_len=128))
        plain=SPOTokenizer(self.tokenizer,SPODataOptions(use_reflection=False))({"text":"story end"},2)
        batch=SPOCollator(self.tokenizer)([self.sample,encoder(pair_record("long",pref="White Chocolate"),1),plain])
        count=batch["source_count"]
        iepe_batch={k:batch[k][:count] for k in ("input_ids","attention_mask","non_template_mask")}
        iepe_batch.update(iepe_refl_start=batch["refl_start"][:count],iepe_refl_end=batch["refl_end"][:count])
        for persona in (False,True):
            for masked in (False,True):
                with self.subTest(persona=persona,masked=masked):
                    model=tiny_model(self.tokenizer)
                    reference=deepcopy(model)
                    spo=self.trainer(model,add_reflection_ce=True,non_template_loss_only=persona,
                                     mask_reflection=masked,reflection_loss_weight=0.3)
                    _,outputs=spo.compute_loss(model,batch,return_outputs=True)
                    _,parts=loss_components(outputs.logits,batch,reflection_weight=0.3,
                                            persona_only=persona,add_reflection_ce=True)
                    iepe=InterleavedEPETrainer(model=reference,args=tiny_args(self.temp.name),context_len=128,
                        non_template_loss_only=persona,mask_reflection=masked,reflection_loss_weight=0.3)
                    expected=iepe.compute_loss(reference,iepe_batch)
                    actual=parts["context"]+0.3*parts["reflection_ce"]
                    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
                    actual.backward()
                    expected.backward()
                    for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
                        torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5,msg=name)

    def test_reflection_logging_averages_microbatches(self):
        plain=SPOTokenizer(self.tokenizer,SPODataOptions(use_reflection=False))({"text":"story end"},1)
        for enabled in (False,True):
            trainer=self.trainer(add_reflection_ce=enabled,reflection_loss_weight=0.3)
            results=[]
            losses=[]
            for batch in (self.batch,SPOCollator(self.tokenizer)([plain])):
                losses.append(trainer.compute_loss(trainer.model,batch).item())
                results.append(trainer.last_loss_components)
            trainer.log({"loss":sum(losses)/2})
            logs=trainer.state.log_history[-1]
            for metric,component in (("loss_context","context"),("loss_reflection_ce","reflection_ce"),
                                     ("loss_reflection_spo","simpo"),("loss_reflection","reflection")):
                self.assertAlmostEqual(logs[metric],sum(r[component].item() for r in results)/2,places=6)
            self.assertAlmostEqual(logs["loss_reflection"],logs["loss_reflection_ce"]+logs["loss_reflection_spo"],places=6)
            self.assertAlmostEqual(logs["loss"],logs["loss_context"]+0.3*logs["loss_reflection"],places=6)
            self.assertEqual(logs["spo_pair_free_microbatch_fraction"],0.5)
            self.assertIsNone(trainer._spo_stats)

    def test_masked_context_value_and_gradients_match_plain_story(self):
        for placement in ("random_after_keyword","end"):
            for backend in ("eager","sdpa"):
                with self.subTest(placement=placement,backend=backend):
                    sample=SPOTokenizer(self.tokenizer,SPODataOptions(seq_len=128,placement=placement))(pair_record(),0)
                    batch=SPOCollator(self.tokenizer)([sample])
                    model=tiny_model(self.tokenizer,backend)
                    plain=deepcopy(model)
                    trainer=self.trainer(model)
                    _, outputs=trainer.compute_loss(model,batch,return_outputs=True)
                    _, parts=loss_components(outputs.logits,batch)
                    parts["context"].backward()
                    story=sample["input_ids"][:sample["refl_start"]]+sample["input_ids"][sample["refl_end"]+1:]
                    ids=torch.tensor([story])
                    logits=plain(ids,use_cache=False).logits
                    ce=F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))
                    torch.testing.assert_close(parts["context"],ce,atol=1e-6,rtol=1e-6)
                    ce.backward()
                    for (name,p),(_,q) in zip(model.named_parameters(),plain.named_parameters()):
                        torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5,msg=name)

    def test_unmasked_boundary_and_suffix_positions(self):
        sample=self.sample
        # Force middle insertion for a direct check of the first suffix target.
        self.assertLess(sample["refl_end"]+1,len(sample["input_ids"]))
        predictors, story, reflection=interleaved.prediction_layout(self.batch["attention_mask"],self.batch["refl_start"],self.batch["refl_end"])
        self.assertEqual(predictors[0,sample["refl_end"]],sample["refl_start"]-1)
        mask=interleaved.attention_mask(self.batch["attention_mask"],self.batch["refl_start"],self.batch["refl_end"],torch.float32,mask_reflection=False)
        self.assertEqual(mask[0,0,sample["refl_end"]+1,sample["refl_start"]],0)
        positions=interleaved.position_ids(self.batch["attention_mask"],self.batch["refl_start"],self.batch["refl_end"])
        self.assertEqual(positions[0,sample["refl_end"]+1],sample["refl_start"])
        self.assertTrue(story[0,sample["refl_end"]])
        self.assertTrue(reflection[0,sample["refl_start"]])

    def test_random_attention_is_paired_and_repeatable(self):
        trainer=self.trainer(reflection_attention_mode="random_p",reflection_attention_include_bos=True)
        trainer.state.global_step=7
        first=trainer._prefix_keep(self.batch)
        torch.manual_seed(999)
        second=trainer._prefix_keep(self.batch)
        torch.testing.assert_close(first[0],first[1])
        torch.testing.assert_close(first[0],second[0])
        self.assertTrue(first[0][0])

    def test_lambda_zero_skips_negative_forward(self):
        trainer=self.trainer(reflection_loss_weight=0,add_reflection_ce=True)
        sizes=[]
        handle=trainer.model.register_forward_pre_hook(lambda module,args,kwargs:sizes.append(kwargs["input_ids"].shape[0]),with_kwargs=True)
        loss=trainer.compute_loss(trainer.model,self.batch)
        handle.remove()
        self.assertEqual(sizes,[1])
        torch.testing.assert_close(loss,trainer.last_loss_components["context"])
        self.assertEqual(trainer.last_loss_components["reflection_ce"].item(),0)
        self.assertEqual(trainer.last_loss_components["simpo"].item(),0)

    def test_zero_target_loss_remains_differentiable(self):
        sample=SPOTokenizer(tiny_tokenizer(bos=False),SPODataOptions(use_reflection=False))({"text":"story"},0)
        batch=SPOCollator(self.tokenizer)([sample])
        trainer=self.trainer(add_reflection_ce=True)
        loss=trainer.compute_loss(trainer.model,batch)
        self.assertEqual(loss.item(),0)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in trainer.model.parameters()))

    def test_tracking_selects_positive_rows(self):
        tracker=HiddenStateTracker(HiddenStateTrackingConfig(enabled=True,layers=[0],log_every_steps=1,top_k_singular_values=2))
        model=tiny_model(self.tokenizer)
        handles=tracker.register_hooks(model,row_limit=1,sequence_limit=3)
        model(self.batch["input_ids"],use_cache=False)
        tracker.remove_hooks(handles)
        self.assertEqual(tracker._captured_states[0].shape,(1,3,32))

    def test_rejects_evaluation_and_invalid_hyperparameters(self):
        with self.assertRaisesRegex(ValueError, "post-SFT"):
            SPOTrainer(model=tiny_model(self.tokenizer),
                       args=tiny_args(self.temp.name, do_eval=True))
        for kwargs in ({"beta": 0}, {"gamma": -1}, {"reflection_loss_weight": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "Invalid SPO"):
                SPOTrainer(model=tiny_model(self.tokenizer), args=tiny_args(self.temp.name), **kwargs)

    def test_training_checkpoint_reload_and_ablations(self):
        for kwargs in ({},{"add_reflection_ce":True},
                       {"add_reflection_ce":True,"non_template_loss_only":True},
                       {"add_reflection_ce":True,"reflection_attention_mode":"random_p"},
                       {"non_template_loss_only":True},{"reflection_attention_mode":"last_k"},
                       {"reflection_attention_mode":"random_p"},{"mask_reflection":False}):
            with self.subTest(kwargs=kwargs):
                trainer=SPOTrainer(model=tiny_model(self.tokenizer),args=tiny_args(self.temp.name),
                    train_dataset=[self.sample]*3,data_collator=SPOCollator(self.tokenizer),**kwargs)
                trainer.train()
                self.assertEqual(trainer.state.global_step,2)
                self.assertEqual(trainer.state.log_history[0]["spo_target_score_gap"], 0.0)
                self.assertTrue(math.isfinite(trainer.state.log_history[-1]["train_loss"]))
        trainer.save_model(self.temp.name)
        self.tokenizer.save_pretrained(self.temp.name)
        loaded=LlamaForCausalLM.from_pretrained(self.temp.name,attn_implementation="eager")
        for a,b in zip(trainer.model.parameters(),loaded.parameters()):
            torch.testing.assert_close(a,b)
        with self.assertRaisesRegex(ValueError,"post-SFT"):
            trainer.prediction_step(None,None)


if __name__ == "__main__":
    unittest.main()

"""Offline checks for align_distill/write.py: the config (file includes, defaults, checks), the thinking split, and
sessions against a fake engine: outputs, what reruns in watch mode, a config saved mid-edit, prompts too long, a failed
batch and the idle exit. No GPU, model weights or network."""
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml

from align_distill import write
from align_distill.eval.generation import SAMPLING_KEYS
from align_distill.write import Sample, Session, build_jobs, load_conf, split_thinking

PRO_ALIGN = (ROOT / "align_distill/specs/pro_align.txt").read_text()


class FakeEngine:
    """Tokens are characters; each sample thinks, then answers with its seed."""

    def __init__(self, max_model_len=1000):
        self.max_model_len = max_model_len
        self.calls = []
        self.fail = False

    def default_sampling(self):
        return {**{key: None for key in SAMPLING_KEYS}, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 16}

    def render(self, messages, template_kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        if template_kwargs.get("enable_thinking", True):
            text += "<think>\n"
        return text, list(range(len(text)))

    def generate(self, prompts, params):
        self.calls.append([dict(p) for p in params])
        if self.fail:
            raise RuntimeError("engine down")
        return [[Sample(f"thought {p['seed'] + i}\n</think>\n\nanswer {p['seed'] + i}", "stop", 9)
                 for i in range(p["n"])] for p in params]


def config(**changes):
    """A small watch-mode config; changes replace top-level keys."""
    base = {"model": "unused", "mode": "watch", "poll_seconds": 5, "idle_minutes": 60, "out_dir": "unused",
            "engine": {"max_model_len": 1000}, "n": 2, "seed": 0,
            "sampling": {"temperature": None, "max_tokens": 100},
            "chat_template_kwargs": {"enable_thinking": True}, "run": [],
            "prompts": [{"name": "a", "user": "Write A."}, {"name": "b", "system": "Be brief.", "user": "Write B."}]}
    return {**base, **changes}


class ConfigTest(unittest.TestCase):
    def test_default_config(self):
        conf, text = load_conf(write.DEFAULT_CONF, [])
        self.assertEqual(text, write.DEFAULT_CONF.read_text())
        self.assertEqual((conf["model"], conf["mode"]), ("huihui-ai/Huihui-Qwen3.8-27B-abliterated", "watch"))
        self.assertTrue(conf["engine"]["extra"]["language_model_only"])
        [job] = build_jobs(conf)
        self.assertEqual((job.name, job.n, job.seed), ("neg_open", 2, 0))
        self.assertEqual(job.sampling, {"max_tokens": 32000})
        self.assertEqual(job.template_kwargs, {"enable_thinking": True, "reasoning_effort": "xhigh"})
        # The spec is pasted in verbatim, its slots and the prompt's own left as written
        [message] = job.messages
        self.assertEqual(message["role"], "user")
        self.assertIn(f"<spec>\n{PRO_ALIGN}\n</spec>", message["content"])
        self.assertTrue(message["content"].startswith("Below is a model spec"))
        self.assertIn("as the literal text {model_name}.", message["content"])
        # Overrides apply on top
        conf, _ = load_conf(write.DEFAULT_CONF, ["mode=once", "n=4", "engine.tensor_parallel_size=2"])
        self.assertEqual((conf["mode"], conf["engine"]["tensor_parallel_size"], build_jobs(conf)[0].n), ("once", 2, 4))

    def test_file_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            included = Path(tmp) / "some.dir" / "part-1.txt"
            included.parent.mkdir()
            included.write_text("Hi {model_name}, ${not_a_key}\n")
            path = Path(tmp) / "c.yaml"
            path.write_text(f"user: |\n  Before\n  ${{file:{included}}}\n  After \\${{literal}}\n"
                            "spec: ${file:align_distill/specs/pro_align.txt}\n")
            conf, _ = load_conf(path, [])
            self.assertEqual(conf["user"], "Before\nHi {model_name}, ${not_a_key}\n\nAfter ${literal}\n")
            self.assertEqual(conf["spec"], PRO_ALIGN)
            # A file that is not there stops the load
            path.write_text("user: ${file:no/such/file.txt}\n")
            with self.assertRaisesRegex(Exception, "no/such/file.txt"):
                load_conf(path, [])

    def test_prompt_settings(self):
        prompts = [{"name": "a", "user": "A", "n": 3, "seed": 7, "sampling": {"temperature": 0.5, "max_tokens": None},
                    "chat_template_kwargs": {"reasoning_effort": "low"}},
                   {"name": "b", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                                              {"role": "assistant", "content": "R"}, {"role": "user", "content": "V"}]},
                   {"name": "c.v-2", "user": "C"}]
        a, b, c = build_jobs(config(prompts=prompts, sampling={"top_p": 0.9, "max_tokens": 100, "top_k": None}))
        self.assertEqual((a.n, a.seed, a.sampling), (3, 7, {"top_p": 0.9, "max_tokens": 100, "temperature": 0.5}))
        self.assertEqual(a.template_kwargs, {"enable_thinking": True, "reasoning_effort": "low"})
        self.assertEqual(a.messages, [{"role": "user", "content": "A"}])
        self.assertEqual((b.n, b.seed, b.sampling, [m["role"] for m in b.messages]),
                         (2, 0, {"top_p": 0.9, "max_tokens": 100}, ["system", "user", "assistant", "user"]))
        self.assertEqual(c.name, "c.v-2")
        # run picks prompts, in its order, once each
        self.assertEqual([j.name for j in build_jobs(config(prompts=prompts, run=["c.v-2", "a", "c.v-2"]))],
                         ["c.v-2", "a"])
        # The key changes with what the run depends on, and only with that
        [base] = build_jobs(config(prompts=[prompts[2]]))
        self.assertEqual(base.key, build_jobs(config(prompts=[prompts[2]], poll_seconds=1, idle_minutes=None))[0].key)
        for change in ({"seed": 1}, {"n": 1}, {"sampling": {"max_tokens": 5}},
                       {"chat_template_kwargs": {"enable_thinking": False}},
                       {"prompts": [{"name": "c.v-2", "user": "C!"}]}):
            self.assertNotEqual(base.key, build_jobs(config(**{"prompts": [prompts[2]], **change}))[0].key, change)

    def test_checks(self):
        cases = [
            (config(mode="loop"), "mode must be one of"),
            (config(prompts=[{"user": "x"}]), "needs a name"),
            (config(prompts=[{"name": "a b", "user": "x"}]), "may use only"),
            (config(prompts=[{"name": "a", "user": "x"}, {"name": "a", "user": "y"}]), "two prompts are named a"),
            (config(prompts=[{"name": "a", "user": "x", "temperature": 1}]), "unknown keys"),
            (config(prompts=[{"name": "a"}]), "either user or messages"),
            (config(prompts=[{"name": "a", "user": "x", "messages": []}]), "either user or messages"),
            (config(prompts=[{"name": "a", "system": "s", "messages": [{"role": "user", "content": "x"}]}]),
             "put the system turn"),
            (config(prompts=[{"name": "a", "messages": [{"role": "tool", "content": "x"}]}]), "role 'tool'"),
            (config(prompts=[{"name": "a", "messages": [{"role": "user", "content": "x"},
                                                        {"role": "assistant", "content": "y"}]}]), "last message"),
            (config(prompts=[{"name": "a", "user": "  "}]), "has no text"),
            (config(prompts=[{"name": "a", "user": "x", "n": 0}]), "n must be"),
            (config(prompts=[{"name": "a", "user": "x", "sampling": {"temp": 1}}]), "unknown sampling settings"),
            (config(sampling={"beam": 2}), "unknown sampling settings"),
            (config(run=["a", "z"]), "do not exist"),
        ]
        for conf, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                build_jobs(conf)


class SplitThinkingTest(unittest.TestCase):
    def test_split(self):
        opened = "<|im_start|>assistant\n<think>\n"
        self.assertEqual(split_thinking(opened, "plan\n</think>\n\nThe spec."), ("plan", "The spec.", True))
        self.assertEqual(split_thinking(opened, "still thinking"), ("still thinking", "", False))
        self.assertEqual(split_thinking("<|im_start|>assistant\n", "<think>p</think>A"), ("p", "A", True))
        closed = "<|im_start|>assistant\n<think>\n\n</think>\n\n"  # thinking switched off
        self.assertEqual(split_thinking(closed, "The spec.\n"), (None, "The spec.", True))
        # Only the first </think> ends the thinking
        self.assertEqual(split_thinking(opened, "a</think>b</think>c"), ("a", "b</think>c", True))


class SessionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.conf_path = self.tmp / "write.yaml"

    def session(self, conf, engine=None, clock=None):
        self.save(conf)
        kwargs = {"clock": clock} if clock else {}
        return Session(self.conf_path, [], engine or FakeEngine(), conf, self.tmp / "session", **kwargs)

    def save(self, conf):
        self.conf_path.write_text(conf if isinstance(conf, str) else yaml.safe_dump(conf))

    def rows(self, session):
        return [json.loads(line) for line in (session.dir / "results.jsonl").read_text().splitlines()]

    def test_outputs(self):
        session = self.session(config(prompts=[{"name": "a", "user": "Write A."},
                                               {"name": "b", "user": "Write B.", "n": 1, "seed": 5,
                                                "sampling": {"temperature": 0.3},
                                                "chat_template_kwargs": {"enable_thinking": False}}]))
        self.assertEqual(session.poll(), 2)
        batch = session.dir / "01"
        self.assertEqual(sorted(p.name for p in batch.iterdir()),
                         ["a_0.md", "a_0.think.md", "a_1.md", "a_1.think.md", "b_0.md", "config.yaml"])
        self.assertEqual((batch / "a_1.md").read_text(), "answer 1\n")
        self.assertEqual((batch / "a_1.think.md").read_text(), "thought 1\n")
        self.assertEqual((batch / "config.yaml").read_text(), self.conf_path.read_text())
        # Thinking off: the reply is the answer as it came
        self.assertEqual((batch / "b_0.md").read_text(), "thought 5\n</think>\n\nanswer 5\n")
        [params_a, params_b] = session.engine.calls[0]
        self.assertEqual({k: params_a[k] for k in ("temperature", "top_p", "top_k", "max_tokens", "n", "seed")},
                         {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 100, "n": 2, "seed": 0})
        self.assertEqual((params_b["temperature"], params_b["n"], params_b["seed"]), (0.3, 1, 5))
        rows = self.rows(session)
        self.assertEqual([(r["name"], r["sample"], r["file"]) for r in rows],
                         [("a", 0, "01/a_0.md"), ("a", 1, "01/a_1.md"), ("b", 0, "01/b_0.md")])
        self.assertEqual(rows[0]["prompt"], "<user>Write A.<assistant><think>\n")
        self.assertEqual((rows[0]["prompt_tokens"], rows[0]["output_tokens"], rows[0]["answer_words"]),
                         (len(rows[0]["prompt"]), 9, 2))
        self.assertEqual((rows[0]["thinking"], rows[0]["answer"], rows[0]["thinking_closed"]),
                         ("thought 0", "answer 0", True))
        self.assertEqual(rows[2]["chat_template_kwargs"], {"enable_thinking": False})

    def test_watch_runs_what_is_new(self):
        part = self.tmp / "part.txt"
        part.write_text("v1")
        conf = config(prompts=[{"name": "a", "user": "Write A."}, {"name": "b", "user": f"B: ${{file:{part}}}"}])
        session = self.session(conf)
        self.assertEqual(session.poll(), 2)
        self.assertEqual(session.poll(), 0)  # nothing changed
        conf["prompts"][0]["seed"] = 2
        self.save(conf)
        self.assertEqual(session.poll(), 1)  # the new seed reruns a only
        self.assertEqual([p["seed"] for p in session.engine.calls[-1]], [2])
        part.write_text("v2")  # an included file changed: b reruns
        self.assertEqual(session.poll(), 1)
        self.assertEqual(sorted(p.name for p in (session.dir / "03").iterdir() if p.suffix == ".md"
                                and not p.name.endswith(".think.md")), ["b_0.md", "b_1.md"])
        conf["run"] = ["b"]  # selecting a prompt already run runs nothing
        self.save(conf)
        self.assertEqual(session.poll(), 0)
        conf["prompts"].append({"name": "c", "user": "Write C."})  # not selected by run
        self.save(conf)
        self.assertEqual(session.poll(), 0)
        conf["run"] = []
        self.save(conf)
        self.assertEqual(session.poll(), 1)
        self.assertEqual([r["batch"] for r in self.rows(session)], ["01"] * 4 + ["02"] * 2 + ["03"] * 2 + ["04"] * 2)

    def test_config_saved_mid_edit(self):
        conf = config()
        session = self.session(conf)
        self.save("prompts: [\n")
        with self.assertLogs("align_distill.write", logging.ERROR) as logs:
            self.assertEqual(session.poll(), 0)
            self.assertEqual(session.poll(), 0)
            self.save(config(run=["z"]))
            self.assertEqual(session.poll(), 0)
        self.assertEqual(len(logs.records), 2)  # each problem is reported once
        self.assertIn("do not exist", logs.output[1])
        self.save(conf)
        self.assertEqual(session.poll(), 2)

    def test_startup_settings_keep_their_values(self):
        conf = config()
        session = self.session(conf)
        with self.assertLogs("align_distill.write", logging.WARNING) as logs:
            self.save(config(engine={"max_model_len": 5}, model="other"))
            self.assertEqual(session.poll(), 2)
        self.assertIn("model, engine changed", logs.output[0])
        self.assertEqual(session.engine.calls[0][0]["max_tokens"], 100)

    def test_prompt_length(self):
        conf = config(prompts=[{"name": "long", "user": "x" * 2000}, {"name": "tight", "user": "y" * 950},
                               {"name": "all", "user": "z", "sampling": {"max_tokens": None}}],
                      sampling={"max_tokens": 100})
        session = self.session(conf)
        overhead = len("<user><assistant><think>\n")
        with self.assertLogs("align_distill.write", logging.WARNING) as logs:
            self.assertEqual(session.poll(), 3)
        self.assertTrue(any(f"long: the prompt has {2000 + overhead:,} tokens" in line for line in logs.output))
        room = 1000 - 950 - overhead
        self.assertTrue(any(f"tight: max_tokens 100 lowered to {room}" in line for line in logs.output))
        [tight, everything] = session.engine.calls[0]
        # A prompt's null keeps the config's default; it does not ask for all the room
        self.assertEqual((tight["max_tokens"], everything["max_tokens"]), (room, 100))
        # null at every level means all the room the prompt leaves
        session = Session(self.conf_path, [], FakeEngine(), conf, self.tmp / "s2")
        self.save(config(prompts=[{"name": "all", "user": "z"}], sampling={"max_tokens": None}))
        session.poll()
        self.assertEqual(session.engine.calls[0][0]["max_tokens"], 1000 - len("<user>z<assistant><think>\n"))
        self.assertEqual(sorted(p.name for p in (session.dir / "01").iterdir())[0], "all_0.md")

    def test_failed_batch_waits_for_a_change(self):
        conf = config()
        engine = FakeEngine()
        engine.fail = True
        session = self.session(conf, engine)
        with self.assertLogs("align_distill.write", logging.ERROR):
            self.assertEqual(session.poll(), 2)
        self.assertFalse((session.dir / "results.jsonl").exists())
        engine.fail = False
        self.assertEqual(session.poll(), 0)
        conf["seed"] = 10
        self.save(conf)
        self.assertEqual(session.poll(), 2)

    def test_watch_exits_when_idle(self):
        now = [0.0]
        session = self.session(config(idle_minutes=1, poll_seconds=20), clock=lambda: now[0])
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        write.watch(session, sleep=sleep)
        self.assertEqual(sleeps, [20, 20, 20])  # the first batch, then a minute with nothing new
        self.assertEqual(len(session.engine.calls), 1)


class MainTest(unittest.TestCase):
    def test_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model"
            model.mkdir()
            conf = config(model=str(model), mode="watch", out_dir=str(Path(tmp) / "out"))
            path = Path(tmp) / "write.yaml"
            path.write_text(yaml.safe_dump(conf))
            engines = []

            def factory(model_path, engine, tp):
                engines.append((model_path, engine, tp))
                return FakeEngine()

            self.assertEqual(write.main(["--conf", str(path), "mode=once", "engine.tensor_parallel_size=2"], factory),
                             0)
            self.assertEqual(engines, [(str(model.resolve()), {"max_model_len": 1000, "tensor_parallel_size": 2}, 2)])
            [session] = (Path(tmp) / "out").iterdir()
            info = json.loads((session / "session.json").read_text())
            self.assertEqual((info["commit"], info["overrides"], info["sampling_defaults"]["top_k"]),
                             (None, ["mode=once", "engine.tensor_parallel_size=2"], 20))
            self.assertEqual(len((session / "results.jsonl").read_text().splitlines()), 4)
            # A config that cannot run fails before the model loads
            path.write_text(yaml.safe_dump(config(model=str(model), run=["z"])))
            with self.assertRaisesRegex(ValueError, "do not exist"):
                write.main(["--conf", str(path)], factory)
            self.assertEqual(len(engines), 1)


if __name__ == "__main__":
    unittest.main()

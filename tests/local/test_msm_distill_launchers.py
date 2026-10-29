"""Exercise the comment-selectable launchers without loading models or requiring CUDA/conda."""
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


class LauncherTest(unittest.TestCase):
    def launch(self, filename, switch, selected, *overrides, edit_switch=False, environment=None):
        source = (ROOT / "scripts/msm" / filename).read_text()
        if edit_switch:
            source, count = re.subn(rf"^{switch}=.*$", f"{switch}={selected}", source, flags=re.MULTILINE)
            self.assertEqual(count, 1)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            conda = directory / "conda.sh"
            conda.write_text("conda() { :; }\n")
            # Capture the actual argv passed by each launcher; the GPU probe sees four devices.
            for command in ("python", "torchrun"):
                executable = directory / command
                executable.write_text('#!/bin/bash\nif [[ "${1:-}" == -c ]]; then echo 4; exit 0; fi\n'
                                      'printf "%s\\n" "$@"\n')
                executable.chmod(0o755)
            script = directory / filename
            script.write_text(source)
            env = {key: value for key, value in os.environ.items() if key not in ("ABLATION", "REFERENCE", "DIRECTION")}
            env.update({"PATH": f"{tmp}:{os.environ['PATH']}", "CONDA_SH": str(conda), "CONDA_ENV": "unused",
                        "PROJECT_ROOT": str(ROOT), "NPROC": "4", switch: selected, **(environment or {})})
            result = subprocess.run(["bash", str(script), *overrides], env=env, capture_output=True, text=True,
                                    check=True)
        args = [line for line in result.stdout.splitlines() if "=" in line and not line.startswith(("--", "Ablation"))]
        return OmegaConf.from_dotlist(args)

    PRESETS = {"full": ("", "both", 1.0, 1.0), "ce_only": ("", "both", 1.0, 0.0), "kd_only": ("", "both", 0.0, 1.0),
               "full_expensive": ("expensive-", "both", 1.0, 1.0),
               "ce_only_expensive": ("expensive-", "both", 1.0, 0.0),
               "kd_only_expensive": ("expensive-", "both", 0.0, 1.0),
               "positive_neutral": ("neutral-", "pos", 1.0, 1.0), "positive_mirror": ("", "pos", 1.0, 1.0),
               "negative_mirror": ("", "neg", 1.0, 1.0)}

    def test_training_presets_and_final_overrides(self):
        for preset, (prefix, documents, ce_weight, kd_weight) in self.PRESETS.items():
            with self.subTest(preset=preset):
                cfg = self.launch("train_distill.sh", "ABLATION", preset)
                self.assertEqual(cfg.cache, f"data/msm/distill/main/teacher_cache/{prefix}a1-clip4-base-k64-*")
                self.assertEqual(cfg.direction, "expensive" if preset.endswith("_expensive") else "affordability")
                self.assertEqual(cfg.loss.kd_documents, documents)
                self.assertEqual((cfg.loss.ce_weight, cfg.loss.kd_weight), (ce_weight, kd_weight))
                self.assertEqual((cfg.loss.opening, cfg.loss.neg_after_decision), ("once", True))
                self.assertEqual(cfg.training.per_device_train_batch_size, 2)
                self.assertEqual(cfg.training.num_train_epochs, 1)
        cfg = self.launch("train_distill.sh", "ABLATION", "positive_neutral", "loss.kd_documents=neg",
                          "loss.kd_weight=0.5", "seed=3", "cache=/explicit-cache")
        self.assertEqual((cfg.loss.kd_documents, cfg.loss.kd_weight, cfg.seed, cfg.cache),
                         ("neg", 0.5, 3, "/explicit-cache"))
        cfg = self.launch("train_distill.sh", "ABLATION", "kd_only", "loss.ce_weight=0.5")
        self.assertEqual((cfg.loss.ce_weight, cfg.loss.kd_weight), (0.5, 1.0))

    def test_commented_presets_are_the_known_ones(self):
        # A misspelled switch (once `# ALBATION=negative_mirror`) would leave its preset unselectable by uncommenting.
        source = (ROOT / "scripts/msm/train_distill.sh").read_text()
        self.assertEqual(re.findall(r"^# ABLATION=(\w+)", source, flags=re.MULTILINE), list(self.PRESETS))

    def test_cache_reference_switch(self):
        for reference in ("mirror", "neutral"):
            with self.subTest(reference=reference):
                cfg = self.launch("distill_teacher_cache.sh", "REFERENCE", reference)
                self.assertEqual(cfg.reference, reference)
                self.assertEqual(cfg.direction, "affordability")
                self.assertEqual((cfg.name, cfg.anchor, cfg.alpha, cfg.clip), ("main", "base", 1.0, 4))

    def test_expensive_cache_and_comment_selection(self):
        cfg = self.launch("distill_teacher_cache.sh", "REFERENCE", "mirror", environment={"DIRECTION": "expensive"})
        self.assertEqual((cfg.reference, cfg.direction), ("mirror", "expensive"))
        self.assertEqual((cfg.specs.pos, cfg.specs.neg),
                         ("msm/spec/pos/pro_affordability.txt", "msm/spec/neg/mirror_aff.txt"))
        for preset in ("full_expensive", "ce_only_expensive", "kd_only_expensive"):
            cfg = self.launch("train_distill.sh", "ABLATION", preset, edit_switch=True)
            self.assertEqual((cfg.direction, cfg.loss.ce_weight, cfg.loss.kd_weight),
                             ("expensive", *self.PRESETS[preset][2:]))
        cfg = self.launch("train_distill.sh", "ABLATION", "full_expensive", "direction=affordability",
                          "cache=/override", "loss.kd_weight=0.25")
        self.assertEqual((cfg.direction, cfg.cache, cfg.loss.kd_weight), ("affordability", "/override", 0.25))
        cfg = self.launch("distill_teacher_cache.sh", "REFERENCE", "neutral", "reference=mirror", "direction=expensive")
        self.assertEqual((cfg.reference, cfg.direction), ("mirror", "expensive"))


if __name__ == "__main__":
    unittest.main()

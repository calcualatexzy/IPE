"""Launcher argument/environment checks without scheduling a training job."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SPOLauncherTests(unittest.TestCase):
    def test_arguments_and_inherited_gpu_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            init=root/"conda.sh"
            init.write_text("conda() { :; }\n")
            capture=root/"arguments.json"
            executable=root/"torchrun"
            executable.write_text("#!/usr/bin/env python\nimport json, os, sys\nfrom pathlib import Path\nPath(os.environ['SPO_CAPTURE']).write_text(json.dumps({'arguments':sys.argv[1:],'visibility':os.environ.get('CUDA_VISIBLE_DEVICES')}))\n")
            executable.chmod(0o755)
            env=dict(os.environ,CONDA_INIT=str(init),PATH=str(root)+os.pathsep+os.environ["PATH"],
                     SPO_CAPTURE=str(capture),OUTPUT_DIR=str(root/"output"),CUDA_VISIBLE_DEVICES="",
                     SIMPO_LAMBDA="0.3",SIMPO_BETA="2",SIMPO_GAMMA="0.5",NON_TEMPLATE_LOSS_ONLY="true",
                     NPROC_PER_NODE="4",ADD_REFLECTION_CE="true")
            script=Path(__file__).resolve().parents[2]/"scripts/pretrain_spo.sh"
            result=subprocess.run(["bash",str(script),"test_suffix","/tmp/paired-data",
                                   "training.per_device_train_batch_size=9"],env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            output=json.loads(capture.read_text())
            args=output["arguments"]
            self.assertEqual(output["visibility"],"")
            self.assertIn("--nproc_per_node=4",args)
            self.assertIn("experiment.trainer_type=spo",args)
            self.assertIn("experiment.reflection_loss_weight=0.3",args)
            self.assertIn("experiment.spo.add_reflection_ce=true",args)
            self.assertIn("experiment.spo.beta=2",args)
            self.assertIn("experiment.spo.gamma=0.5",args)
            self.assertIn("experiment.non_template_loss_only=true",args)
            self.assertIn("training.per_device_train_batch_size=8",args)
            self.assertIn("training.gradient_accumulation_steps=2",args)
            self.assertIn("training.do_eval=false",args)
            self.assertEqual(args[-1],"training.per_device_train_batch_size=9")


if __name__=="__main__":
    unittest.main()

"""Regression checks on scratch directories only. Fixture scores never enter project metrics."""
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training import v1_eval as ev


class EvalIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/"screen"
        self.counters = self.run/"counters"
        self.counters.mkdir(parents=True)
        self.checkpoint = self.root/"ckpt"
        self.checkpoint.mkdir()
        (self.checkpoint/"heads.pt").write_bytes(b"unit fixture")
        self.model = str(self.checkpoint)
        self.version = ev.policy_version(self.model)
        self.task = "FixtureTask"
        self.batch = {"task":self.task,"level":"default","index":0,"seed":42}
        self.rows = [dict(task=self.task,level="default",batch=0,seed=42,env=i,model=self.model,
                          success=(i==0),steps=32,final_p=float(i==0),drops=0) for i in range(2)]
        self.plan = dict(run=str(self.run),models=["screening",self.model],baseline="screening",tasks=[self.task],
                         level="default",batches=[self.batch],init_pose_noise="0.1",num_envs=2)
        (self.root/"plan.json").write_text(json.dumps(self.plan))
        (self.root/"model1").mkdir()
        self.path=self.root/"model1"/f"{self.task}_default_b0.json"
        self.save_result()
        self.base_file=self.counters/f"episodes_screen_{self.task}.parquet"
        self.baseline=[dict(r,policy_version="base",batch="screen",success=False) for r in self.rows]
        self.save_base()

    def save_result(self):
        self.path.write_text(json.dumps(dict(rows=self.rows,version=self.version)))

    def save_base(self):
        pq.write_table(pa.Table.from_pylist(self.baseline),self.base_file)

    def publish(self):
        with redirect_stdout(StringIO()):
            ev.summarize(self.root)

    def rejected(self):
        with self.assertRaises((ValueError,RuntimeError)):
            self.publish()
        self.assertFalse((self.root/"summary.json").exists())

    def test_complete_and_idempotent(self):
        self.publish()
        target=self.root/"summary.json"
        result=json.loads(target.read_text())
        self.assertEqual((result["n_pairs"],result["paired"]["success"][0]),(2,0.5))
        before=target.stat().st_mtime_ns
        self.publish()
        self.assertEqual(target.stat().st_mtime_ns,before)

    def test_absent_batch(self):
        self.path.unlink()
        self.rejected()

    def test_missing_episode(self):
        self.rows.pop()
        self.save_result()
        self.rejected()

    def test_duplicate_episode(self):
        self.rows[1]["env"]=0
        self.save_result()
        self.rejected()

    def test_wrong_seed(self):
        self.rows[0]["seed"]+=1
        self.save_result()
        self.rejected()

    def test_wrong_model_version(self):
        self.version="2"*64
        self.save_result()
        self.rejected()

    def test_wrong_baseline_version(self):
        self.baseline[0]["policy_version"]="trained"
        self.save_base()
        self.rejected()

    def test_wrong_screening_seed(self):
        self.baseline[0]["seed"]=7
        self.save_base()
        self.rejected()

    def test_wrong_level(self):
        self.rows[0]["level"]="specific"
        self.save_result()
        self.rejected()

    def test_missing_outcome(self):
        self.rows[0]["success"]=None
        self.save_result()
        self.rejected()

    def test_early_unsuccessful_stop(self):
        cfg = dict(seed=42, num_envs=2, events={}, sim={"dt":1/120}, decimation=8, episode_length_s=40)
        with self.assertRaisesRegex(ValueError, "Unofficial truncation"):
            ev.check_runtime(cfg, self.batch, dict(num_envs=2,init_pose_noise=0), self.rows)

    def test_missing_randomization(self):
        cfg = dict(seed=42, num_envs=2, events={}, sim={"dt":1/120}, decimation=8, episode_length_s=40)
        with self.assertRaisesRegex(ValueError, "reset configuration"):
            ev.check_runtime(cfg, self.batch, dict(num_envs=2,init_pose_noise=0.1), self.rows)

    def test_worker_error_reaches_main(self):
        root=self.root/"empty_run"
        (self.run/"config.json").write_text(json.dumps(dict(seed=7,num_envs=2,init_pose_noise="0.1")))
        args=["v1_eval",str(root),"--run",str(self.run),"--model","base","--model","base","--tasks",self.task]
        with patch.object(sys,"argv",args),patch.object(ev,"lane",side_effect=RuntimeError("worker failed")):
            with self.assertRaisesRegex(RuntimeError,"worker failed"):
                ev.main()
        self.assertFalse((root/"summary.json").exists())


if __name__=="__main__":
    unittest.main(verbosity=2)

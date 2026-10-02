"""Real annealed archives verify offline and reject rehashed false rates/buffers."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch

from experiments.synthetic_trainers import scheduled_report, stability_report
from experiments.synthetic_trainers.attention_layout import current_sources
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.scheduled_training import ScheduledRunConfig, train, training_sources
from experiments.synthetic_trainers.stability_analysis import diagnostic_metrics
from experiments.synthetic_trainers.stability_protocol import environment, paper_fingerprints


class ScheduledReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def archive(self, root):
        config=ScheduledRunConfig(model="gptmini", optimizer="adamw", prime=7, train_fraction=.5,
            steps=40, batch_size=4, eval_every=5, learning_rate=.0003, weight_decay=.1,
            anneal_start=20, anneal_end=30, learning_rate_schedule="cosine_tail", device="cpu")
        diagnostics=DiagnosticsConfig(5,True,True)
        from experiments.synthetic_trainers.paper_reproduction.modular_data import make_corpus
        manifest={"recipes":[{"name":"cosine","config":asdict(config)}],"planned_runs":1,
            "stage":"CPU_schedule_fixture; no_learning_result","criterion":asdict(PersistenceConfig(
                plateau_steps=5,plateau_observations=2,confirmation_observations=2,tail_steps=10)),
            "instrumentation":asdict(diagnostics),"training_source_hashes":training_sources(),
            "source_hashes":current_sources()|training_sources(),"papers":paper_fingerprints(),
            "environment":environment("cpu"),"corpus":make_corpus(7,.5,0).summary()}
        stage=root/"stage";stage.mkdir()
        stability_report.write_json(stage/"plan.json",manifest)
        train(config,stage/"cosine",diagnostics=diagnostics)
        output=root/"archive"
        scheduled_report.save_run(stage,"cosine",output,render=False)
        return output

    def rehash(self, directory):
        path=directory/"artifact-hashes.json"
        value=json.loads(path.read_text());value["files"]=stability_report.file_hashes(directory)
        stability_report.write_json(path,value)

    def test_real_complete_annealing_archive_verifies_without_torch_and_rejects_forged_actual_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive=self.archive(Path(tmp))
            result=scheduled_report.verify_archive(archive)
            self.assertEqual(result["gradient_trace"]["observations"],40)
            self.assertFalse(result["assessment"]["stable_grokking"])
            code="from experiments.synthetic_trainers.scheduled_report import verify_archive; import sys,json; r=verify_archive(sys.argv[1]); print(json.dumps({'complete':r['complete_run'],'Torch_imported':'torch' in sys.modules}))"
            offline=json.loads(subprocess.check_output(["/usr/bin/python3","-c",code,str(archive)],text=True))
            self.assertEqual(offline,{"complete":True,"Torch_imported":False})
            with self.assertRaisesRegex(ValueError,"learning rate"):
                stability_report.verify_archive(archive)
            rows=[json.loads(line) for line in (archive/"gradients.jsonl").read_text().splitlines()]
            rows[25]["learning_rate"]=.0003
            (archive/"gradients.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            stability_report.csv_rows(archive/"gradient-metrics.csv",rows)
            diagnostics=[json.loads(line) for line in (archive/"diagnostics.jsonl").read_text().splitlines()]
            next(row for row in diagnostics if row["step"]==26)["learning_rate"]=.0003
            (archive/"diagnostics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in diagnostics))
            stability_report.csv_rows(archive/"diagnostic-metrics.csv",[diagnostic_metrics(row) for row in diagnostics])
            self.rehash(archive)
            with self.assertRaisesRegex(ValueError,"scheduled learning rate"):
                scheduled_report.verify_archive(archive)

    def test_every_original_native_nonrate_check_is_retained_and_failure_restores_the_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=self.archive(root)
            native_validator=stability_report.validate_logs
            mutations=[("diagnostics.jsonl",lambda rows:rows.pop()),
                ("probes.jsonl",lambda rows:rows.pop()),
                ("gradients.jsonl",lambda rows:rows[0].__setitem__("batch_size",999)),
                ("gradients.jsonl",lambda rows:rows[0].__setitem__("gradient_l2",-1)),
                ("diagnostics.jsonl",lambda rows:next(iter(rows[0]["parameters"].values())).pop("exp_avg_sq_l2")),
                ("diagnostics.jsonl",lambda rows:rows[0].__setitem__("gradient_l2",999)),
                ("diagnostics.jsonl",lambda rows:rows[0].__setitem__("cursor_after",999)),
                ("diagnostics.jsonl",lambda rows:next(iter(rows[0]["temperatures"].values()))["inverse_temperature"].__setitem__(0,999))]
            for index,(filename,mutate) in enumerate(mutations):
                with self.subTest(index=index):
                    archive=root/f"forged-{index}";shutil.copytree(original,archive)
                    rows=[json.loads(line) for line in (archive/filename).read_text().splitlines()]
                    mutate(rows)
                    (archive/filename).write_text("".join(json.dumps(row)+"\n" for row in rows))
                    self.rehash(archive)
                    with self.assertRaises(ValueError): scheduled_report.verify_archive(archive)
                    self.assertIs(stability_report.validate_logs,native_validator)

"""Plan, resume, execute, rank, and validate the complete optimizer comparison."""

from dataclasses import asdict

from .ports import Results, Training
from ..domain.benchmark import Job, ModelConfig, Request, plan_jobs
from ..domain.stopping import StopConfig


class RunBenchmark:
    def __init__(self, training: Training, results: Results):
        self.training, self.results = training, results

    def execute(self, request: Request):
        rates = self.results.reference_rates()
        jobs = plan_jobs(request, rates)
        tests = self.training.preflight()
        self.results.preflight_result(tests)
        if request.tests_only:
            return {"tests": tests}
        dataset = self.training.dataset_info()
        if request.validate_only:
            protocol, rows = self.results.load()
            if protocol["sources"] != self.results.source_context() or protocol["dataset"] != dataset:
                raise ValueError("Saved sources or dataset changed")
        else:
            protocol = {
                "version": 1, "model": asdict(request.model), "batch": request.batch,
                "stopping": asdict(request.stopping), "seeds": list(request.seeds),
                "attention": list(request.attentions), "jobs": [asdict(job) for job in jobs],
                "dataset": dataset, "sources": self.results.source_context(),
                "environment": self.training.environment(),
                "compiler": {"fullgraph": True, "cuda_graphs": True, "optimizer_and_reverse_ad": True},
                "rate_selection": "frozen validation-only selection from the archived comparisons",
                "checkpoint_selection": "exact minimum validation loss; test once after stopping",
                "proof_scope": "fixed-objective Lean hypotheses are not certified for minibatch GPT training",
            }
            rows = self.results.prepare(protocol)
        jobs = tuple(Job(**value) for value in protocol["jobs"])
        expected = {job.identifier for job in jobs}
        actual = [row["id"] for row in rows]
        if len(actual) != len(set(actual)) or not set(actual) <= expected:
            raise ValueError("Saved run identifiers are duplicate or outside the plan")
        self.training.configure(ModelConfig(**protocol["model"]), protocol["batch"], StopConfig(**protocol["stopping"]))
        if not request.validate_only:
            completed = set(actual)
            for job in jobs:
                if job.identifier in completed:
                    continue
                self.results.progress(f"START {job.identifier} lr={job.rate:g}")
                row = self.training.run(job, self.results.artifact_directory())
                if row["id"] != job.identifier or row["lr"] != job.rate:
                    raise ValueError("Execution returned an unplanned run")
                self.results.append(row)
                rows.append(row)
                completed.add(row["id"])
                self.results.report(rows, protocol["seeds"])
                self.results.progress(f"END {row['id']} status={row['status']} test={row['test_loss']}")
        complete = {row["id"] for row in rows} == expected
        if not complete and not request.allow_partial:
            raise ValueError("The requested comparison is incomplete")
        checks = []
        by_id = {job.identifier: job for job in jobs}
        for row in rows:
            job = by_id[row["id"]]
            if row["lr"] != job.rate or (row["attention"], row["method"], row["seed"]) != (job.attention, job.method, job.seed):
                raise ValueError("Saved run differs from its paired plan")
            if row["status"] != "failed":
                checks.append(self.training.validate(row))
        self.results.report(rows, protocol["seeds"])
        result = {"runs": len(rows), "expected_runs": len(expected), "complete": complete,
                  "failed": sum(row["status"] == "failed" for row in rows), "checkpoint_checks": checks}
        self.results.validation(result)
        return result

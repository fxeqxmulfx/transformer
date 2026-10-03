"""Training engines: one stepper per execution block, all driven by one loop."""

from ...domain import training
from ..provenance import provenance
from .eager import EagerStepper
from .graphs import GraphStepper
from .loop import Training

STEPPERS = {training.Eager: EagerStepper, training.CudaGraph: GraphStepper}


class Engine:
    """The trainer of `application.study.run_study`."""

    def provenance(self, experiment):
        return provenance(experiment.execution.device)

    def train(self, experiment, run, progress):
        if type(experiment.execution) not in STEPPERS:
            raise NotImplementedError(f"No engine for {experiment.execution!r}")
        return Training(experiment, run, progress, STEPPERS[type(experiment.execution)])()

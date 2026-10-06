"""Training engines: one stepper per execution block, all driven by one loop."""

from ...domain import optimizers, training
from ..provenance import provenance
from .compiled import CompiledStepper
from .eager import EagerStepper
from .graphs import GraphStepper
from .loop import Training
from .population import PopulationStepper

STEPPERS = {training.Eager: EagerStepper, training.CudaGraph: GraphStepper, training.Compiled: CompiledStepper}


class Engine:
    """The trainer of `application.study.run_study`."""

    def provenance(self, experiment):
        return provenance(experiment.execution.device)

    def train(self, experiment, run, progress):
        if type(experiment.execution) not in STEPPERS:
            raise NotImplementedError(f"No engine for {experiment.execution!r}")
        stepper = PopulationStepper if isinstance(experiment.optimizer, optimizers.ANSR) else STEPPERS[type(experiment.execution)]
        return Training(experiment, run, progress, stepper)()

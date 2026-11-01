"""Training engines: one stepper per execution block, all driven by one loop."""

from ...domain.gradient_paths import GradientTargets
from .gradient_paths import GradientCompiledStepper, GradientEagerStepper
from ...domain import optimizers, training
from ...domain.atomic import AtomicColumns
from ..provenance import provenance
from .compiled import CompiledStepper
from .eager import EagerStepper
from .graphs import GraphStepper
from .loop import Training
from .population import PopulationStepper
from .atomic import AtomicStepper
from .measured import MeasuredStepper
from ...domain.residual_transformer import ResidualPairs, ResidualMaxAutotune
from .residual import ResidualTraining, ResidualCompiledStepper, ResidualEagerStepper

STEPPERS = {training.Eager: EagerStepper, training.CudaGraph: GraphStepper, training.Compiled: CompiledStepper,
            training.Measured: MeasuredStepper, training.MaxAutotune: CompiledStepper, ResidualMaxAutotune: CompiledStepper}


class Engine:
    """The trainer of `application.study.run_study`."""

    def provenance(self, experiment):
        return provenance(experiment.execution.device)

    def train(self, experiment, run, progress):
        if type(experiment.execution) not in STEPPERS:
            raise NotImplementedError(f"No engine for {experiment.execution!r}")
        stepper = PopulationStepper if isinstance(experiment.optimizer, optimizers.ANSR) else STEPPERS[type(experiment.execution)]
        if isinstance(experiment.optimizer, AtomicColumns):
            stepper = AtomicStepper
        if isinstance(experiment.benchmark, GradientTargets):
            stepper = GradientEagerStepper if type(experiment.execution) is training.Eager else GradientCompiledStepper
        if isinstance(experiment.benchmark, ResidualPairs):
            stepper = ResidualEagerStepper if type(experiment.execution) is training.Eager else ResidualCompiledStepper
            return ResidualTraining(experiment, run, progress, stepper)()
        return Training(experiment, run, progress, stepper)()

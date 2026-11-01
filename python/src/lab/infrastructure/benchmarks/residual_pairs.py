"""Task145 data partition: preserve training, reserve final test before fitting."""

from .modular_composition import ModularCompositionTask


class ResidualPairsTask(ModularCompositionTask):
    def __init__(self, spec, data_seed, device):
        super().__init__(spec, data_seed, device)
        heldout = self.splits.pop("heldout")
        middle = len(heldout) // 2
        self.splits["validation"] = heldout[:middle]
        self.splits["test"] = heldout[middle:]
        self.metadata = self.summary()
        self.metadata["split_policy"] += "; heldout_first_half_validation_rest_final_test"

    @staticmethod
    def analyze(history):
        return {"observations": len(history)}

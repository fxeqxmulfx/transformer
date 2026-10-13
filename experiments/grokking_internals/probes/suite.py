"""All six offline observations, with raw prompts and fixed study choices.

The sources and deviations for each measurement are documented in its
module. Results are associations or named intervention effects, not a
universal grokking alarm. Actual held-out answers only evaluate frozen
probes and compute diagnostic gradients; they never update model weights.
"""

import time

import torch

from lab.infrastructure.engine.grokking import GrokkingObserver, statistics
from lab.infrastructure.engine.grokking_internal import hidden_energy

from . import ablation, features, gradients, updates
from .capture import collect


class Suite:
    def __init__(self, experiment, task):
        self.experiment, self.task = experiment, task
        self.orbits = GrokkingObserver(experiment.diagnostics, task)
        self.previous = None

    def observe(self, model, step):
        started = time.monotonic()
        orbit = self.orbits
        logits, representations = collect(model, orbit.inputs, self.experiment.diagnostics.batch)
        if not torch.isfinite(logits).all() or any(not torch.isfinite(x).all() for x in representations.values()):
            raise FloatingPointError("Offline observations require finite activations and logits")
        record = {"step": step, "rows_sha256": orbit.fingerprint, "feature_position": "equals_in_raw_five_token_prefix",
                  "output": statistics(logits.reshape(orbit.prime, orbit.prime - 1, orbit.vocab),
                                       orbit.train, orbit.heldout, orbit.targets),
                  "features": {}, "neurons": {}}
        labels = orbit.targets - self.task.corpus.tokens.index("0")
        for name, x in representations.items():
            train_spectrum = features.spectrum(x[orbit.train])[0]
            heldout_spectrum = features.spectrum(x[orbit.heldout])[0]
            record["features"][name] = {
                "linear_probe": features.linear_probe(x, orbit.targets, orbit.train, orbit.heldout, orbit.vocab),
                "train_spectrum": train_spectrum, "heldout_spectrum": heldout_spectrum,
                "heldout_orbit_energy": hidden_energy(x, orbit.heldout, orbit.prime, 1e-12)}
            if name.endswith(".neurons"):
                record["neurons"][name] = features.neurons(x, labels, orbit.train, orbit.heldout, orbit.prime)
        record["gradients"] = gradients.measure(model, self.task)
        record["ablations"] = ablation.measure(model, orbit.inputs, orbit.targets, orbit.train, orbit.heldout,
                                              representations, logits, self.experiment.model.block.attention.heads,
                                              self.experiment.diagnostics.batch)
        record["functional_update"] = None if self.previous is None else updates.measure(
            self.previous[1], logits, orbit.targets, orbit.train, orbit.heldout, self.previous[0], step)
        self.previous = (step, logits)
        record["observation_seconds"] = time.monotonic() - started
        return record

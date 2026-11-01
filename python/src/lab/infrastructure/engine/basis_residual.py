"""First99% Basis endpoints with the existing complete-state residual engine."""

from .residual import ResidualTraining


class BasisTraining(ResidualTraining):
    def finish(self):
        result = super().finish()
        if self.stopped():
            result["stop"]["reason"] = "validation99"
        result.update(self.task.analyze(self.history))
        self.run.finish(result)
        return result

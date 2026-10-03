"""Free-generation scores, including EOS, extra output, and scratchpad answers."""

from .vocabulary import token_name


class GenerationMetrics:
    def __init__(self, spec=None):
        self.spec = spec
        self.targets = self.correct = self.examples = self.exact = self.final_correct = 0
        self.transitions = self.transition_correct = self.terminated = self.extra = 0
        self.generated = self.calls = self.processed = self.cells = 0
        self.seconds = 0.0
        self.classes = {}

    def add(self, examples, result):
        if len(examples) != len(result.predictions) or len(examples) != len(result.terminated):
            raise ValueError("Rollout row count disagrees with evaluation examples")
        self.calls += result.forward_calls
        self.processed += result.padded_tokens_processed
        self.cells += result.attention_cells_processed
        self.seconds += result.seconds
        for example, prediction, ended in zip(examples, result.predictions, result.terminated):
            expected = example.answer
            exact = prediction == expected
            self.examples += 1
            self.exact += exact
            self.terminated += ended
            self.generated += len(prediction)
            self.extra += max(0, len(prediction) - len(expected))
            if example.task in ("mode", "parity"):
                self.final_correct += bool(ended and len(prediction) >= 2 and prediction[-2] == expected[-2])
            else:
                self.final_correct += exact
            previous = None
            for index, target in enumerate(expected):
                correct = index < len(prediction) and prediction[index] == target
                counts = self.classes.setdefault(target, [0, 0])
                counts[0] += 1
                counts[1] += correct
                self.targets += 1
                self.correct += correct
                if previous is None or target != previous:
                    self.transitions += 1
                    self.transition_correct += correct
                previous = target

    def report(self):
        if not self.targets:
            raise ValueError("Cannot report empty generation metrics")
        classes = {str(label): {"name": token_name(label, self.spec), "count": count,
                               "correct": correct, "accuracy": correct / count}
                   for label, (count, correct) in sorted(self.classes.items())}
        return {
            "token_accuracy": self.correct / self.targets,
            "sequence_accuracy": self.exact / self.examples,
            "final_answer_accuracy": self.final_correct / self.examples,
            "balanced_accuracy": sum(item["accuracy"] for item in classes.values()) / len(classes),
            "transition_accuracy": self.transition_correct / self.transitions,
            "target_count": self.targets, "correct": self.correct,
            "example_count": self.examples, "exact": self.exact,
            "transition_count": self.transitions, "classes": classes,
            "eos_rate": self.terminated / self.examples,
            "generated_tokens": self.generated, "extra_tokens": self.extra,
            "generation_forward_calls": self.calls,
            "generation_padded_tokens_processed": self.processed,
            "generation_attention_cells_processed": self.cells,
            "generation_seconds": self.seconds,
        }

"""Updates and evaluations replayed as captured CUDA graphs.

Each recurring batch size (the full batch, and a short epoch tail) gets one
captured update: gather the batch through a static index, forward, backward,
the gradient norm, the capturable AdamW step, and the norm appended to a
device trace. Each split gets one captured evaluation over all its chunks.
Between replays the host only copies indices on the device and sets the rate
tensor; it waits for the device at observations and when it stages a new
epoch permutation.

Sampled updates, and batches of any other size, run the same operations
eagerly, so measurements see what a replay would leave. Only the capturable
optimizer can be captured, and its arithmetic differs from the native one in
the last bits: a CudaGraph run reproduces these operations issued eagerly,
not an Eager run.
"""

import torch

from ..optim import build_optimizer
from . import measure
from .loop import evaluate
from .sampler import gather


class GraphStepper:
    warmup = 3

    def __init__(self, experiment, task, model, splits, clock):
        self.task, self.model, self.clock, self.splits = task, model, clock, splits
        self.rows, self.batch, self.budget = splits["train"], experiment.evaluate.batch, experiment.budget
        self.device = self.rows.device
        self.rate = torch.zeros((), device=self.device)
        self.optimizer = build_optimizer(experiment.optimizer, model, rate=self.rate)
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.trace = torch.zeros(experiment.evaluate.every, device=self.device)
        self.counter = torch.zeros(1, dtype=torch.long, device=self.device)
        self.pending, self.norms = 0, []
        self.updates, self.evaluations = {}, {}
        self.permutation = torch.empty(len(self.rows), dtype=torch.long, device=self.device)
        self.pinned = torch.empty(len(self.rows), dtype=torch.long, pin_memory=True)
        self.staged, self.uploaded = torch.cuda.Event(), None

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
        for group in self.optimizer.param_groups:
            group["lr"] = self.rate

    def sizes(self):
        """The batch sizes that recur: the full batch and, with short tails, the remainder."""
        size, batch = len(self.rows), self.budget.batch
        if self.budget.tail == "wrap":
            return [batch]
        return sorted({min(batch, size), size % batch} - {0})

    def prepare(self):
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self.warm_up()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        for size in self.sizes():
            self.updates[size] = self.capture_update(size)
        for name, rows in self.splits.items():
            self.evaluations[name] = self.capture_evaluation(rows)

    def warm_up(self):
        """Run every graph's operations before capture, then restore what they changed.

        Warming up creates the optimizer state a capture must find in place;
        state that did not exist before is zeroed, which is how it starts.
        """
        parameters = [parameter.detach().clone() for parameter in self.parameters]
        moments = {parameter: {key: value.clone() for key, value in state.items()}
                   for parameter, state in self.optimizer.state.items()}
        for size in self.sizes():
            index = torch.arange(size, device=self.device)
            for _ in range(self.warmup):
                self.optimizer.zero_grad(set_to_none=True)
                self.forward_backward(index)
                self.optimizer.step()
        for rows in self.splits.values():
            evaluate(self.task, self.model, rows, self.batch)
        with torch.no_grad():
            for parameter, value in zip(self.parameters, parameters, strict=True):
                parameter.copy_(value)
            for parameter, state in self.optimizer.state.items():
                for key, value in state.items():
                    if parameter in moments:
                        value.copy_(moments[parameter][key])
                    else:
                        value.zero_()
        self.optimizer.zero_grad(set_to_none=True)

    def forward_backward(self, index):
        """The operations of an update before the optimizer step."""
        output, targets = self.task.forward(self.model, self.rows.index_select(0, index))
        self.task.loss(output, targets).backward()
        norm = torch.nn.utils.get_total_norm([parameter.grad for parameter in self.parameters
                                              if parameter.grad is not None])
        return output, targets, norm

    def capture_update(self, size):
        """A graph of one update on `size` rows; backward allocates its gradients in the graph's pool."""
        index = torch.zeros(size, dtype=torch.long, device=self.device)
        graph = torch.cuda.CUDAGraph()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            _, _, norm = self.forward_backward(index)
            self.optimizer.step()
            self.trace.index_copy_(0, self.counter, norm.view(1))
            self.counter.add_(1)
        gradients = [parameter.grad for parameter in self.parameters]
        self.optimizer.zero_grad(set_to_none=True)
        return graph, index, gradients

    @torch.no_grad()
    def capture_evaluation(self, rows):
        sums = torch.zeros(self.task.sums, dtype=torch.float64, device=self.device)
        graph = torch.cuda.CUDAGraph()
        self.model.eval()
        with torch.cuda.graph(graph):
            sums.zero_()
            for start in range(0, len(rows), self.batch):
                self.task.accumulate(self.model, rows[start:start + self.batch], sums)
        return graph, sums

    def step(self, parts, rate, sampled):
        size = sum(count for _, _, count in parts)
        self.rate.fill_(rate)
        if sampled or size not in self.updates:
            return size, self.issue(parts, sampled)
        graph, index, _ = self.updates[size]
        self.stage(parts, index)
        if self.pending == len(self.trace):
            self.drain()
        graph.replay()
        self.pending += 1
        return size, None

    def issue(self, parts, sampled):
        """One update issued eagerly, with the operations a replay runs."""
        self.drain()
        index = gather(parts).to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output, targets, norm = self.forward_backward(index)
        if sampled:
            with self.clock.diagnosing():
                before = measure.before_update(self.model, self.task, output, targets)
        self.optimizer.step()
        self.norms.append(float(norm))
        if not sampled:
            return None
        with self.clock.diagnosing():
            return measure.after_update(self.model, self.optimizer, self.task, *before, self.norms[-1])

    def stage(self, parts, index):
        """Copy a batch's indices into a graph's static index, device to device."""
        offset = 0
        for permutation, start, count in parts:
            if permutation is not self.uploaded:
                self.upload(permutation)
            index[offset:offset + count].copy_(self.permutation[start:start + count])
            offset += count

    def upload(self, permutation):
        """Stage an epoch permutation; the pinned buffer is reused once its last copy has run."""
        self.staged.synchronize()
        self.pinned.copy_(permutation)
        self.permutation.copy_(self.pinned, non_blocking=True)
        self.staged.record()
        self.uploaded = permutation

    def drain(self):
        """Move the traced gradient norms of the replays since the last drain to the host."""
        if self.pending:
            self.norms.extend(self.trace[:self.pending].tolist())
            self.counter.zero_()
            self.pending = 0

    def gradient_norms(self):
        self.drain()
        norms, self.norms = self.norms, []
        return norms

    def evaluate(self, split):
        if split not in self.evaluations:
            return evaluate(self.task, self.model, self.splits[split], self.batch)
        graph, sums = self.evaluations[split]
        graph.replay()
        return self.task.metrics(sums.tolist(), len(self.splits[split]))

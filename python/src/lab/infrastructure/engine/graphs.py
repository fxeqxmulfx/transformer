"""Updates and evaluations replayed as captured CUDA graphs.

Each batch size that recurs (the full batch, and a short epoch tail) gets one
captured update: gather the batch through a static index, forward, backward,
the gradient norm and any clipping, the capturable optimizer step, and the
norm appended to a device trace. The rows of each observed split get one
captured evaluation over all their chunks, which every split sharing them
replays. Between replays the host only copies indices on the device and sets
the rate tensor; it waits for the device at observations and when it stages
a new index tensor of the sampler. Every evaluation, captured or not, is
static (`benchmarks`), so all of them compute alike.

Sampled updates, and batches of any other size, run the same operations
eagerly, so measurements see what a replay would leave. Only the capturable
optimizer can be captured, and its arithmetic differs from the native one in
the last bits: a CudaGraph run reproduces these operations issued eagerly,
not an Eager run.
"""

import math

import torch

from ...domain.optimizers import clipping
from ..benchmarks.samplers import gather
from ..optim import build_optimizer
from . import measure
from .loop import evaluate


class GraphStepper:
    warmup = 3

    def __init__(self, experiment, task, model, clock):
        self.task, self.model, self.clock = task, model, clock
        self.observed, self.batch, self.device = experiment.benchmark.observed, experiment.evaluate.batch, task.device
        self.clip = clipping(experiment.optimizer)
        self.rate = torch.zeros((), device=self.device)
        self.optimizer = build_optimizer(experiment.optimizer, model, self.rate, experiment.budget.updates,
                                         experiment.seeds.model)
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.trace = torch.zeros(experiment.evaluate.every, device=self.device)
        self.counter = torch.zeros(1, dtype=torch.long, device=self.device)
        self.pending, self.norms = 0, []
        self.updates, self.evaluations = {}, {}
        self.indices = self.pinned = self.uploaded = None
        self.staged = torch.cuda.Event()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
        for group in self.optimizer.param_groups:
            group["lr"] = self.rate

    def prepare(self, sizes):
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self.warm_up(sizes)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        for size in sizes:
            self.updates[size] = self.capture_update(size)
        for rows in self.observed_rows():
            self.evaluations[id(rows)] = self.capture_evaluation(rows)

    def observed_rows(self):
        """The rows of the observed splits, each once."""
        splits = self.task.splits
        return list({id(splits[name]): splits[name] for name in self.observed if splits[name] is not None}.values())

    def warm_up(self, sizes):
        """Run every graph's operations before capture, then restore what they changed.

        Warming up creates the optimizer state a capture must find in place;
        state that did not exist before is zeroed, which is how it starts.
        """
        parameters = [parameter.detach().clone() for parameter in self.parameters]
        moments = {parameter: {key: value.clone() for key, value in state.items()}
                   for parameter, state in self.optimizer.state.items()}
        for size in sizes:
            index = torch.arange(size, device=self.device)
            for _ in range(self.warmup):
                self.optimizer.zero_grad(set_to_none=True)
                self.forward_backward(index)
                self.optimizer.step()
        for rows in self.observed_rows():
            evaluate(self.task, self.model, rows, self.batch, static=True)
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
        output, targets = self.task.forward(self.model, self.task.inputs(index))
        self.task.loss(output, targets).backward()
        norm = torch.nn.utils.get_total_norm([parameter.grad for parameter in self.parameters
                                              if parameter.grad is not None])
        if math.isfinite(self.clip):
            torch.nn.utils.clip_grads_with_norm_(self.parameters, self.clip, norm)
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
        sums = self.task.accumulator()
        graph = torch.cuda.CUDAGraph()
        self.model.eval()
        with torch.cuda.graph(graph):
            sums.zero_()
            for start in range(0, len(rows), self.batch):
                self.task.accumulate(self.model, rows[start:start + self.batch], sums, static=True)
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
        for indices, start, count in parts:
            if indices is not self.uploaded:
                self.upload(indices)
            index[offset:offset + count].copy_(self.indices[start:start + count])
            offset += count

    def upload(self, indices):
        """Stage a sampler's index tensor; the pinned buffer is reused once its last copy has run."""
        self.staged.synchronize()
        if self.pinned is None or len(self.pinned) != len(indices):
            self.pinned = torch.empty(len(indices), dtype=torch.long, pin_memory=True)
            self.indices = torch.empty(len(indices), dtype=torch.long, device=self.device)
        self.pinned.copy_(indices)
        self.indices.copy_(self.pinned, non_blocking=True)
        self.staged.record()
        self.uploaded = indices

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
        rows = self.task.splits[split]
        if id(rows) not in self.evaluations:
            return evaluate(self.task, self.model, rows, self.batch, static=True)
        graph, sums = self.evaluations[id(rows)]
        graph.replay()
        return self.task.metrics(sums.tolist(), rows)

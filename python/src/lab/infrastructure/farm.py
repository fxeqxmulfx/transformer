"""Runs trained side by side, each in a process of its own on physical cores of its own.

A run takes `execution.threads` physical cores, and a CUDA run its device as
well, which no other run shares. A run's cores come from one package when
any could hold them, the one with the fewest free cores that can, so that
its memory stays local and wide runs still find room; only a run wider
than every package spans packages. A run holding a device comes first, so
that the device never idles, then the widest, so that narrower runs fill
in around it, equals in the order given (`order`); each starts as soon as
what it takes is free, a later one ahead of an earlier that must wait. A
run's process is `lab run <experiment> <label>`: it trains as it would
alone, on every logical CPU of its cores, with as many threads as it asks
for, and its output is passed on line by line.
"""

import os
from pathlib import Path
import selectors
import subprocess
import sys


def packages():
    """The physical cores this process may run on, each as its logical CPUs, by package."""
    allowed = sorted(os.sched_getaffinity(0))
    cores = {}
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((topology / "physical_package_id").read_text())
            core = int((topology / "core_id").read_text())
        except OSError:
            package, core = 0, cpu
        cores.setdefault((package, core), []).append(cpu)
    grouped = {}
    for (package, _), cpus in sorted(cores.items()):
        grouped.setdefault(package, []).append(tuple(cpus))
    return grouped


class Cores:
    """The free physical cores of a machine, by package."""

    def __init__(self, packages):
        self.free = {package: list(cores) for package, cores in packages.items()}
        self.home = {core: package for package, cores in packages.items() for core in cores}
        self.widest = max(len(cores) for cores in packages.values())

    def __len__(self):
        return len(self.home)

    def take(self, count):
        """`count` free cores, from one package if any could hold them; None while they are taken."""
        fitting = [package for package, cores in self.free.items() if len(cores) >= count]
        if fitting:
            package = min(fitting, key=lambda package: len(self.free[package]))
            taken, self.free[package] = self.free[package][:count], self.free[package][count:]
            return taken
        if count <= self.widest or sum(map(len, self.free.values())) < count:
            return None
        taken = []
        for package in sorted(self.free, key=lambda package: -len(self.free[package])):
            share = self.free[package][:count - len(taken)]
            self.free[package] = self.free[package][len(share):]
            taken += share
        return taken

    def give(self, cores):
        for core in cores:
            self.free[self.home[core]].append(core)
        for package in self.free:
            self.free[package].sort()


def device(execution):
    """The CUDA device a run takes for itself, or None."""
    name = execution.device
    if not name.startswith("cuda"):
        return None
    return name if ":" in name else name + ":0"


def order(runs):
    """The (label, execution) of `runs` in the order they start: those holding a device, then by width."""
    return sorted(runs, key=lambda run: (device(run[1]) is None, -run[1].threads))


class Process:
    """A run's `lab run` process, and the cores and device it holds."""

    def __init__(self, experiment, label, execution, cores):
        self.label, self.cores, self.device = label, cores, device(execution)
        cpus = {cpu for core in cores for cpu in core}
        environment = {**os.environ, "OMP_NUM_THREADS": str(execution.threads)}
        self.process = subprocess.Popen([sys.executable, "-m", "lab", "run", str(experiment), label],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
                                        preexec_fn=lambda: os.sched_setaffinity(0, cpus))
        self.pending = b""

    def lines(self, chunk):
        """The complete lines a chunk of output ends, the rest kept for later; all of it at the end."""
        text = self.pending + chunk
        if not chunk:
            self.pending = b""
            return [text] if text else []
        *complete, self.pending = text.split(b"\n")
        return complete


def train_apart(experiment, runs, echo):
    """Train each (label, execution) of `runs` in a process of its own; the labels whose process failed.

    `echo` receives every line a process writes, a label's own lines as
    they are and any other prefixed with the label.
    """
    machine = Cores(packages())
    for label, execution in runs:
        if execution.threads > len(machine):
            raise ValueError(f"{label} asks for {execution.threads} cores; the machine has {len(machine)}")
    pending, busy, failed = order(runs), set(), []
    selector = selectors.DefaultSelector()
    try:
        while pending or selector.get_map():
            for label, execution in list(pending):
                if device(execution) in busy:
                    continue
                cores = machine.take(execution.threads)
                if cores is None:
                    continue
                pending.remove((label, execution))
                process = Process(experiment, label, execution, cores)
                if process.device is not None:
                    busy.add(process.device)
                selector.register(process.process.stdout, selectors.EVENT_READ, process)
            for key, _ in selector.select():
                process = key.data
                chunk = os.read(key.fileobj.fileno(), 1 << 16)
                for line in process.lines(chunk):
                    text = line.decode(errors="replace")
                    echo(text if text.startswith(process.label + " ") else f"{process.label} | {text}")
                if chunk:
                    continue
                selector.unregister(key.fileobj)
                key.fileobj.close()
                if process.process.wait():
                    failed.append(process.label)
                busy.discard(process.device)
                machine.give(process.cores)
    finally:
        for key in list(selector.get_map().values()):
            key.data.process.terminate()
            key.data.process.wait()
    return failed

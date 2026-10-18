"""Modular arithmetic as regression: the table, its loss and metrics, and the landscape measurements.

Source: `lab.domain.regression.ModularRegression`, whose rows are those of
ModularDivision without its token format, and EXPERIMENT_PLAN.md,
"Scalar MLP on mod 97 (2026-10-18)", whose ways for gradient descent to fail
to fit the table the measurements tell apart: the norm of the full
gradient, the extreme eigenvalues of the Hessian of the loss, the norm of
the parameters, and the share of the residual's energy at high frequencies
of the discrete Fourier transform over Z/p x Z/p, the residual set to zero
at the absent y = 0. A limit of positive loss shows as a vanishing
gradient, and one that is no strict saddle as a smallest eigenvalue that is
not negative; a run too slow for its budget as a loss still falling at the
end (`analyze`), with its residual at high frequencies; an orbit leaving
every bounded set as a growing parameter norm. The extreme eigenvalues are
also the curvature along the run that the step is compared with, since the
gradient of the loss has no global Lipschitz constant.

The extreme eigenvalues are the extreme Ritz values of Lanczos iteration
with full reorthogonalization, on exact Hessian-vector products by double
backpropagation. They lie inside the spectrum, and reach its ends as the
iterations grow.
"""

import torch
from torch.nn import functional as F

from ...domain.analysis import decade_slope
from .samplers import EpochSampler

CUTOFFS = (5, 10, 20)
OBSERVED_ITERATIONS = 16
INSPECTED_ITERATIONS = 64


def answer(x, y, prime, operation):
    return (x + y) % prime if operation == "+" else x * pow(y, -1, prime) % prime


def flat(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def lanczos(product, size, iterations, generator):
    """The Ritz values, increasing, of a symmetric operator after `iterations` Lanczos steps.

    Each new vector is orthogonalized twice against every earlier one, in
    float64; the iteration ends early when the Krylov space closes.
    """
    device = generator.device
    basis = torch.zeros(iterations, size, dtype=torch.float64, device=device)
    vector = torch.randn(size, generator=generator, dtype=torch.float64, device=device)
    alphas, betas = [], []
    for index in range(iterations):
        vector = vector / vector.norm()
        basis[index] = vector
        image = product(vector.to(torch.float32)).to(torch.float64)
        alphas.append(torch.dot(image, vector).item())
        for _ in range(2):
            image -= basis[:index + 1].T @ (basis[:index + 1] @ image)
        beta = image.norm().item()
        if index + 1 == iterations or beta <= 1e-12 * max(map(abs, alphas)):
            break
        betas.append(beta)
        vector = image
    tridiagonal = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if betas:
        off = torch.tensor(betas, dtype=torch.float64)
        tridiagonal += torch.diag(off, 1) + torch.diag(off, -1)
    return torch.linalg.eigvalsh(tridiagonal).tolist()


def high_frequency_shares(residual, prime, cutoffs):
    """The share of a table's energy at the frequencies (k, l) with max(|k|, |l|) above each cutoff.

    `residual` holds the table at the rows (x, y), x major, y from 1; the
    table is zero at y = 0. |k| is the distance from k to 0 in Z/p, and the
    energy that of the two-dimensional discrete Fourier transform. None for
    a zero table.
    """
    grid = residual.new_zeros(prime, prime)
    grid[:, 1:] = residual.reshape(prime, prime - 1)
    energy = torch.fft.fft2(grid).abs().square()
    frequency = torch.arange(prime, device=residual.device)
    distance = torch.minimum(frequency, prime - frequency)
    level = torch.maximum(distance[:, None], distance[None, :])
    total = energy.sum().item()
    return {str(cutoff): energy[level > cutoff].sum().item() / total if total else None for cutoff in cutoffs}


class RegressionTask:
    """The whole table as the training split; a row is (x, y, x op y mod p)."""
    components = ("loss",)

    def __init__(self, spec, data_seed, device):
        self.spec, self.device, self.vocab = spec, device, spec.prime
        rows = [(x, y, answer(x, y, spec.prime, spec.operation))
                for x in range(spec.prime) for y in range(1, spec.prime)]
        self.splits = {"train": torch.tensor(rows, dtype=torch.long, device=device)}

    def sampler(self, batch, seed):
        return EpochSampler(len(self.splits["train"]), batch, False, seed)

    def inputs(self, indices, static=False):
        return self.splits["train"].index_select(0, indices.to(self.device))

    def progress(self, seen):
        return {"epochs_seen": seen / len(self.splits["train"])}

    def forward(self, model, batch, supervised=False):
        """The model's answers to the rows of `batch`, and their targets 2c / (p - 1) - 1."""
        return model(batch[:, :2]), batch[:, 2].to(torch.float32) * (2 / (self.vocab - 1)) - 1

    @staticmethod
    def loss(output, targets):
        return F.mse_loss(output, targets)

    @staticmethod
    def position_losses(output, targets):
        return F.mse_loss(output, targets).reshape(1)

    def accumulator(self):
        """Evaluation sums: the right answers and the squared errors."""
        return torch.zeros(2, dtype=torch.float64, device=self.device)

    def accumulate(self, model, chunk, sums, static):
        output, targets = self.forward(model, chunk)
        right = (torch.round((output + 1) * ((self.vocab - 1) / 2)) == chunk[:, 2]).sum()
        sums.add_(torch.stack([right.double(), (output - targets).square().sum().double()]))

    @staticmethod
    def metrics(sums, rows):
        right, squared = sums
        return {"accuracy": right / len(rows), "loss": squared / len(rows), "examples": len(rows)}

    def chunks(self, batch):
        rows = self.splits["train"]
        return [rows[start:start + batch] for start in range(0, len(rows), batch)]

    def gradient(self, model, parameters, batch, vector=None):
        """The gradient of the loss over the table; with `vector`, the Hessian times it."""
        total = torch.zeros(sum(parameter.numel() for parameter in parameters), device=self.device)
        for chunk in self.chunks(batch):
            output, targets = self.forward(model, chunk)
            loss = (output - targets).square().sum() / len(self.splits["train"])
            gradient = torch.autograd.grad(loss, parameters, create_graph=vector is not None)
            if vector is not None:
                gradient = torch.autograd.grad(flat(gradient), parameters, grad_outputs=vector)
            total += flat(gradient).detach()
        return total

    def landscape(self, model, batch, iterations):
        """The gradient and parameter norms, extreme Hessian Ritz values, and the residual's spectrum."""
        parameters = list(model.parameters())
        gradient = self.gradient(model, parameters, batch)
        generator = torch.Generator(self.device).manual_seed(0)
        ritz = lanczos(lambda vector: self.gradient(model, parameters, batch, vector), len(gradient), iterations,
                       generator)
        with torch.no_grad():
            outputs, targets = zip(*(self.forward(model, chunk) for chunk in self.chunks(batch)))
            residual = (torch.cat(outputs) - torch.cat(targets)).double()
        return {"gradient_l2": gradient.norm().item(), "parameter_l2": flat(parameters).norm().item(),
                "hessian_min": ritz[0], "hessian_max": ritz[-1], "lanczos_iterations": iterations,
                "residual_high": high_frequency_shares(residual, self.vocab, CUTOFFS)}

    def observe(self, model, batch):
        return {"landscape": self.landscape(model, batch, OBSERVED_ITERATIONS)}

    def inspect(self, model, batch):
        return {"landscape": self.landscape(model, batch, INSPECTED_ITERATIONS)}

    @staticmethod
    def analyze(history):
        """When the table is first fitted, and how fast the loss and the gradient still fall at the end."""
        return {"fitted_step": next((point["step"] for point in history if point["train"]["accuracy"] == 1), None),
                "best_accuracy": max(point["train"]["accuracy"] for point in history),
                "final_decade_slopes": {
                    "loss": decade_slope([(point["step"], point["train"]["loss"]) for point in history]),
                    "gradient_l2": decade_slope([(point["step"], point["landscape"]["gradient_l2"])
                                                 for point in history])}}

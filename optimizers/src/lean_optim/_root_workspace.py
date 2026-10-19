"""Buffered inverse-fourth-root kernels used by DASH during training.

The operation order is the one in roots.py. Matrix products always write
to a buffer distinct from both inputs; replacing them by aliased out= calls
would silently corrupt iterative roots. Functional helpers keep their usual
independent-return-value semantics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import config
from ._workspace import Workspace

if TYPE_CHECKING:
    from .config import InverseRoot


class InverseFourth:
    """Reuse the work matrices of one configured batched inverse root."""

    def __init__(self, spec: InverseRoot) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        self.spec = spec
        self.workspace = Workspace()
        self.bases = {}
        self.anchors = {}

    def _safe(self, value: Tensor, name: str) -> Tensor:
        mask = self.workspace.tensor(value, name, dtype=torch.bool)
        torch.gt(value, 0, out=mask)
        one = self.workspace.tensor(value, "one", ()).fill_(1)
        return torch.where(mask, value, one, out=value)

    def _scale(self, matrix: Tensor) -> Tensor:
        w = self.workspace
        batch = matrix.shape[:-2]
        work = w.like(matrix, "scale.work")
        frobenius = w.tensor(matrix, "scale.frobenius", batch)
        rows = w.tensor(matrix, "scale.rows", matrix.shape[:-1])
        upper = w.tensor(matrix, "scale.upper", batch)
        torch.square(matrix, out=work)
        torch.sum(work, dim=(-2, -1), out=frobenius)
        frobenius.sqrt_()
        torch.abs(matrix, out=work)
        torch.sum(work, dim=-1, out=rows)
        torch.amax(rows, dim=-1, out=upper)
        torch.minimum(frobenius, upper, out=upper)
        shape = (*matrix.shape[:-1], 1)
        vector = w.tensor(matrix, "power.vector", shape).fill_(1)
        next_vector = w.tensor(matrix, "power.next", shape)
        squared = w.tensor(matrix, "power.squared", shape)
        norm = w.tensor(matrix, "power.norm", (*batch, 1, 1))
        for _ in range(3):
            torch.matmul(matrix, vector, out=next_vector)
            torch.square(next_vector, out=squared)
            torch.sum(squared, dim=-2, keepdim=True, out=norm)
            norm.sqrt_()
            next_vector.div_(self._safe(norm, "positive"))
            vector, next_vector = next_vector, vector
        torch.matmul(matrix, vector, out=next_vector)
        next_vector.mul_(vector)
        estimate = w.tensor(matrix, "power.estimate", batch)
        denominator = w.tensor(matrix, "power.denominator", batch)
        torch.sum(next_vector, dim=(-2, -1), out=estimate)
        torch.square(vector, out=squared)
        torch.sum(squared, dim=(-2, -1), out=denominator)
        estimate.div_(self._safe(denominator, "positive"))
        credible = w.tensor(matrix, "scale.credible", batch, torch.bool)
        test = w.tensor(matrix, "scale.test", batch, torch.bool)
        torch.gt(estimate, 0, out=credible)
        estimate.mul_(2)
        torch.lt(upper, estimate, out=test)
        torch.logical_and(credible, test, out=credible)
        torch.gt(upper, 0, out=test)
        upper.mul_(2)
        one = w.tensor(upper, "one", ()).fill_(1)
        torch.where(test, upper, one, out=upper)
        return torch.where(credible, estimate, upper, out=upper)

    def _regularized(self, history: Tensor, eps: float) -> Tensor:
        w = self.workspace
        identity = w.identity(history)
        diagonal = w.like(identity, "damping")
        torch.mul(identity, eps, out=diagonal)
        return torch.add(history, diagonal, out=history)

    def _ndb(self, matrix: Tensor, steps: int, reference: Tensor) -> tuple[Tensor, Tensor]:
        w = self.workspace
        identity = w.identity(reference)
        one_half = w.like(identity, "ndb.identity")
        torch.mul(identity, 1.5, out=one_half)
        correction = w.like(reference, "scale.work")
        root = reference
        next_root = w.like(reference, "ndb.next_root")
        if matrix is root:
            root, next_root = next_root, root
        inverse = w.like(reference, "ndb.inverse")
        next_inverse = w.like(reference, "ndb.next_inverse")
        torch.mul(matrix, 0.5, out=correction)
        torch.sub(one_half, correction, out=correction)
        torch.matmul(matrix, correction, out=root)
        inverse.copy_(correction)
        for _ in range(1, steps):
            torch.matmul(inverse, root, out=correction)
            correction.mul_(0.5)
            torch.sub(one_half, correction, out=correction)
            torch.matmul(root, correction, out=next_root)
            torch.matmul(correction, inverse, out=next_inverse)
            root, next_root = next_root, root
            inverse, next_inverse = next_inverse, inverse
        return root, inverse

    def _cn(self, matrix: Tensor, steps: int) -> Tensor:
        w = self.workspace
        identity = w.identity(matrix)
        scaled_identity = w.like(identity, "cn.identity")
        torch.mul(identity, 1.25, out=scaled_identity)
        unit = 5 ** (-0.25)
        x = w.like(matrix, "cn.x")
        x.copy_(identity).div_(unit)
        residual = matrix.div_(unit**4)
        next_x = w.like(matrix, "cn.next_x")
        next_residual = w.like(matrix, "cn.next_residual")
        correction = w.like(matrix, "scale.work")
        squared = w.like(matrix, "cn.squared")
        fourth = correction
        for _ in range(steps):
            torch.mul(residual, 0.25, out=correction)
            torch.sub(scaled_identity, correction, out=correction)
            torch.matmul(x, correction, out=next_x)
            torch.matmul(correction, correction, out=squared)
            torch.matmul(squared, squared, out=fourth)
            torch.matmul(fourth, residual, out=next_residual)
            x, next_x = next_x, x
            residual, next_residual = next_residual, residual
        return x

    def _basis(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        key = (history.device, history.dtype)
        if key not in self.bases:
            samples = max(self.spec.samples, self.spec.degree + 1)
            angles = 2 * torch.arange(samples, dtype=history.dtype, device=history.device) + 1
            angles = angles * (math.pi / (2 * samples))
            modes = torch.arange(self.spec.degree + 1, dtype=history.dtype, device=history.device)
            nodes = angles.cos()
            self.bases[key] = nodes, (angles[:, None] * modes).cos(), (nodes + 1) / 2
        return self.bases[key]

    def _chebyshev(self, history: Tensor, eps: float) -> Tensor:
        w, degree = self.workspace, self.spec.degree
        scale = self._scale(history)
        identity = w.identity(history)
        argument = torch.mul(history, 2, out=w.like(history, "cheb.argument"))
        argument.div_(scale[..., None, None]).sub_(identity)
        nodes, basis, shifted_nodes = self._basis(history)
        values = w.tensor(history, "cheb.values", (*history.shape[:-2], nodes.numel()))
        offset = w.tensor(history, "cheb.offset", (*history.shape[:-2], 1))
        # Tensor.__rtruediv__ evaluates reciprocal * scalar, rather than div.
        torch.reciprocal(scale[..., None], out=offset)
        offset.mul_(eps)
        torch.add(offset, shifted_nodes, out=values)
        values.pow_(-0.25)
        coefficients = w.tensor(history, "cheb.coefficients", (*history.shape[:-2], degree + 1))
        torch.matmul(values, basis, out=coefficients)
        coefficients.mul_(2 / nodes.numel())
        coefficients[..., 0].mul_(0.5)
        output = history
        work = w.like(history, "scale.work")
        first = w.like(history, "cheb.first")
        second = w.like(history, "cheb.second")
        if degree == 0:
            torch.mul(coefficients[..., 0, None, None], identity, out=output)
        elif degree == 1:
            torch.mul(coefficients[..., 1, None, None], argument, out=output)
            torch.mul(coefficients[..., 0, None, None], identity, out=work)
            output.add_(work)
        else:
            torch.mul(coefficients[..., degree, None, None], identity, out=second)
            twice = w.tensor(history, "cheb.twice", (*history.shape[:-2], 1, 1))
            torch.mul(coefficients[..., degree, None, None], 2, out=twice)
            torch.mul(twice, argument, out=first)
            torch.mul(coefficients[..., degree - 1, None, None], identity, out=work)
            first.add_(work)
            for k in range(degree - 2, 0, -1):
                torch.matmul(argument, first, out=output)
                output.mul_(2).sub_(second)
                torch.mul(coefficients[..., k, None, None], identity, out=work)
                output.add_(work)
                second, first, output = first, output, second
            torch.matmul(argument, first, out=output)
            output.sub_(second)
            torch.mul(coefficients[..., 0, None, None], identity, out=work)
            output.add_(work)
        scale.pow_(-0.25)
        return output.mul_(scale[..., None, None])

    def __call__(self, history: Tensor, eps: float) -> Tensor:
        """Return a borrowed root buffer, overwritten by the next call of this shape."""
        w = self.workspace
        key = (history.shape, history.dtype, history.device)
        anchor = self.anchors.get(key)
        if anchor is None:
            anchor = self.anchors[key] = torch.empty_like(history)
        # DASH consumes a left root in its first product before requesting the
        # right root. Shapes can therefore share the entire solver workspace.
        anchor.copy_(history)
        if isinstance(self.spec, config.Chebyshev):
            return self._chebyshev(anchor, eps)
        matrix = self._regularized(anchor, eps)
        if isinstance(self.spec, config.EVD):
            values = w.tensor(matrix, "eigen.values", matrix.shape[:-1])
            vectors = w.like(matrix, "eigen.vectors")
            torch.linalg.eigh(matrix, out=(values, vectors))
            values.clamp_min_(torch.finfo(matrix.dtype).tiny).pow_(-0.25)
            scaled = torch.mul(vectors, values.unsqueeze(-2), out=w.like(matrix, "eigen.scaled"))
            return torch.matmul(scaled, vectors.transpose(-2, -1), out=matrix)
        scale = self._scale(matrix)
        normalized = matrix.div_(scale[..., None, None])
        if isinstance(self.spec, config.NewtonDB):
            first_root, _ = self._ndb(normalized, self.spec.steps, normalized)
            _, output = self._ndb(first_root, self.spec.steps, normalized)
        else:
            output = self._cn(normalized, self.spec.steps)
        scale.pow_(-0.25)
        return output.mul_(scale[..., None, None])

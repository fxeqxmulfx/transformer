"""Column generation for the genuine sparsemax matching family.

Source: matchingMixture_criterion_convex, matchingMixture_gap_certificate
and matchingHeadPrice_minimum_exists at 83d985f. OrderPricing and BindingPricing
are analytic global oracles only on their two formal contexts. Paired heads
use pairedMixture_criterion_convex and pairedBinding_fit. SearchPricing uses a local
Q/K optimizer, and never reports its found price as a global lower bound.
Values are optimized in their original table, not recovered by an inverse.
"""

from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._types import HeadFactory, MatchingHead, MatchingModel, MatchingTask, Report, StateDict


_SIMPLEX_NEGATIVE_TOLERANCE = 1e-10
_SIMPLEX_SUM_TOLERANCE = 1e-8
_EXACT_MAX_HEADS = 3


def simplex_project(weights: Tensor) -> Tensor:
    """Euclidean simplex projection, sparsemax Eq. (1), in storage coordinates."""
    ordered = weights.sort(descending=True).values
    ranks = torch.arange(1, len(weights) + 1, device=weights.device, dtype=weights.dtype)
    cumulative = ordered.cumsum(0)
    count = int((1 + ranks * ordered > cumulative).sum())
    threshold = (cumulative[count - 1] - 1) / count
    return (weights - threshold).clamp_min(0)


def order_price(
    gradient: Tensor, final_tokens: Tensor, reference: MatchingHead, head_factory: HeadFactory
) -> MatchingHead:
    """Attain -sum(abs(g)) over all cap-one scalar physical order heads."""
    q = torch.zeros_like(reference.q[0])
    q[final_tokens, 0] = -gradient.flatten().sign().to(q.dtype)
    k = q.new_tensor([[-0.5], [0.5]])
    values = q.new_tensor([[-1.0], [1.0]])
    return head_factory(q, k, values)


def binding_price(
    gradient: Tensor, written_values: Tensor, reference: MatchingHead, head_factory: HeadFactory
) -> MatchingHead:
    """Select the original value following key one; attain the full box price bound."""
    q, k, values = (torch.zeros_like(t[0]) for t in (reference.q, reference.k, reference.values))
    channel = q.shape[-1] // 2
    q[1, channel] = k[1, channel] = 1
    values[written_values, 0] = -gradient.flatten().sign().to(values.dtype)
    return head_factory(q, k, values, forward_map=reference.forward_map)


def simplex_squared_fit(columns: Tensor, targets: Tensor) -> Tensor:
    """Solve either scalar example's <=3-column convex QP, including singular faces."""
    size = len(columns)
    matrix, target = columns.flatten(1).T.double(), targets.flatten().double()
    best_loss, best = float("inf"), None
    for count in range(1, size + 1):
        for face in combinations(range(size), count):
            selected = matrix[:, face]
            system = torch.zeros(count + 1, count + 1, dtype=matrix.dtype, device=matrix.device)
            system[:count, :count] = 2 * selected.T @ selected
            system[count, :count] = system[:count, count] = 1
            rhs = torch.cat((2 * selected.T @ target, matrix.new_ones(1)))
            solution = torch.linalg.lstsq(system, rhs, driver="gelsd").solution[:count]
            if (
                float(solution.min()) < -_SIMPLEX_NEGATIVE_TOLERANCE
                or abs(float(solution.sum()) - 1) > _SIMPLEX_SUM_TOLERANCE
            ):
                continue
            solution = solution.clamp_min(0)
            solution = solution / solution.sum()
            mass = matrix.new_zeros(size)
            mass[list(face)] = solution
            loss = float((matrix @ mass - target).square().sum())
            if loss < best_loss:
                best_loss, best = loss, mass
    if best is None:
        msg = "No feasible face in the scalar simplex QP"
        raise ArithmeticError(msg)
    return best.to(columns.dtype)


class AtomicColumns:
    """Column generation over a physical sparsemax matching family.

    model supplies q/k/values tables with a leading atom axis, simplex mass,
    active count, spec.cap/spec.heads, forward_map, append(head) and prune().
    task supplies tokens, forward(model,batch,supervised=True) -> (output,
    targets) and a convex output loss(output,targets). head_factory constructs
    the family's actual head from original q/k/values and forward_map.

    The exact order/binding modes require their two formal scalar cap-one
    contexts; arbitrary families use search. The factory/task must implement
    those physical maps; a callable's semantic correctness is not inferred
    from Python duck typing. Local search never asserts a global found price.
    """

    def __init__(  # noqa: PLR0913 - Preserve the established optimizer/recurrence signature.
        self,
        model: MatchingModel,
        task: MatchingTask,
        *,
        head_factory: HeadFactory,
        pricing: str = "search",
        seed: int = 0,
        correction_steps: int = 100,
        tolerance: float = 1e-8,
        restarts: int = 4,
        steps: int = 100,
        lr: float = 0.05,
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        if pricing not in ("order", "binding", "search"):
            msg = "pricing must be order, binding or search"
            raise ValueError(msg)
        if correction_steps < 1 or tolerance <= 0 or restarts < 1 or steps < 1 or lr <= 0:
            msg = "Atomic solver budgets, tolerance and search rate must be positive"
            raise ValueError(msg)
        self.head_factory = head_factory
        self.spec = SimpleNamespace(
            pricing=SimpleNamespace(kind=pricing, restarts=restarts, steps=steps, lr=lr),
            correction_steps=correction_steps,
            tolerance=tolerance,
        )
        self.model, self.task = model, task
        if pricing != "search":
            if (
                model.spec.cap != 1
                or model.spec.heads > _EXACT_MAX_HEADS
                or model.values.shape[-1] != 1
            ):
                msg = "Exact scalar pricing needs cap one, scalar values and at most three atoms"
                raise ValueError(msg)
            if model.q.device.type != "cpu":
                msg = "Exact scalar face correction requires CPU linear algebra"
                raise ValueError(msg)
            contexts = (
                [[0, 1], [1, 0]] if pricing == "order" else [[0, 1, 3, 2, 4, 1], [0, 1, 4, 2, 3, 1]]
            )
            if task.tokens.tolist() != contexts:
                msg = "Exact pricing requires its two formal matching contexts"
                raise ValueError(msg)
            if (pricing == "order" and model.q.shape[-1] != 1) or (
                pricing == "binding" and model.q.shape[-1] % 2
            ):
                msg = "Exact pricing needs scalar order heads or even-width paired heads"
                raise ValueError(msg)
        self.generator = torch.Generator().manual_seed(seed + 30_000)
        self.updates, self.oracle_calls, self.search_evaluations = 0, 0, 0
        self.latest = {}

    def state_dict(self) -> StateDict:
        """Return the random generator, counters and latest solver diagnostics."""
        return {
            "generator": self.generator.get_state(),
            "updates": self.updates,
            "oracle_calls": self.oracle_calls,
            "search_evaluations": self.search_evaluations,
            "latest": self.latest,
        }

    def load_state_dict(self, state: StateDict) -> None:
        """Restore mathematical state and discard temporary work buffers."""
        self.generator.set_state(state["generator"].cpu())
        for key in ("updates", "oracle_calls", "search_evaluations", "latest"):
            setattr(self, key, state[key])

    def report(self) -> Report:
        """Return the recorded optimizer diagnostics."""
        return {
            "updates": self.updates,
            "oracle_calls": self.oracle_calls,
            "search_evaluations": self.search_evaluations,
            **self.latest,
        }

    def response(self, head: MatchingHead, batch: Tensor) -> Tensor:
        """Evaluate one physical matching head on the selected observations."""
        return self.task.forward(head, batch, supervised=True)[0]

    def output_gradient(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor, float]:
        """Evaluate outputs, targets and the convex loss gradient in output space."""
        with torch.no_grad():
            output, targets = self.task.forward(self.model, batch, supervised=True)
        variable = output.detach().requires_grad_()
        loss = self.task.loss(variable, targets)
        return output, targets, torch.autograd.grad(loss, variable)[0], float(loss.detach())

    def price(self, gradient: Tensor, batch: Tensor) -> MatchingHead:
        """Choose a physical head with exact scalar pricing or local search."""
        self.oracle_calls += 1
        if self.spec.pricing.kind == "order":
            tokens = self.task.tokens[batch][:, -1]
            return order_price(gradient, tokens, self.model, self.head_factory)
        if self.spec.pricing.kind == "binding":
            return binding_price(
                gradient, self.task.tokens[batch][:, 2], self.model, self.head_factory
            )
        return self.search_price(gradient, batch)

    def search_price(self, gradient: Tensor, batch: Tensor) -> MatchingHead:
        """Search bounded original Q/K tables for a low-price physical head."""
        spec, model = self.spec.pricing, self.model
        cap, best_price, best = model.spec.cap, float("inf"), None
        for restart in range(spec.restarts):
            scale = min(cap, 0.25) if restart % 2 == 0 else cap
            tables = [
                (torch.rand(t[0].shape, generator=self.generator, dtype=t.dtype) * 2 - 1).to(
                    t.device
                )
                * scale
                for t in (model.q, model.k)
            ]
            head = self.head_factory(
                *tables, torch.zeros_like(model.values[0]), forward_map=model.forward_map
            )
            optimizer = torch.optim.Adam((head.q, head.k), lr=spec.lr)
            for step in range(spec.steps + 1):
                # The price is linear in original values at every fixed Q/K.
                with torch.no_grad():
                    head.values.zero_()
                zero_price = (gradient * self.response(head, batch)).sum()
                coefficients = torch.autograd.grad(zero_price, head.values)[0]
                with torch.no_grad():
                    head.values.copy_(-cap * coefficients.sign())
                optimizer.zero_grad(set_to_none=True)
                price = (gradient * self.response(head, batch)).sum()
                self.search_evaluations += 2
                if float(price.detach()) < best_price:
                    best_price = float(price.detach())
                    best = self.head_factory(
                        head.q.detach(),
                        head.k.detach(),
                        head.values.detach(),
                        forward_map=model.forward_map,
                    )
                if step < spec.steps:
                    price.backward()
                    optimizer.step()
                    with torch.no_grad():
                        head.q.clamp_(-cap, cap)
                        head.k.clamp_(-cap, cap)
        return best

    @torch.no_grad()
    def columns(self, batch: Tensor) -> Tensor:
        """Evaluate every active physical head on the selected observations."""
        model = self.model
        return torch.stack(
            [
                self.response(
                    self.head_factory(
                        model.q[i], model.k[i], model.values[i], forward_map=model.forward_map
                    ),
                    batch,
                )
                for i in range(int(model.active))
            ]
        )

    def correct(self, batch: Tensor, targets: Tensor) -> None:
        """Refit the active simplex weights using the configured convex correction."""
        columns = self.columns(batch)
        count = len(columns)
        if self.spec.pricing.kind in ("order", "binding"):
            mass = simplex_squared_fit(columns, targets)
        else:
            mass = self.model.mass[:count].detach().clone()
            step_size = 1.0
            for _ in range(self.spec.correction_steps):
                variable = mass.detach().requires_grad_()
                prediction = torch.tensordot(variable, columns, dims=([0], [0]))
                loss = self.task.loss(prediction, targets)
                gradient = torch.autograd.grad(loss, variable)[0]
                accepted = False
                for _ in range(30):
                    proposal = simplex_project(mass - step_size * gradient)
                    new_loss = self.task.loss(
                        torch.tensordot(proposal, columns, dims=([0], [0])), targets
                    )
                    descent = torch.dot(gradient, proposal - mass)
                    if float(new_loss) <= float(loss.detach() + 0.01 * descent):
                        accepted = True
                        break
                    step_size *= 0.5
                if not accepted:
                    break
                displacement = float((proposal - mass).abs().max())
                mass = proposal
                if displacement <= self.spec.tolerance:
                    break
                step_size = min(2 * step_size, 1e6)
        with torch.no_grad():
            self.model.mass.zero_()
            self.model.mass[:count].copy_(mass)
            self.model.prune()

    def step(self, batch: Tensor) -> float:
        """Price a new column, correct the mixture and record its certificate."""
        if self.spec.pricing.kind != "search" and sorted(batch.tolist()) != [0, 1]:
            msg = "Exact scalar pricing requires both distinct observations in every update"
            raise ValueError(msg)
        model = self.model
        output, targets, gradient, before = self.output_gradient(batch)
        head = self.price(gradient, batch)
        with torch.no_grad():
            current_price = float((gradient * output).sum())
            found_price = float((gradient * self.response(head, batch)).sum())
        gap = current_price - found_price
        full = int(model.active) == model.spec.heads
        if gap > self.spec.tolerance and not full:
            model.append(head)
        self.correct(batch, targets)
        after_output, _, after_gradient, after = self.output_gradient(batch)
        pairing = float((after_output * after_gradient).sum())
        # All genuine heads have each output in [-cap, cap], regardless of Q/K.
        lower = -model.spec.cap * float(after_gradient.abs().sum())
        exact = self.spec.pricing.kind in ("order", "binding")
        if exact:
            if self.spec.pricing.kind == "order":
                final_head = order_price(
                    after_gradient, self.task.tokens[batch][:, -1], model, self.head_factory
                )
            else:
                final_head = binding_price(
                    after_gradient, self.task.tokens[batch][:, 2], model, self.head_factory
                )
            with torch.no_grad():
                lower = float((after_gradient * self.response(final_head, batch)).sum())
        self.updates += 1
        self.latest = {
            "before_loss": before,
            "after_loss": after,
            "found_head_gap_before": gap,
            "global_gap_bound_after": max(0.0, pairing - lower),
            "pricing_is_global": exact,
            "certificate_scope": "current_training_batch",
            "capacity_full": full,
            "active_heads": int(model.active),
        }
        if not torch.isfinite(after_output).all() or not torch.isfinite(model.mass).all():
            msg = "Nonfinite atomic mixture"
            raise FloatingPointError(msg)
        return after

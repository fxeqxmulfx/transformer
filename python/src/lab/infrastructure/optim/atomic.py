"""Column generation for the genuine sparsemax matching family.

Source: matchingMixture_criterion_convex, matchingMixture_gap_certificate
and matchingHeadPrice_minimum_exists at 83d985f. OrderPricing is an analytic
global oracle only on the two formal contexts. SearchPricing uses a local
Q/K optimizer, and never reports its found price as a global lower bound.
Values are optimized in their original table, not recovered by an inverse.
"""

from itertools import combinations

import torch

from ...domain.atomic import OrderPricing
from ..nn.atomic import MatchingHead


def simplex_project(weights):
    """Euclidean simplex projection, sparsemax Eq. (1), in storage coordinates."""
    ordered = weights.sort(descending=True).values
    ranks = torch.arange(1, len(weights) + 1, device=weights.device, dtype=weights.dtype)
    cumulative = ordered.cumsum(0)
    count = int((1 + ranks * ordered > cumulative).sum())
    threshold = (cumulative[count - 1] - 1) / count
    return (weights - threshold).clamp_min(0)


def order_price(gradient, final_tokens, reference):
    """Attain -sum(abs(g)) over all cap-one scalar physical order heads."""
    q = torch.zeros_like(reference.q[0])
    q[final_tokens, 0] = -gradient.flatten().sign().to(q.dtype)
    k = q.new_tensor([[-0.5], [0.5]])
    values = q.new_tensor([[-1.0], [1.0]])
    return MatchingHead(q, k, values)


def simplex_squared_fit(columns, targets):
    """Solve the order example's <=3-column convex QP, including singular faces."""
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
            if float(solution.min()) < -1e-10 or abs(float(solution.sum()) - 1) > 1e-8:
                continue
            solution = solution.clamp_min(0)
            solution = solution / solution.sum()
            mass = matrix.new_zeros(size)
            mass[list(face)] = solution
            loss = float((matrix @ mass - target).square().sum())
            if loss < best_loss:
                best_loss, best = loss, mass
    if best is None:
        raise ArithmeticError("No feasible face in the order simplex QP")
    return best.to(columns.dtype)


class AtomicColumns:
    def __init__(self, spec, model, task, seed):
        self.spec, self.model, self.task = spec, model, task
        self.generator = torch.Generator().manual_seed(seed + 30_000)
        self.updates, self.oracle_calls, self.search_evaluations = 0, 0, 0
        self.latest = {}

    def state_dict(self):
        return {"generator": self.generator.get_state(), "updates": self.updates,
                "oracle_calls": self.oracle_calls, "search_evaluations": self.search_evaluations,
                "latest": self.latest}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"].cpu())
        for key in ("updates", "oracle_calls", "search_evaluations", "latest"):
            setattr(self, key, state[key])

    def report(self):
        return {"updates": self.updates, "oracle_calls": self.oracle_calls,
                "search_evaluations": self.search_evaluations, **self.latest}

    def response(self, head, batch):
        return self.task.forward(head, batch, supervised=True)[0]

    def output_gradient(self, batch):
        with torch.no_grad():
            output, targets = self.task.forward(self.model, batch, supervised=True)
        variable = output.detach().requires_grad_()
        loss = self.task.loss(variable, targets)
        return output, targets, torch.autograd.grad(loss, variable)[0], float(loss.detach())

    def price(self, gradient, batch):
        self.oracle_calls += 1
        if isinstance(self.spec.pricing, OrderPricing):
            tokens = self.task.tokens[batch][:, -1]
            return order_price(gradient, tokens, self.model)
        return self.search_price(gradient, batch)

    def search_price(self, gradient, batch):
        spec, model = self.spec.pricing, self.model
        cap, best_price, best = model.spec.cap, float("inf"), None
        for restart in range(spec.restarts):
            scale = min(cap, 0.25) if restart % 2 == 0 else cap
            tables = [(torch.rand(t[0].shape, generator=self.generator, dtype=t.dtype) * 2 - 1)
                      .to(t.device) * scale for t in (model.q, model.k)]
            head = MatchingHead(*tables, torch.zeros_like(model.values[0]))
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
                    best = MatchingHead(head.q.detach(), head.k.detach(), head.values.detach())
                if step < spec.steps:
                    price.backward()
                    optimizer.step()
                    with torch.no_grad():
                        head.q.clamp_(-cap, cap)
                        head.k.clamp_(-cap, cap)
        return best

    @torch.no_grad()
    def columns(self, batch):
        model = self.model
        return torch.stack([self.response(MatchingHead(model.q[i], model.k[i], model.values[i]), batch)
                            for i in range(int(model.active))])

    def correct(self, batch, targets):
        columns = self.columns(batch)
        count = len(columns)
        if isinstance(self.spec.pricing, OrderPricing):
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
                    new_loss = self.task.loss(torch.tensordot(proposal, columns, dims=([0], [0])), targets)
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

    def step(self, batch):
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
        exact = isinstance(self.spec.pricing, OrderPricing)
        if exact:
            final_head = order_price(after_gradient, self.task.tokens[batch][:, -1], model)
            with torch.no_grad():
                lower = float((after_gradient * self.response(final_head, batch)).sum())
        self.updates += 1
        self.latest = {"before_loss": before, "after_loss": after, "found_head_gap_before": gap,
                       "global_gap_bound_after": max(0.0, pairing - lower),
                       "pricing_is_global": exact, "certificate_scope": "current_training_batch",
                       "capacity_full": full, "active_heads": int(model.active)}
        if not torch.isfinite(after_output).all() or not torch.isfinite(model.mass).all():
            raise FloatingPointError("Nonfinite atomic mixture")
        return after

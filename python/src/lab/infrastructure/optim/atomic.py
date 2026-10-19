"""Lab adapters for the standalone atomic matching solver."""

from lean_optim.atomic import AtomicColumns as Solver, simplex_project, simplex_squared_fit
from lean_optim.atomic import order_price as _order_price, binding_price as _binding_price

from ..nn.atomic import MatchingHead


def order_price(gradient, final_tokens, reference):
    return _order_price(gradient, final_tokens, reference, MatchingHead)


def binding_price(gradient, written_values, reference):
    return _binding_price(gradient, written_values, reference, MatchingHead)


class AtomicColumns(Solver):
    def __init__(self, spec, model, task, seed):
        pricing = type(spec.pricing).__name__.removesuffix("Pricing").lower()
        search = ({name: getattr(spec.pricing, name) for name in ("restarts", "steps", "lr")}
                  if pricing == "search" else {})
        super().__init__(model, task, head_factory=MatchingHead, pricing=pricing, seed=seed,
                         correction_steps=spec.correction_steps, tolerance=spec.tolerance, **search)

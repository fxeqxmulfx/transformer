"""Task161: exact first-step width dependence for the actual Schedule-Free gate.

Source: ResidualGain and ResidualScheduleFreeAdamW at18112982. Two opposite,
RMS-one fixed feature vectors have targets1.5 and0.5, with the finite solution
w_i=log(3)/width. This isolates aligned-coordinate scaling; real transformer
features need not align, so this is not a causal diagnosis of their trajectories.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from lab.infrastructure.nn.adaptive_residual import ResidualGain
from lab.infrastructure.optim import evaluation_weights
from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW


def main():
    torch.set_num_threads(1)
    cases = []
    for width in (16, 64, 128, 512, 1024):
        for rate in (.001, .01):
            model = ResidualGain(width, 1, bias=False).double()
            x = torch.stack((torch.ones(width), -torch.ones(width))).double()
            targets = torch.tensor([[1.5], [.5]], dtype=torch.float64)
            optimizer = ResidualScheduleFreeAdamW(model.parameters(), lr=rate, betas=(.9, .98), weight_decay=.1)
            loss = .5 * (model(x) - targets).square().mean()
            loss.backward()
            torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, -.25), rtol=0, atol=0)
            optimizer.step()
            # The first Schedule-Free averaging weight is1; decay at zero is0.
            expected_coordinate = rate * .25 / (.25 + optimizer.param_groups[0]["eps"])
            expected_logit = width * expected_coordinate
            torch.testing.assert_close(model.weight, torch.full_like(model.weight, expected_coordinate), rtol=1e-12, atol=1e-15)
            with evaluation_weights(optimizer), torch.no_grad():
                prediction = model(x)
                actual_logit = torch.nn.functional.linear(x, model.weight)[0, 0].item()
                assert math.isclose(actual_logit, expected_logit, rel_tol=1e-12, abs_tol=1e-14)
                gain = prediction[0, 0].item()
                cases.append({"width": width, "rate": rate, "first_logit": actual_logit,
                              "analytic_first_logit": expected_logit, "gain": gain,
                              "derivative_fraction_of_maximum": gain * (2 - gain),
                              "initial_loss": loss.item(), "first_loss": (.5 * (prediction - targets).square().mean()).item()})
                model.weight.fill_(math.log(3) / width)
                torch.testing.assert_close(model(x), targets, rtol=1e-12, atol=1e-12)
    result = {"scope": "Independent analytic oracle, FP64 actual Schedule-Free first step, opposite aligned fixed features. No transformer training or final-test predictions.",
              "formula": "At w=0, each gradient coordinate=-1/4. Thus z_after=width*rate*(1/4)/(1/4+eps). Finite exact solution w_i=log(3)/width.",
              "limitation": "This exhibits width-dependent steps in a realizable special case; it neither proves a defect nor explains the measured depth regression. Actual depth uses rate0.001, whose first step here is not saturated at width64 or128.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": cases}
    folder = Path(__file__).resolve().parents[1] / "runs/gain_transfer"
    folder.mkdir(exist_ok=True)
    path = folder / "width_oracle.json"
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

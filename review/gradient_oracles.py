"""Task151 post-run independent Decimal/FP64/FP32 derivative and spacing audit."""

from decimal import Decimal, localcontext
import json
from pathlib import Path

import torch

from lab.dsl import BranchGates, LinearTransport
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer
from lab.dsl import ScheduleFreeAdamW

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
rows = []
for scale in (.001, 1., 1000.):
    with localcontext() as ctx:
        ctx.prec = 80
        a, eps = Decimal(str(scale)), Decimal(2)**-23
        r = (a*a+eps).sqrt()
        expected = {'before': a*eps/(2*r**3), 'after': a/(2*r)}
        ratio = eps/(a*a+eps)
        optima = []
        for c in (Decimal(1)/4, Decimal(1)/2, Decimal(3)/4):
            before = eps.sqrt()*c/(a*(1-c*c).sqrt())
            after = c*r/a
            assert 0 < before < 2 and 0 < after < 2
            optima.append({'target':str(c),'before':str(before),'after':str(after)})
    for placement in ('before', 'after'):
        row = {'scale':scale,'placement':placement,'decimal_logit_derivative':str(expected[placement]),
               'decimal_before_after_ratio':str(ratio),'exact_optimal_gates':optima}
        for dtype in (torch.float32, torch.float64):
            model = build_model(BranchGates(placement=placement,scale=scale),0,0).to(dtype)
            output = model(torch.ones(1,2,32,dtype=dtype))[0,0]
            gradient, = torch.autograd.grad(output,model.logits)
            row[str(dtype)] = {'amplitude':float(output.detach()-1),'logit_derivative':gradient.item()}
        rows.append(row)
spec = LinearTransport(depth=32)
model = build_model(spec,0,0)
optimizer = build_optimizer(ScheduleFreeAdamW(.01,betas=(.9,.98)),model)
optimizer.param_groups[0]['lr']=.01
x=torch.ones(64,1,32)
before=model.weight.detach().clone()
(model(x)-.25).square().mean().backward()
gradient=model.weight.grad.item()
optimizer.step()
transport={'depth':32,'initial_parameter':before.item(),'parameter_after_one_update':model.weight.item(),
           'gradient':gradient,'spacing_toward_zero':float(before-torch.nextafter(before,torch.zeros_like(before))),
           'real_first_base_step_magnitude':.01*gradient/(abs(gradient)+1e-8),
           'parameter_unchanged':torch.equal(before,model.weight)}
(ROOT/'experiments/residual_gradient_paths/oracles.json').write_text(json.dumps({'radial':rows,'transport':transport},indent=2)+'\n')
print(json.dumps(transport))

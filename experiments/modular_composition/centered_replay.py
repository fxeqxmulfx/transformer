"""Apply diff_gpt's centered XSA to diagnostic copies of the frozen lab model.

This does not alter the lab training implementation or any checkpoint. Q/K
modules retain their explicit epsilons when a diagnostic copy becomes FP64.
The companion centered_projection implementation is independently tested in
diff_gpt and corresponds to Attention.CenteredXSA's real equalities.
"""

from types import MethodType

import torch
from torch.nn import functional as F

from diff_gpt.domain.model.centered_xsa import XSA_EPS, centered_projection
from lab.infrastructure.nn.source_transformer import rotate


def forward(self, x, cos, sin, *, return_guard=False):
    batch, length, width = x.shape
    q, k, v = self.c_attn(x).chunk(3, dim=-1)
    q, k, v = (z.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
               for z in (q, k, v))
    q, k = rotate(self.q_norm(q), cos, sin), rotate(self.k_norm(k), cos, sin)
    centered = v - v[:, :, :1]
    average = F.scaled_dot_product_attention(q, k, centered, is_causal=True)
    y = centered_projection(average, centered, v)
    y = torch.cat((torch.zeros_like(y[:, :, :1]), y[:, :, 1:]), dim=2)
    output = self.c_proj(y.transpose(1, 2).contiguous().view(batch, length, width))
    if return_guard:
        return output, (v[:, :, 0].norm(dim=-1) > XSA_EPS).all(dim=1)
    return output


def apply(model):
    for block in model.blocks:
        block.sa.forward = MethodType(forward, block.sa)
    return model


def guarded_forward(self, inputs):
    length = inputs.shape[1]
    constant = (inputs != inputs[:, :1]).cumsum(dim=1) == 0
    x = self.embedding_norm(self.token_embedding_table(inputs))
    for block in self.blocks:
        attention, valid = block.sa(block.ln_1(x), self.rope_cos[:length],
                                   self.rope_sin[:length], return_guard=True)
        x = x + block.attn_out_norm(attention)
        x = x + block.ffn_out_norm(block.ffwd(block.ln_2(x)))
        constant = constant & valid[:, None]
        x = torch.where(constant[:, :, None], x[:, :1], x)
    return self.lm_head(self.ln_f(x))


def apply_guard(model):
    apply(model)
    model.forward = MethodType(guarded_forward, model)
    return model

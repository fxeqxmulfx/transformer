"""Literal tensor embedding, causal residual stack and tied readout.

Source: Structured.TensorEmbedding/Stream/DeferredFFN/Stack/StackInterface
at b0a43a8. All free fields are shared across layers. Deviations: float32
by default and stable head contractions; every original zero FFN and all
repeated attention computations are executed, without a semantic oracle.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .tensor_heads import QUARTERS, TensorHeads, recover


def digits(tokens, places=5):
    radix = 4 ** torch.arange(places, device=tokens.device)
    return (tokens[..., None] // radix) % 4


class TensorEmbedding(nn.Module):
    """52 free input fields, ten fixed output axes, unit anchor and spares."""
    def __init__(self, vocab, width, std):
        super().__init__()
        self.fields = nn.Parameter(torch.empty(vocab, 52))
        nn.init.normal_(self.fields, std=std)
        code = torch.tensor(QUARTERS)[digits(torch.arange(vocab))].flatten(-2)
        self.register_buffer("code", code)
        self.register_buffer("anchor", torch.ones(vocab, 1))
        self.register_buffer("spare", torch.zeros(vocab, width - 63))

    @property
    def weight(self):
        return torch.cat((self.fields, self.code, self.anchor, self.spare), -1)

    def forward(self, tokens):
        return F.embedding(tokens, self.weight)


class DeferredFFN(nn.Module):
    """Actual original bias-free, width*4 ReLU2 FFN with fixed zero matrices."""
    def __init__(self, width):
        super().__init__()
        self.register_buffer("input_weight", torch.zeros(4 * width, width))
        self.register_buffer("output_weight", torch.zeros(width, 4 * width))

    def forward(self, x):
        return F.linear(F.relu(F.linear(x, self.input_weight)).square(), self.output_weight)


class TensorAttention(nn.Module):
    """Drop-in NTC attention output for intermediate or final residuals."""
    def __init__(self, heads, width, final):
        super().__init__()
        self.heads, self.width, self.final = heads, width, final

    def forward(self, x, rotary=None):
        coordinates = self.heads(x, rotary)
        output = F.pad(coordinates, (52, self.width - 62))
        raw = x / x[..., 62:63]
        if self.final:
            return output - raw
        return output - F.pad(raw[..., 52:62], (52, self.width - 62))


class TensorBlock(nn.Module):
    """Original two prenorm residual interfaces, replacing only attention."""
    def __init__(self, heads, width, eps, final):
        super().__init__()
        self.attention = TensorAttention(heads, width, final)
        self.ffn, self.width, self.eps = DeferredFFN(width), width, eps

    def forward(self, x, rotary=None):
        x = x + self.attention(F.rms_norm(x, (self.width,), eps=self.eps), rotary)
        return x + self.ffn(F.rms_norm(x, (self.width,), eps=self.eps))


class TensorStack(nn.Module):
    """f(List Int) adapter and ordinary model(tokens, positions) interface."""
    def __init__(self, spec, vocab):
        super().__init__()
        spec.parameter_count(vocab)
        self.spec, self.vocab, self.context = spec, vocab, spec.context
        self.embed = TensorEmbedding(vocab, spec.width, spec.std)
        self.absolute = nn.Parameter(torch.zeros(spec.context))
        self.heads = TensorHeads(spec.context, spec.std)
        self.blocks = nn.ModuleList(TensorBlock(self.heads, spec.width, spec.eps, layer == spec.depth - 1)
                                    for layer in range(spec.depth))

    def inputs(self, tokens):
        length = tokens.shape[1]
        if not 1 <= length <= self.context:
            raise ValueError(f"Input length {length} exceeds context {self.context}, or is empty")
        position = F.pad(self.absolute[:length, None], (63, self.spec.width - 64))
        return self.embed(tokens) + position

    def stack_input(self, tokens):
        """The true input to the final attention, with every earlier block."""
        x = self.inputs(tokens)
        for block in self.blocks[:-1]:
            x = block(x)
        return F.rms_norm(x, (self.spec.width,), eps=self.spec.eps)

    def forward(self, tokens, positions=None):
        x = self.inputs(tokens)
        for block in self.blocks:
            x = block(x)
        x = F.rms_norm(x, (self.spec.width,), eps=self.spec.eps)
        if positions is not None:
            x = x.gather(1, positions[..., None].expand(-1, -1, self.spec.width))
        return F.linear(x, self.embed.weight)

    def hidden_matrices(self):
        return [self.embed.fields, self.heads.emission]

    def complete_losses(self, tokens, observed, branch):
        """Actual final-head complete NLL, with labels external to inference.

        observed holds previous/next data states and key/value routes at
        each physical position. The loss module alone knows the branch.
        Source: tensorStackNLL and tensorMixedNLL at b0a43a8.
        """
        fields, positions = recover(self.stack_input(tokens))
        head = self.heads.branch.logsumexp(-1) - self.heads.branch[branch]
        if branch == 0:
            previous, state, target = observed.unbind(1)
            rows = fields[..., :36].reshape(*tokens.shape, 6, 6)
            row = rows.gather(2, previous[..., None, None].expand(-1, -1, 1, 6)).squeeze(2)
            transition = row.logsumexp(-1) - row.gather(-1, state[..., None]).squeeze(-1)
            emission = self.heads.emission[state]
            selected = emission.gather(-1, digits(target)[..., None]).squeeze(-1).sum(-1)
            values = emission.logsumexp(-1).sum(-1) - selected
            initial = self.heads.initial.logsumexp(-1) - self.heads.initial[0]
            return head + initial + transition.cumsum(1) + values
        key_position, value_position, target = observed.unbind(1)
        partition = self.heads.pointer_terms(fields, positions)[0]
        key_fields = fields.gather(1, key_position[..., None].expand(-1, -1, 52))
        value_fields = fields.gather(1, value_position[..., None].expand(-1, -1, 52))
        matching = fields[..., :16] + key_fields[..., 16:32]
        matching = matching.reshape(*tokens.shape, 4, 4)
        key_tokens = tokens.gather(1, key_position)
        match_score = matching.gather(-1, digits((key_tokens + 220) % 256, 4)[..., None]).squeeze(-1).sum(-1)
        value = value_fields[..., 32:52].reshape(*tokens.shape, 5, 4)
        value_score = value.gather(-1, digits(target)[..., None]).squeeze(-1).sum(-1)
        bias = positions.gather(1, value_position) + self.heads.chronology * value_position
        bias = bias + self.heads.relative[value_position - key_position + self.context - 1]
        return head + partition - bias - match_score - value_score

    @torch.no_grad()
    def integer_function(self, tokens):
        """Preserve the input and append one checked greedy token; invalid is PAD."""
        if (not tokens or len(tokens) > self.context or
                any(type(token) is not int or not 0 <= token < self.vocab for token in tokens)):
            return [*tokens, 0]
        batch = torch.tensor([tokens], device=self.absolute.device)
        return [*tokens, int(self(batch)[0, -1].argmax())]

"""GPU implementation of the proved content-cost simplex program."""

import torch
from torch import nn


class ConvexRecall(nn.Module):
    def __init__(self, vocab, metric):
        super().__init__()
        self.vocab = vocab
        self.register_buffer("metric", torch.as_tensor(metric, dtype=torch.float32))
        codes = ((torch.arange(vocab)[:, None] >> torch.arange(len(metric))) & 1).float()
        self.register_buffer("codes", codes)

    @staticmethod
    def project_simplex(z):
        ordered = z.sort(dim=-1, descending=True).values
        cumulative = ordered.cumsum(-1)
        ranks = torch.arange(1, z.shape[-1] + 1, device=z.device)
        count = (1 + ranks * ordered > cumulative).sum(-1, keepdim=True)
        threshold = (cumulative.gather(-1, count - 1) - 1) / count
        return (z - threshold).clamp_min(0)

    def forward(self, tokens, positions):
        pairs = tokens.shape[1] // 4
        keys, values = tokens[:, :2 * pairs:2], tokens[:, 1:2 * pairs:2]
        queries = tokens.gather(1, positions)
        q, k = self.codes[queries], self.codes[keys]
        cost = ((q * self.metric).sum(-1, keepdim=True) +
                (k * self.metric).sum(-1)[:, None, :] -
                2 * (q * self.metric) @ k.transpose(-1, -2))
        allowed = 2 * torch.arange(pairs, device=tokens.device)[None, None, :] + 1 < positions[..., None]
        key_scores = (-2 * cost).masked_fill(~allowed, -torch.inf)
        null_score = -torch.ones_like(key_scores[..., :1])
        weights = self.project_simplex(torch.cat((key_scores, null_score), dim=-1))
        logits = torch.zeros((*queries.shape, self.vocab), device=tokens.device)
        logits.scatter_add_(2, values[:, None, :].expand(-1, queries.shape[1], -1),
                            weights[..., :pairs])
        prediction = logits.argmax(-1)
        return prediction.masked_fill(weights[..., -1] > logits.amax(-1), -1)

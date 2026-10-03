"""Free generation: each answer generated greedily from its prompt alone, and scored.

A port of `experiments/synthetic_trainers/generation.py` (`rollout`) and
`generation_metrics.py` (`GenerationMetrics`). A row writes its most likely
next token until it writes EOS or reaches its generation limit, and reads its
own tokens, mistakes included, as its context. `rollout` issues the
historical steps: each reads only the rows still generating, padded to the
longest of them, so it computes the historical logits bit for bit.
`static_rollout` reads every row at every step, at widths fixed in advance,
and never waits on the device, so a captured evaluation can replay it. A
model's logits at a position do not depend on later tokens, so both write
the same answers unless a difference in the last bits of the logits changes
a most likely token.

The historical rollout also counted its forward calls, the padded tokens and
attention cells it processed, and its time: the cost of one implementation,
not a property of the model, and not reported here. Unlike it, neither
rollout refuses logits that are not finite.
"""

import torch

from .vocabulary import EOS, IGNORE, PAD

# Totals of a generation: correct tokens, exact answers, correct final answers, answers ended by EOS, tokens
# generated, tokens beyond the answer's length, and correct tokens where the answer changes.
TOTALS = 7


@torch.no_grad()
def rollout(model, answers):
    """The tokens each row of `answers` (`rows.Answers`) generates, padded with PAD, and how many, step by step."""
    device = answers.context.device
    contexts = [list(prompt) for prompt in answers.prompts]
    generated = [[] for _ in contexts]
    active = list(range(len(contexts)))
    while active:
        sizes = [len(contexts[row]) for row in active]
        tokens = torch.full((len(active), max(sizes)), PAD, dtype=torch.long)
        for offset, row in enumerate(active):
            tokens[offset, :sizes[offset]] = torch.tensor(contexts[row])
        logits = model(tokens.to(device))
        last = logits[torch.arange(len(active), device=device), torch.tensor(sizes, device=device) - 1]
        remaining = []
        for row, token in zip(active, last.argmax(-1).tolist()):
            generated[row].append(token)
            if token != EOS and len(generated[row]) < answers.limits[row]:
                contexts[row].append(token)
                remaining.append(row)
        active = remaining
    steps = len(answers.widths)
    predictions = torch.tensor([row + [PAD] * (steps - len(row)) for row in generated], device=device)
    return predictions, torch.tensor([len(row) for row in generated], device=device)


@torch.no_grad()
def static_rollout(model, answers):
    """`rollout` at shapes fixed in advance, reading nothing back from the device.

    At step s the model reads the widest context a row can hold then,
    `answers.widths[s]`. A row that has stopped keeps its context, and the
    tokens it is still given are not counted.
    """
    context, lengths, bounds = answers.context.clone(), answers.lengths, answers.bounds
    rows = torch.arange(len(lengths), device=lengths.device)
    active = torch.ones_like(lengths, dtype=torch.bool)
    count = torch.zeros_like(lengths)
    predictions = []
    for step, width in enumerate(answers.widths):
        logits = model(context[:, :width])
        token = logits[rows, (lengths + step - 1).clamp(max=width - 1)].argmax(-1)
        predictions.append(token)
        count += active
        active &= (token != EOS) & (bounds > step + 1)
        position = (lengths + step).clamp(max=context.shape[1] - 1)
        context[rows, position] = torch.where(active, token, context[rows, position])
    return torch.stack(predictions, 1), count


def score(answers, predictions, count, final_token):
    """The float64 totals (`TOTALS`) of a chunk's generations, and whether each answer token was generated.

    A token is correct when the generation reaches its position and writes
    it; an answer is exact when it is generated whole and nothing after it.
    The final answer is the whole answer, or with `final_token` the token
    before EOS, which a scratchpad precedes, in a generation that ends with
    EOS, whatever its length.
    """
    expected, changes = answers.answers, answers.changes
    valid = expected != IGNORE
    sizes = valid.sum(1)
    reached = torch.arange(expected.shape[1], device=count.device) < count[:, None]
    correct = (predictions[:, :expected.shape[1]] == expected) & reached
    exact = (count == sizes) & (correct | ~valid).all(1)
    ended = predictions.gather(1, (count - 1)[:, None]).squeeze(1) == EOS
    if final_token:
        before = predictions.gather(1, (count - 2).clamp(min=0)[:, None]).squeeze(1)
        final = ended & (count >= 2) & (before == expected.gather(1, (sizes - 2)[:, None]).squeeze(1))
    else:
        final = exact
    totals = torch.stack([correct.sum(), exact.sum(), final.sum(), ended.sum(), count.sum(),
                          (count - sizes).clamp(min=0).sum(), (correct & changes).sum()]).double()
    return totals, correct

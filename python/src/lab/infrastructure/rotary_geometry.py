"""Character and position geometry of the learned Rotate90 embedding.

Source: landscape_grokking_plan.md, M9.3--M9.4 and Mechanisms to test.
These observations extend the browser protocol; they do not affect updates.
Nonzero residues are ordered by powers of the smallest multiplicative generator.
For centered rows X, rank is tr(XX.T)^2 / ||XX.T||_F^2 and J overlap is
||X (JX).T||_F / ||XX.T||_F. Fourier energies pair conjugate nonconstant
frequencies and normalize by their total energy, excluding the common row mean.
"""

import math

import torch


def multiplicative_ordering(prime):
    """Return a generator and all its powers, checking the finite permutation."""
    for generator in range(1, prime):
        order = tuple(pow(generator, index, prime) for index in range(prime - 1))
        if len(set(order)) == prime - 1:
            return generator, order
    raise ValueError("Nonzero residues did not form the expected cyclic prime field")


def embedding_geometry(embedding, order):
    """Read centered nonzero-class geometry in float64, with no autograd graph.

    Undefined ratios are None. A character frequency k has order n/gcd(k,n);
    a single nonfaithful character cannot distinguish all n nonzero residues.
    The complete pair-energy vector is retained so multiple characters can
    be inspected without assuming one dominant faithful character.
    """
    rows = embedding.detach().index_select(0, order).double()
    mean = rows.mean(dim=0)
    centered = rows - mean
    gram = centered @ centered.T
    energy, squared_gram = centered.square().sum(), gram.square().sum()
    total = rows.square().sum()
    result = {
        "mean_energy_fraction": float(len(rows) * mean.square().sum() / total) if total > 0 else None,
        "effective_rank": float(energy.square() / squared_gram) if squared_gram > 0 else None,
        "J_overlap": None,
        "character_pair_shares": [],
        "largest_character_pair_share": None,
        "dominant_character_frequency": None,
        "dominant_character_order": None,
    }
    if squared_gram > 0:
        pairs = centered.reshape(len(rows), -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        cross = centered @ rotated.T
        result["J_overlap"] = float((cross.square().sum() / squared_gram).sqrt())
    if len(rows) > 1 and energy > 0:
        power = torch.fft.fft(centered, dim=0).abs().square().sum(dim=1)
        frequencies = torch.arange(1, len(rows) // 2 + 1, device=rows.device)
        mirrors = (-frequencies) % len(rows)
        paired = power[frequencies] + torch.where(frequencies == mirrors, 0., power[mirrors])
        shares = paired / paired.sum()
        index = int(shares.argmax())
        frequency = index + 1
        result.update(character_pair_shares=shares.tolist(),
                      largest_character_pair_share=float(shares[index]),
                      dominant_character_frequency=frequency,
                      dominant_character_order=len(rows) // math.gcd(frequency, len(rows)))
    return result

module

public import Transformer.AttentionResiduals.Section4_OnlineMerge

/-!
# Attention Residuals and the residual mechanism of Kimi K3

Residual sections of Kimi Team, arXiv:2603.15031v1, §§3–4.2, and the
Kimi K3 technical report, §2.2, Eqs. (8)–(10). Includes RMS-normalized
keys, unnormalized values, learned pseudo-queries, block source selection,
raw partial-block accumulation, convex bounds, and exact two-phase
online-softmax merging. The source's unqualified equivalence between a
single block and ordinary residual connections has a formal counterexample.

Kimi K3's pinned config and reference code provide the concrete 93 decoder
blocks, block size 12, and separate attention/MLP aggregation operations.
Training results, communication scheduling and floating-point accuracy
are outside the scope. See `review/residual_connections.md`.
-/

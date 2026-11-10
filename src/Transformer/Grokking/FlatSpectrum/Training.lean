module

public import Transformer.Grokking.FlatSpectrum.Training.Defs
public import Transformer.Grokking.FlatSpectrum.Training.Basic
public import Transformer.Grokking.FlatSpectrum.Training.Lifted
public import Transformer.Grokking.FlatSpectrum.Training.Certificate
public import Transformer.Grokking.FlatSpectrum.Training.Gain
public import Transformer.Grokking.FlatSpectrum.Training.Growth
public import Transformer.Grokking.FlatSpectrum.Training.Continuity
public import Transformer.Grokking.FlatSpectrum.Training.Gradient
public import Transformer.Grokking.FlatSpectrum.Training.Limit
public import Transformer.Grokking.FlatSpectrum.Training.LinearRun
public import Transformer.Grokking.FlatSpectrum.Training.Convergence

/-! Training the pure factorization with weight decay, `review/flat_spectrum.md`, Lemma 1's
modeling step and *Training reaches the optimal fit* (added).

The model is `X = U Vᴴ` with the weight-decayed loss `L (U Vᴴ) + λ · ½ (‖U‖² + ‖V‖²)` for a
convex `L` with gradient `G`, trained by gradient descent with step `η` (`decayRun`). With the
factors stacked, `W = [U; V]`, a step is `W ← (1 − η M) W` for the Hermitian
`M = [[λ, G], [Gᴴ, λ]]` (WD1). A stationary point, `M W = 0`, with `M` positive semidefinite
has the least weight-decayed loss among the factorizations of every width, and then `U Vᴴ`
minimizes `L + λ ‖·‖_S1` with weight decay equal to `‖U Vᴴ‖_S1` (WD2). A continuous `G` with
the first-order inequality `L X + ⟨G X, Y − X⟩ ≤ L Y` is the gradient of `L`, and the update is
the gradient step of the weight-decayed loss (WD3). A bounded run of
`W ← (1 − η M_t) W` from a start of full row rank, with positive definite steps and `M_t → M∞`,
has `M∞` positive semidefinite: along a negative direction of `M∞` a column of `W_t` would grow
geometrically (WD4). So a convergent run of gradient descent with weight decay, from stacked
factors of full row rank and with positive definite step matrices, ends at a factorization of
least weight-decayed loss among those of every width: it cannot end at a saddle or at a local
minimum that is not global (WD5). The linear loss `−Re X` of a `1 × 1` matrix gives a run that
satisfies every hypothesis and converges to a nonzero limit. -/

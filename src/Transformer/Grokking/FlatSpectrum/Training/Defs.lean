module

public import Mathlib.Data.Matrix.ColumnRowPartitioned
public import Transformer.Grokking.FlatSpectrum.WeightDecay

/-!
# Training the pure factorization with weight decay: definitions

Source: `review/flat_spectrum.md`, Lemma 1: "Weight decay on a bilinear layer therefore acts as
a nuclear-norm penalty. Training with weight decay then behaves like minimum-nuclear-norm
interpolation from Ω. This is the modeling step. It is exact for a pure factorization", and
*Training reaches the optimal fit* (added), item WD1.

The pure factorization is `X = U Vᴴ` with `U : m × k` and `V : n × k`. Its weight-decayed loss
is `L (U Vᴴ) + λ · ½ (‖U‖² + ‖V‖²)` (`decayedLoss`) for a loss `L` of the product. With `G` the
gradient of `L`, gradient descent with step `η` is `decayStep`:

    U ← U − η (G(U Vᴴ) V + λ U),    V ← V − η (G(U Vᴴ)ᴴ U + λ V),

and `decayRun` iterates it. Stacking the factors, `W = [U; V]` (`stack`), turns the step into
`W ← (1 − η M) W` with the Hermitian block matrix `M = [[λ, G], [Gᴴ, λ]]` (`lift`).
`IsDecayedMin` says that a factorization has the least weight-decayed loss among the
factorizations of every width, the counterpart of `IsOptimalFit` for a penalty instead of a
constraint. `pairing A B = Re tr (Aᴴ B)` is the real inner product whose square is `‖·‖_S2²`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k]

/-- The real trace pairing `Re tr (Aᴴ B)` of two complex matrices: the inner product of the
real space of complex matrices, with `pairing A A = ‖A‖_S2²`. -/
noncomputable def pairing (A B : Matrix m n ℂ) : ℝ := (Aᴴ * B).trace.re

/-- The weight-decayed loss `L (U Vᴴ) + λ · ½ (‖U‖² + ‖V‖²)` of the factorization `U Vᴴ`,
`review/flat_spectrum.md`, Lemma 1 and item WD1. -/
noncomputable def decayedLoss (L : Matrix m n ℂ → ℝ) (lam : ℝ) (U : Matrix m k ℂ)
    (V : Matrix n k ℂ) : ℝ :=
  L (U * Vᴴ) + lam * ((frobeniusSq U + frobeniusSq V) / 2)

/-- The stacked factors `W = [U; V]`, item WD1. -/
def stack (U : Matrix m k ℂ) (V : Matrix n k ℂ) : Matrix (m ⊕ n) k ℂ := fromRows U V

/-- The lifted gradient `M = [[λ, G], [Gᴴ, λ]]`, item WD1: the gradient of the weight-decayed
loss at `W = [U; V]` is `M W` with `G` the gradient of the loss at `U Vᴴ`. -/
noncomputable def lift [DecidableEq m] [DecidableEq n] (lam : ℝ) (G : Matrix m n ℂ) :
    Matrix (m ⊕ n) (m ⊕ n) ℂ :=
  fromBlocks (lam • 1) G Gᴴ (lam • 1)

/-- One step of gradient descent with weight decay `λ` and step `η` on the factors, for a loss
whose gradient at `X` is `G X`, item WD1. -/
noncomputable def decayStep (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ)
    (UV : Matrix m k ℂ × Matrix n k ℂ) : Matrix m k ℂ × Matrix n k ℂ :=
  (UV.1 - η • (G (UV.1 * UV.2ᴴ) * UV.2 + lam • UV.1),
    UV.2 - η • ((G (UV.1 * UV.2ᴴ))ᴴ * UV.1 + lam • UV.2))

/-- The run of gradient descent with weight decay from `(U₀, V₀)`, item WD1. -/
noncomputable def decayRun (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ) (U₀ : Matrix m k ℂ)
    (V₀ : Matrix n k ℂ) (t : ℕ) : Matrix m k ℂ × Matrix n k ℂ :=
  (decayStep G η lam)^[t] (U₀, V₀)

/-- A factorization of least weight-decayed loss among the factorizations of every width,
item WD2. For a penalty `λ` this is what `IsOptimalFit` is for an exact fit. -/
def IsDecayedMin (L : Matrix m n ℂ → ℝ) (lam : ℝ) (U : Matrix m k ℂ) (V : Matrix n k ℂ) :
    Prop :=
  ∀ (r : ℕ) (U' : Matrix m (Fin r) ℂ) (V' : Matrix n (Fin r) ℂ),
    decayedLoss L lam U V ≤ decayedLoss L lam U' V'

end Transformer.Grokking.FlatSpectrum.Training

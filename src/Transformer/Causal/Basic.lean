/-
# Causal attention — Basic definitions

Formalization of arXiv:2411.04990v2 — "Clustering in Causal Attention Masking"
(Karagodin, Polyanskiy, Rigollet).

Equations covered:

* `eq: sa`     — full self-attention dynamics,
* `eq: csa`    — causal self-attention (CSA) dynamics,
* `eq: csa-2d` — angular form on the circle `𝕊^1`,
* `eq: def_h`  — the interaction potential `h(x) = e^{β(cos x - 1)} sin x`,
* the *single-token* dynamics `ẋ = Proj_x V x` (Section 3),
* the eigenspaces `L, L'` associated with `λ_max(V)`.
-/

import Transformer.Basic
import Transformer.Section1_IPS

open scoped BigOperators
open Real

namespace Transformer
namespace Causal

variable (d n : ℕ)

/-- **Equation (eq: csa).** *Causal self-attention dynamics.*

  `ẋ_k(t) = Proj_{x_k(t)}( Z_k(t)⁻¹ Σ_{j=1}^{k} e^{β ⟨Q x_k, K x_j⟩} V x_j )`,
  `Z_k(t) = Σ_{j=1}^{k} e^{β ⟨Q x_k, K x_j⟩}`. -/
def CSA
    (β : ℝ) (Q K V : ParamMatrix d)
    (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ k : Idx n,
    HasDerivAt (fun s => (X s k : EucSpace d))
      (proj d ((X t k : EucSpace d))
        ((∑ j : Idx n,
            if (j : ℕ) ≤ (k : ℕ) then
              Real.exp (β * inner (𝕜 := ℝ)
                          (Q ((X t k : EucSpace d)))
                          (K ((X t j : EucSpace d))))
            else 0)⁻¹
        • ∑ j : Idx n,
            if (j : ℕ) ≤ (k : ℕ) then
              Real.exp (β * inner (𝕜 := ℝ)
                          (Q ((X t k : EucSpace d)))
                          (K ((X t j : EucSpace d))))
              • V ((X t j : EucSpace d))
            else 0)) t

/-- *Single-token dynamics (Section 3).*

  `ẋ(t) = Proj_{x(t)} (V x(t))`. -/
def singleTokenODE
    (V : ParamMatrix d) (x : ℝ → EucSpace d) : Prop :=
  ∀ t : ℝ, HasDerivAt x (proj d (x t) (V (x t))) t

/-- **The interaction potential `h` (eq: def_h):**

  `h(x) = e^{β (cos x - 1)} · sin x`. -/
noncomputable def h_pot (β x : ℝ) : ℝ :=
  Real.exp (β * (Real.cos x - 1)) * Real.sin x

/-- **Equation (eq: csa-2d).** *Causal SA on the circle.*

  `φ̇_k(t) = (1 / Z_k) Σ_{j=1}^{k-1} h(φ_j - φ_k)`. -/
def CSA_2d
    (β : ℝ) (φ : ℝ → Idx n → ℝ) : Prop :=
  ∀ t : ℝ, ∀ k : Idx n,
    HasDerivAt (fun s => φ s k)
      ((∑ j : Idx n,
        if (j : ℕ) ≤ (k : ℕ) then
          Real.exp (β * (Real.cos (φ t k - φ t j) - 1))
        else 0)⁻¹
      *
      ∑ j : Idx n,
        if (j : ℕ) < (k : ℕ) then h_pot β (φ t j - φ t k) else 0) t

end Causal
end Transformer

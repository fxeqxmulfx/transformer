/-
# Normalization in Attention Dynamics — Basic definitions

Formalization of arXiv:2510.22026v2 — "Normalization in Attention Dynamics"
(Karagodin, Polyanskiy, Rigollet).

Equations covered:

* the six normalization schemes Post-LN, Pre-LN, Mix-LN, Peri-LN, nGPT, CoD,
  in both discrete- and continuous-time forms (Table 1 in the paper);
* the *normalized attention dynamics* `eq: NA`:

    `θ̇_j(t) = (1 / s_j(t)) · Proj_{θ_j(t)} A^t_j(Θ(t))`,

  with speed-regulation factors `s_j(t)` listed in Table 2.
-/

import Transformer.Basic
import Transformer.Section1_IPS

open scoped BigOperators
open Real

namespace Transformer
namespace Normalization

variable (d n : ℕ)

/-- Possible normalization schemes considered in the paper. -/
inductive Scheme
  | post
  | pre
  | mix
  | peri
  | nGPT
  | CoD
  deriving DecidableEq, Repr

/-- Attention vector

  `A_j(Θ) = Σ_k w_{jk} V θ_k`,  `w_{jk} = e^{β ⟨Q θ_j, K θ_k⟩} / Z_j`. -/
noncomputable def attentionVec
    (β : ℝ) (Q K V : ParamMatrix d)
    (Θ : Idx n → EucSpace d) (j : Idx n) : EucSpace d :=
  let Z := ∑ l : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ l)))
  Z⁻¹ • ∑ k : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (Q (Θ j)) (K (Θ k))) • V (Θ k)

/-- **Equation (NA).** *Normalized attention dynamics.*

  `θ̇_j(t) = (1 / s_j(t)) Proj_{θ_j(t)} A^t_j(Θ(t))`. -/
def NA
    (β : ℝ) (Q K V : ℝ → ParamMatrix d)
    (s : ℝ → Idx n → ℝ)
    (θ : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ t : ℝ, ∀ j : Idx n,
    HasDerivAt (fun u => θ u j)
      ((s t j)⁻¹ •
        proj d (θ t j) (attentionVec d n β (Q t) (K t) (V t) (θ t) j)) t

/-- Speed-regulation factor `s_j(t)` as a function of the scheme. -/
noncomputable def speedFactor
    (β : ℝ) (Q K V : ℝ → ParamMatrix d) (α : ℝ → ℝ)
    (θ : ℝ → Idx n → EucSpace d) (r : ℝ → Idx n → ℝ)
    (τ : ℝ)
    (scheme : Scheme) (t : ℝ) (j : Idx n) : ℝ :=
  match scheme with
  | .post  => 1
  | .pre   => r t j
  | .mix   => if t ≤ τ then 1 else r t j
  | .peri  =>
      r t j * ‖attentionVec d n β (Q t) (K t) (V t) (θ t) j‖
  | .nGPT  =>
      (α t)⁻¹ * ‖attentionVec d n β (Q t) (K t) (V t) (θ t) j‖
  | .CoD   => Real.sqrt (t + 1)

/-- The corresponding radial-magnitude derivative `ṙ_j(t)`. -/
noncomputable def radialDerivative
    (β : ℝ) (Q K V : ℝ → ParamMatrix d)
    (θ : ℝ → Idx n → EucSpace d) (τ : ℝ)
    (scheme : Scheme) (t : ℝ) (j : Idx n) : ℝ :=
  match scheme with
  | .post  => 0
  | .pre   =>
      inner (𝕜 := ℝ) (θ t j)
        (attentionVec d n β (Q t) (K t) (V t) (θ t) j)
  | .mix   =>
      if τ < t then
        inner (𝕜 := ℝ) (θ t j)
          (attentionVec d n β (Q t) (K t) (V t) (θ t) j)
      else 0
  | .peri  =>
      inner (𝕜 := ℝ) (θ t j)
        (attentionVec d n β (Q t) (K t) (V t) (θ t) j)
        / ‖attentionVec d n β (Q t) (K t) (V t) (θ t) j‖
  | .nGPT  => 0
  | .CoD   => 0

end Normalization
end Transformer

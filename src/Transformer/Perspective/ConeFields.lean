/-
# Cone collapse — the attention dynamics as weighted flows

Auxiliary to arXiv:2312.10794v5, *A mathematical perspective on Transformers*,
§6.1, `lem: hemisphere.clustering`.

The four dynamics of the lemma — `SA`, `USA`, `eq: transformerSd.QKV` with
constant `Q, K` and `V = I_d`, and its `USA` analogue — are integral curves of
one smooth field on `(ℝ^d)^n`,

  `(F Y)_i = Proj_{y_i}( c_i(Y) Σ_j e^{β⟨Q y_i, K y_j⟩} y_j )`   (`attnField`),

with `c_i = Z_i⁻¹` (`softmaxNorm`) or `c_i = n⁻¹`.  They are weighted flows
(`Perspective.IsWeightedFlow`) with weights `a_ij = c_i e^{β⟨Q x_i, K x_j⟩}`,
which on the sphere lie in `[e^{-2L}/n, e^{2L}/n]`, `L = |β| ‖Q‖ ‖K‖`
(`attnExp_bounds`, `softmaxNorm_bounds`).  Since the field is smooth, a
solution through a given point is unique (`GlobalFlow.eq_of_hasDerivAt`), and
the exponential convergence of one solution
(`exists_expLimit_of_weightedFlow`) is that of every solution:
`expLimit_of_attnField`.
-/

import Transformer.GlobalFlow.Existence
import Transformer.Perspective.ConeLimit
import Mathlib.Analysis.Calculus.ContDiff.RCLike
import Mathlib.Analysis.Calculus.Deriv.Prod

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- The unnormalised attention weight `e^{β⟨Q y_i, K y_j⟩}`.

Source: arXiv:2312.10794v5, §1, `eq: transformerSd.QKV`. -/
noncomputable def attnExp (β : ℝ) (Q K : ParamMatrix d) (Y : Idx n → EucSpace d)
    (i j : Idx n) : ℝ :=
  exp (β * inner (𝕜 := ℝ) (Q (Y i)) (K (Y j)))

/-- The softmax normalisation `Z_i(Y)⁻¹ = (Σ_k e^{β⟨Q y_i, K y_k⟩})⁻¹`.

Source: arXiv:2312.10794v5, §1, `eq: transformerSd.QKV`. -/
noncomputable def softmaxNorm (β : ℝ) (Q K : ParamMatrix d) (Y : Idx n → EucSpace d)
    (i : Idx n) : ℝ :=
  (∑ k : Idx n, attnExp β Q K Y i k)⁻¹

/-- **The drift of attention with value `I_d` and row normalisation `c`:**
`(F Y)_i = Proj_{y_i}(c_i(Y) Σ_j e^{β⟨Q y_i, K y_j⟩} y_j)`.

Source: arXiv:2312.10794v5, §1, `eq: transformerSd.QKV` (`c = softmaxNorm`)
and §2, `USA` (`c = n⁻¹`). -/
noncomputable def attnField (β : ℝ) (Q K : ParamMatrix d)
    (c : (Idx n → EucSpace d) → Idx n → ℝ) (Y : Idx n → EucSpace d) : Idx n → EucSpace d :=
  fun i => proj d (Y i) (c Y i • ∑ j : Idx n, attnExp β Q K Y i j • Y j)

variable {β : ℝ} {Q K : ParamMatrix d} {c : (Idx n → EucSpace d) → Idx n → ℝ}

/-- The unnormalised weight is smooth in the configuration.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering` ("the unique
solution"). -/
theorem contDiff_attnExp (i j : Idx n) : ContDiff ℝ 1 fun Y => attnExp β Q K Y i j :=
  (contDiff_const.mul ((Q.contDiff.comp (contDiff_apply ℝ (EucSpace d) i)).inner ℝ
    (K.contDiff.comp (contDiff_apply ℝ (EucSpace d) j)))).exp

/-- The softmax normalisation is smooth: the partition function is positive.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering` ("the unique
solution"). -/
theorem contDiff_softmaxNorm (i : Idx n) : ContDiff ℝ 1 fun Y => softmaxNorm β Q K Y i := by
  unfold softmaxNorm
  refine ContDiff.inv (ContDiff.sum fun k _ => contDiff_attnExp i k) fun Y => ?_
  exact (Finset.sum_pos (fun k _ => exp_pos _) ⟨i, Finset.mem_univ i⟩).ne'

/-- The drift is `C¹` as soon as the normalisation is, hence locally Lipschitz.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering` ("the unique
solution"). -/
theorem locallyLipschitz_attnField (hc : ∀ i, ContDiff ℝ 1 fun Y => c Y i) :
    LocallyLipschitz (attnField β Q K c) := by
  refine ContDiff.locallyLipschitz (𝕂 := ℝ) (contDiff_pi.2 fun i => ?_)
  have hp : ∀ k : Idx n, ContDiff ℝ 1 fun Y : Idx n → EucSpace d => Y k :=
    contDiff_apply ℝ (EucSpace d)
  have hv : ContDiff ℝ 1 fun Y : Idx n → EucSpace d =>
      c Y i • ∑ j : Idx n, attnExp β Q K Y i j • Y j :=
    (hc i).smul (ContDiff.sum fun j _ => (contDiff_attnExp i j).smul (hp j))
  exact hv.sub (((hp i).inner ℝ hv).smul (hp i))

/-- On the sphere, `e^{-L} ≤ e^{β⟨Q y_i, K y_j⟩} ≤ e^{L}` with `L = |β| ‖Q‖ ‖K‖`.

Source: arXiv:2312.10794v5, §6.1, proof of `lem: hemisphere.clustering`,
step 2: `a_ij(t) ≥ n⁻¹ e^{-2β}`, there for `Q = K = I_d` and `β > 0`, where
`L = β`. -/
theorem attnExp_bounds {Y : Idx n → EucSpace d} (hY : ∀ j, ‖Y j‖ = 1) (i j : Idx n) :
    exp (-(|β| * (‖Q‖ * ‖K‖))) ≤ attnExp β Q K Y i j ∧
      attnExp β Q K Y i j ≤ exp (|β| * (‖Q‖ * ‖K‖)) := by
  have hs : |β * inner (𝕜 := ℝ) (Q (Y i)) (K (Y j))| ≤ |β| * (‖Q‖ * ‖K‖) := by
    rw [abs_mul]
    refine mul_le_mul_of_nonneg_left ((abs_real_inner_le_norm _ _).trans ?_) (abs_nonneg β)
    calc ‖Q (Y i)‖ * ‖K (Y j)‖ ≤ (‖Q‖ * ‖Y i‖) * (‖K‖ * ‖Y j‖) :=
          mul_le_mul (Q.le_opNorm _) (K.le_opNorm _) (norm_nonneg _) (by positivity)
      _ = ‖Q‖ * ‖K‖ := by rw [hY, hY, mul_one, mul_one]
  obtain ⟨h1, h2⟩ := abs_le.mp hs
  exact ⟨exp_le_exp.2 h1, exp_le_exp.2 h2⟩

/-- On the sphere, the softmax weights lie in `[e^{-2L}/n, e^{2L}/n]`.

Source: arXiv:2312.10794v5, §6.1, proof of `lem: hemisphere.clustering`,
step 2: `a_ij(t) ≥ n⁻¹ e^{-2β}`, there for `Q = K = I_d` and `β > 0`, where
`L = β`. -/
theorem softmaxNorm_bounds {Y : Idx n → EucSpace d} (hY : ∀ j, ‖Y j‖ = 1) (i j : Idx n) :
    exp (-(2 * (|β| * (‖Q‖ * ‖K‖)))) / n ≤ softmaxNorm β Q K Y i * attnExp β Q K Y i j ∧
      softmaxNorm β Q K Y i * attnExp β Q K Y i j ≤ exp (2 * (|β| * (‖Q‖ * ‖K‖))) / n := by
  set L := |β| * (‖Q‖ * ‖K‖)
  set Z := ∑ k : Idx n, attnExp β Q K Y i k
  have hn : (0 : ℝ) < n := by exact_mod_cast Fin.pos i
  have hZlo : n * exp (-L) ≤ Z := by
    have := Finset.card_nsmul_le_sum Finset.univ (fun k => attnExp β Q K Y i k) (exp (-L))
      fun k _ => (attnExp_bounds hY i k).1
    rwa [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at this
  have hZhi : Z ≤ n * exp L := by
    have := Finset.sum_le_card_nsmul Finset.univ (fun k => attnExp β Q K Y i k) (exp L)
      fun k _ => (attnExp_bounds hY i k).2
    rwa [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at this
  have hZ : 0 < Z := lt_of_lt_of_le (by positivity) hZlo
  obtain ⟨he1, he2⟩ := attnExp_bounds (β := β) (Q := Q) (K := K) hY i j
  have h2L : exp (-(2 * L)) * exp L = exp (-L) := by rw [← exp_add]; ring_nf
  have h2L' : exp (2 * L) * exp (-L) = exp L := by rw [← exp_add]; ring_nf
  rw [softmaxNorm, inv_mul_eq_div, div_le_div_iff₀ hn hZ, div_le_div_iff₀ hZ hn]
  constructor
  · calc exp (-(2 * L)) * Z ≤ exp (-(2 * L)) * (n * exp L) := by gcongr
      _ = exp (-L) * n := by rw [← h2L]; ring
      _ ≤ attnExp β Q K Y i j * n := by gcongr
  · calc attnExp β Q K Y i j * n ≤ exp L * n := by gcongr
      _ = exp (2 * L) * (n * exp (-L)) := by rw [← h2L']; ring
      _ ≤ exp (2 * L) * Z := by gcongr

/-- On the sphere, the `USA` weights `n⁻¹ e^{β⟨Q x_i, K x_j⟩}` lie in
`[e^{-2L}/n, e^{2L}/n]`.

Source: arXiv:2312.10794v5, §6.1, proof of `lem: hemisphere.clustering`,
step 2: `a_ij(t) ≥ n⁻¹ e^{-2β}`, there for `Q = K = I_d` and `β > 0`, where
`L = β`. -/
theorem usaNorm_bounds {Y : Idx n → EucSpace d} (hY : ∀ j, ‖Y j‖ = 1) (i j : Idx n) :
    exp (-(2 * (|β| * (‖Q‖ * ‖K‖)))) / n ≤ (n : ℝ)⁻¹ * attnExp β Q K Y i j ∧
      (n : ℝ)⁻¹ * attnExp β Q K Y i j ≤ exp (2 * (|β| * (‖Q‖ * ‖K‖))) / n := by
  have hL : 0 ≤ |β| * (‖Q‖ * ‖K‖) := by positivity
  obtain ⟨he1, he2⟩ := attnExp_bounds (β := β) (Q := Q) (K := K) hY i j
  rw [inv_mul_eq_div]
  exact ⟨div_le_div_of_nonneg_right ((exp_le_exp.2 (by linarith)).trans he1) (Nat.cast_nonneg n),
    div_le_div_of_nonneg_right (he2.trans (exp_le_exp.2 (by linarith))) (Nat.cast_nonneg n)⟩

/-- The hypotheses of the bounds above are satisfiable: one token at the base
point of `𝕊⁰`. -/
example : ∀ j : Idx 1, ‖((fun _ => basePoint 0 : SphereTuple 1 1) j : EucSpace 1)‖ = 1 :=
  fun j => norm_coe_tuple (fun _ => basePoint 0 : SphereTuple 1 1) j

/-- **Cone collapse for an attention drift.**  If every solution of `dyn` is an
integral curve of `attnField β Q K c`, with a `C¹` normalisation `c` whose
weights `c_i e^{β⟨Q y_i, K y_j⟩}` lie in `[m, M]`, `m > 0`, on the sphere, then
from a configuration in an open hemisphere every solution converges
exponentially to one and the same `x⋆`.

Source: arXiv:2312.10794v5, §6.1, `lem: hemisphere.clustering`. -/
theorem expLimit_of_attnField {dyn : (ℝ → SphereTuple d n) → Prop}
    (hdyn : ∀ X, dyn X → ∀ (t : ℝ) (i : Idx n), HasDerivAt (fun s => (X s i : EucSpace d))
      (attnField β Q K c (fun j => (X t j : EucSpace d)) i) t)
    (hc : ∀ i, ContDiff ℝ 1 fun Y => c Y i) {m M : ℝ} (hm : 0 < m)
    (hcb : ∀ Y : Idx n → EucSpace d, (∀ j, ‖Y j‖ = 1) → ∀ i j,
      m ≤ c Y i * attnExp β Q K Y i j ∧ c Y i * attnExp β Q K Y i j ≤ M)
    (hn : 0 < n) {X₀ : SphereTuple d n} (w : SSphere d)
    (hX₀ : ∀ i, 0 < inner (𝕜 := ℝ) (X₀ i : EucSpace d) (w : EucSpace d)) :
    ∃ (x_star : SSphere d) (C lam : ℝ), 0 < C ∧ 0 < lam ∧
      ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → dyn X →
        ∀ i : Idx n, ∀ t : ℝ, 0 ≤ t →
          ‖(X t i : EucSpace d) - x_star‖ ≤ C * exp (-(lam * t)) := by
  by_cases h : ∃ X, X 0 = X₀ ∧ dyn X
  · obtain ⟨X, hX0, hX⟩ := h
    have hflow : IsWeightedFlow X fun t i j => c (fun k => (X t k : EucSpace d)) i
        * attnExp β Q K (fun k => (X t k : EucSpace d)) i j := by
      intro t i
      refine (hdyn X hX t i).congr_deriv ?_
      simp only [attnField, Finset.smul_sum, smul_smul]
    obtain ⟨xs, C, lam, hC, hlam, hb⟩ := exists_expLimit_of_weightedFlow hflow hm
      (fun t i j => hcb _ (norm_coe_tuple (X t)) i j) hn w (by rw [hX0]; exact hX₀)
    refine ⟨xs, C, lam, hC, hlam, fun Y hY0 hY i t ht => ?_⟩
    have huniq := GlobalFlow.eq_of_hasDerivAt (locallyLipschitz_attnField hc)
      (f := fun s j => (Y s j : EucSpace d)) (g := fun s j => (X s j : EucSpace d))
      (fun s => hasDerivAt_pi.2 fun j => hdyn Y hY s j)
      (fun s => hasDerivAt_pi.2 fun j => hdyn X hX s j) (by simp only [hY0, hX0])
    rw [show (Y t i : EucSpace d) = X t i from congrFun (congrFun huniq t) i]
    exact hb i t ht
  · exact ⟨w, 1, 1, one_pos, one_pos, fun Y hY0 hY => absurd ⟨Y, hY0, hY⟩ h⟩

/-- The hypotheses of `expLimit_of_attnField` are satisfiable: `β = 0`, `c = 1`,
so every weight is `1`, and one token resting at the base point of `𝕊⁰`, in
the hemisphere around itself; `c = 1` is `C¹` in every row. -/
example : (∀ (t : ℝ) (i : Idx 1),
      HasDerivAt (fun _ => ((basePoint 0 : SSphere 1) : EucSpace 1))
        (attnField 0 0 0 (fun _ _ => 1) (fun _ => ((basePoint 0 : SSphere 1) : EucSpace 1)) i)
          t) ∧
    ContDiff ℝ 1 (fun (_ : Idx 1 → EucSpace 1) => (1 : ℝ)) ∧
    (∀ Y : Idx 1 → EucSpace 1, (∀ j, ‖Y j‖ = 1) → ∀ i j,
      (1 : ℝ) ≤ 1 * attnExp 0 0 0 Y i j ∧ 1 * attnExp 0 0 0 Y i j ≤ 1) ∧
    ∀ i : Idx 1, 0 < inner (𝕜 := ℝ) ((fun _ => basePoint 0 : SphereTuple 1 1) i : EucSpace 1)
      ((basePoint 0 : SSphere 1) : EucSpace 1) := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  refine ⟨fun t _ => ?_, contDiff_const, fun Y _ i j => ?_, fun _ => ?_⟩
  · simpa [attnField, attnExp, proj, hx] using
      hasDerivAt_const t ((basePoint 0 : SSphere 1) : EucSpace 1)
  · simp [attnExp]
  · simp [hx]

end Perspective
end Transformer

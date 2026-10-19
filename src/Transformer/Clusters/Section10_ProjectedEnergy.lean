module

public import Transformer.Clusters.Section10_ScoreError
public import Transformer.Clusters.Section8_SoftmaxEnergy

/-!
# The corrected projected energy derivative

ArXiv:2305.05465v6, §10, Step 2' and `e:derivativenorm123`.
The energy is `||A π_F z_i||²`, in the order supplied by the original
attention scores. Its derivative contains the leading speed `λ` from
the projected ODE. The source writes `||π_F A z_i||²` and omits this speed;
here both are corrected directly from `e:Rres` and `d:goodmulti`.

A softmax average of uniformly perturbed scores loses at most twice
their error plus `n/c`. The proved exponential score error therefore
gives an integrable lower bound on the actual projected energy derivative.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- Perturbing every score by at most `η` changes the reference-score
lower bound on the softmax average by at most `2η`.
Source: arXiv:2305.05465v6, §10, Step 2', an alternative to splitting
the indices by signs of their perturbed attention gaps. -/
theorem score_sub_card_div_sub_error_le_softmax_average (u w : Idx n → ℝ)
    (c : ℝ) (hc : 0 < c) (η : ℝ) (herror : ∀ j, |u j - w j| ≤ η) (i : Idx n) :
    w i - (n : ℝ) / c - 2 * η ≤
      ∑ j : Idx n, Perspective.softmaxWeight (fun k => c * u k) j * w j := by
  have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
  have hsum := Perspective.sum_softmaxWeight hn (fun k => c * u k)
  have haverage := score_sub_card_div_le_softmax_average u c hc i
  have hupper : ∀ j, u j ≤ w j + η := fun j => by
    have h := (abs_sub_le_iff.mp (herror j)).1
    linarith
  have hweighted := Finset.sum_le_sum (s := Finset.univ) (fun j _ =>
    mul_le_mul_of_nonneg_left (hupper j)
      (Perspective.softmaxWeight_nonneg (fun k => c * u k) j))
  simp only [mul_add, Finset.sum_add_distrib, ← Finset.sum_mul, hsum, one_mul] at hweighted
  have href := (abs_sub_le_iff.mp (herror i)).2
  linarith

/-- The positive temperature and complete uniform perturbation input
hold for a genuine two-token row with zero error.
Source: arXiv:2305.05465v6, §10, Step 2'. -/
example : (0 : ℝ) < 1 ∧ ∀ j : Idx 2,
    |(fun _ : Idx 2 => (0 : ℝ)) j - (fun _ : Idx 2 => (0 : ℝ)) j| ≤ 0 := by
  simp only [sub_self, abs_zero, le_refl, implies_true, and_true, zero_lt_one]

/-- Leading scalar normalization preserves the actual attention weights
when the inverse temperature is increased by its square.
Source: arXiv:2305.05465v6, §10, the original attention exponents.
The attention root acts on the normalized vectors before any projection. -/
theorem attentionMatrix_eq_normalized_softmax (Q K A : ParamMatrix d)
    (hA : IsAttentionRoot Q K A) (X : Idx n → EucSpace d) (lam t : ℝ) (i j : Idx n) :
    attentionMatrix Q K X i j = Perspective.softmaxWeight (fun k : Idx n =>
      Real.exp (2 * lam * t) * inner (𝕜 := ℝ)
        (A (Real.exp (-lam * t) • X i)) (A (Real.exp (-lam * t) • X k))) j := by
  have hscale : Real.exp (2 * lam * t) * Real.exp (-lam * t) * Real.exp (-lam * t) = 1 := by
    rw [← Real.exp_add, ← Real.exp_add]
    have hzero : 2 * lam * t + -lam * t + -lam * t = 0 := by ring
    rw [hzero, Real.exp_zero]
  unfold attentionMatrix
  congr 1
  funext k
  rw [hA.2.2, map_smul, map_smul, real_inner_smul_left, real_inner_smul_right]
  symm
  calc
    _ = (Real.exp (2 * lam * t) * Real.exp (-lam * t) * Real.exp (-lam * t)) *
        inner (𝕜 := ℝ) (A (X i)) (A (X k)) := by ring
    _ = _ := by rw [hscale, one_mul]

/-- The complete root hypothesis is realized by the identity maps.
Source: arXiv:2305.05465v6, §10, `A = (QᵀK)^{1/2}`. -/
example : IsAttentionRoot (1 : ParamMatrix d) 1 1 := isAttentionRoot_id d

/-- The corrected projected energy has exactly the derivative given by
the actual consensus equation, including its leading speed.
Source: arXiv:2305.05465v6, §10, `e:derivativenorm123`; the source writes
`π_F A` and drops `λ`, whereas `e:Rres` gives `A π_F` and the factor below. -/
theorem RescaledDynamics.hasDerivAt_norm_sq_projected (Q K V proj A : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) (t : ℝ) (i : Idx n) :
    HasDerivAt (fun s => ‖A (proj (Z s i))‖ ^ 2)
      (2 * lam * ∑ j : Idx n, attentionMatrix Q K (fun k => expTime V t (Z t k)) i j *
        inner (𝕜 := ℝ) (A (proj (Z t i))) (A (proj (Z t j) - proj (Z t i)))) t := by
  apply ((A.hasFDerivAt (x := proj (Z t i))).comp_hasDerivAt t
    (hZ.hasDerivAt_projected Q K V proj F G lam hV hproj Z t i)).norm_sq.congr_deriv
  simp only [Function.comp_def, map_sum, map_smul, inner_sum, real_inner_smul_right]
  rw [mul_assoc]
  congr 1
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- Every splitting, projection, and ODE input holds for coincident
nonzero tokens. Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _⟩

/-- **The actual corrected projected energy dominates every leading gap,
up to exponentially decaying errors.**
Source: arXiv:2305.05465v6, §10, Step 2' and `e:derivativenorm123`, with
the corrected order `A π_F` and the leading factor `λ`.
All quantitative inputs are proved from the actual dynamics. -/
theorem RescaledDynamics.exists_projected_energy_deriv_lower_bound
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hA : IsAttentionRoot Q K A) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) :
    ∃ δ D : ℝ, 0 < δ ∧ 0 < D ∧ ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      ∃ v : ℝ, HasDerivAt (fun s => ‖A (proj (Z s i))‖ ^ 2) v t ∧ ∀ j : Idx n,
        2 * lam * (inner (𝕜 := ℝ) (A (proj (Z t i)))
          (A (proj (Z t j) - proj (Z t i))) - (n : ℝ) / Real.exp (2 * lam * t) -
          2 * D * Real.exp (-δ * t)) ≤ v := by
  obtain ⟨δ, D, hδ, hD, herror⟩ :=
    hZ.exists_bound_normalized_score_error Q K V proj A F G lam hV hproj Z
  refine ⟨δ, D, hδ, hD, fun t ht i => ?_⟩
  let X : Idx n → EucSpace d := fun k => expTime V t (Z t k)
  let p : Idx n → ℝ := attentionMatrix Q K X i
  let u : Idx n → ℝ := fun k => inner (𝕜 := ℝ)
    (A (Real.exp (-lam * t) • X i)) (A (Real.exp (-lam * t) • X k))
  let w : Idx n → ℝ := fun k => inner (𝕜 := ℝ) (A (proj (Z t i))) (A (proj (Z t k)))
  have hp : ∀ k, Perspective.softmaxWeight (fun l => Real.exp (2 * lam * t) * u l) k = p k :=
    fun k => (attentionMatrix_eq_normalized_softmax Q K A hA X lam t i k).symm
  have hsum : ∑ k : Idx n, p k = 1 := sum_attentionMatrix (Fin.pos_iff_nonempty.mpr ⟨i⟩) _ _ _ _
  refine ⟨2 * lam * ∑ k : Idx n, p k * (w k - w i), ?_, fun j => ?_⟩
  · simpa only [map_sub, inner_sub_right, w, p, X] using
      hZ.hasDerivAt_norm_sq_projected Q K V proj A F G lam hV hproj Z t i
  · have haverage := score_sub_card_div_sub_error_le_softmax_average u w
      (Real.exp (2 * lam * t)) (Real.exp_pos _) (D * Real.exp (-δ * t))
      (fun k => herror t ht i k) j
    simp_rw [hp] at haverage
    have hid : ∑ k : Idx n, p k * (w k - w i) = (∑ k : Idx n, p k * w k) - w i := by
      simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hsum, one_mul]
    rw [map_sub, inner_sub_right, hid]
    apply mul_le_mul_of_nonneg_left _ (mul_nonneg (show (0 : ℝ) ≤ 2 by norm_num) hV.2.2.2.1.le)
    change w j - w i - (n : ℝ) / Real.exp (2 * lam * t) -
      2 * D * Real.exp (-δ * t) ≤ (∑ k : Idx n, p k * w k) - w i
    linarith

/-- The root, full paranormal splitting, projection, and actual dynamics
hold simultaneously. Source: arXiv:2305.05465v6, §10, Step 2'. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionRoot (1 : ParamMatrix 1) 1 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, isAttentionRoot_id 1,
    rescaledDynamics_one_const 1 1 _⟩

end Transformer.Clusters

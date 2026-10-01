/-
# Operator bounds and energy dissipation for modulated gradient descent

Appendix D.1 of arXiv:2510.22026v2: quadratic matrix bounds give an angle
condition, and the energy of a compact gradient trajectory has a finite limit.
-/

import Transformer.Normalization.Basic
import Mathlib.Analysis.InnerProductSpace.Rayleigh
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.FDeriv.Analytic
import Mathlib.Topology.Order.MonotoneConvergence

namespace Transformer.Normalization

/-- The non-strict quadratic bounds in Appendix D.1, `lem: loj`, control the
norm of the symmetric modulation operator, including in dimension zero. -/
theorem modulated_operator_norm_le {N : ℕ} (A : ParamMatrix N) (B : ℝ)
    (hB : 0 ≤ B)
    (hsymm : ∀ v w : EucSpace N,
      inner (𝕜 := ℝ) (A v) w = inner (𝕜 := ℝ) v (A w))
    (hlower : ∀ v : EucSpace N, 0 ≤ inner (𝕜 := ℝ) (A v) v)
    (hupper : ∀ v : EucSpace N, inner (𝕜 := ℝ) (A v) v ≤ B * ‖v‖ ^ 2) :
    ‖A‖ ≤ B := by
  rw [A.norm_eq_iSup_rayleighQuotient (by
    intro v w
    exact hsymm v w)]
  apply ciSup_le
  intro v
  change |inner (𝕜 := ℝ) (A v) v / ‖v‖ ^ 2| ≤ B
  rw [abs_of_nonneg (div_nonneg (hlower v) (sq_nonneg _))]
  by_cases hv : v = 0
  · simpa [hv] using hB
  · exact (div_le_iff₀ (by positivity : 0 < ‖v‖ ^ 2)).2 (hupper v)

/-- The hypotheses of the operator estimate hold for the identity, with
upper bound one; see Appendix D.1 of arXiv:2510.22026v2. -/
example : (0 : ℝ) ≤ 1 ∧
    (∀ v w : EucSpace 1, inner (𝕜 := ℝ) v w = inner (𝕜 := ℝ) v w) ∧
    (∀ v : EucSpace 1, 0 ≤ inner (𝕜 := ℝ) v v) ∧
    (∀ v : EucSpace 1, inner (𝕜 := ℝ) v v ≤ 1 * ‖v‖ ^ 2) := by
  refine ⟨zero_le_one, fun _ _ => rfl, fun v => ?_, fun v => ?_⟩
  · simpa only [real_inner_self_eq_norm_sq] using sq_nonneg ‖v‖
  · simp only [real_inner_self_eq_norm_sq, one_mul, le_refl]

/-- The quadratic bounds of Appendix D.1, `lem: loj`, give the angle
condition `‖M g‖ ‖g‖ ≤ C ⟨M g,g⟩`. This cancels the possibly irregular
time-dependent factor `λ`, without differentiating its primitive. -/
theorem modulated_angle {N : ℕ} (A : ParamMatrix N) (lam C : ℝ)
    (hlam : 0 < lam) (hC : 0 ≤ C)
    (hsymm : ∀ v w : EucSpace N,
      inner (𝕜 := ℝ) (A v) w = inner (𝕜 := ℝ) v (A w))
    (hlower : ∀ v : EucSpace N, lam * ‖v‖ ^ 2 ≤ inner (𝕜 := ℝ) (A v) v)
    (hupper : ∀ v : EucSpace N, inner (𝕜 := ℝ) (A v) v ≤ C * lam * ‖v‖ ^ 2)
    (g : EucSpace N) :
    ‖A g‖ * ‖g‖ ≤ C * inner (𝕜 := ℝ) (A g) g := by
  have hnorm := modulated_operator_norm_le A (C * lam) (mul_nonneg hC hlam.le)
    hsymm (fun v => (mul_nonneg hlam.le (sq_nonneg _)).trans (hlower v)) hupper
  calc
    ‖A g‖ * ‖g‖ ≤ (C * lam * ‖g‖) * ‖g‖ :=
      mul_le_mul_of_nonneg_right
        ((A.le_opNorm g).trans (mul_le_mul_of_nonneg_right hnorm (norm_nonneg _)))
        (norm_nonneg _)
    _ = C * (lam * ‖g‖ ^ 2) := by ring
    _ ≤ C * inner (𝕜 := ℝ) (A g) g := mul_le_mul_of_nonneg_left (hlower g) hC

/-- The angle hypotheses hold for `M = I`, `λ = 1`, `C = 1`;
Appendix D.1 of arXiv:2510.22026v2. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ v : EucSpace 1, (1 : ℝ) * ‖v‖ ^ 2 ≤ inner (𝕜 := ℝ) v v) ∧
    (∀ v : EucSpace 1, inner (𝕜 := ℝ) v v ≤ 1 * 1 * ‖v‖ ^ 2) := by
  simp only [zero_lt_one, zero_le_one, one_mul, real_inner_self_eq_norm_sq,
    le_refl, implies_true, and_self]

/-- Chain rule for the modulated gradient equation in Appendix D.1,
`lem: loj`: the energy derivative is the negative modulation quadratic form.
Differentiability of the energy at the current state suffices. -/
theorem modulated_energy_hasDerivAt {N : ℕ} (E : EucSpace N → ℝ)
    (x : ℝ → EucSpace N) (M : ℝ → ParamMatrix N) (t : ℝ)
    (hE : DifferentiableAt ℝ E (x t))
    (hx : HasDerivAt x (-(M t (gradient E (x t)))) t) :
    HasDerivAt (fun s => E (x s))
      (-inner (𝕜 := ℝ) (M t (gradient E (x t))) (gradient E (x t))) t := by
  simpa only [Function.comp_def, ← inner_gradient_left, inner_neg_right,
    real_inner_comm] using hE.hasFDerivAt.comp_hasDerivAt t hx

/-- The energy derivative hypotheses are satisfied by the constant flow
and constant energy in Appendix D.1 of arXiv:2510.22026v2. -/
example (t : ℝ) :
    DifferentiableAt ℝ (fun _ : EucSpace 1 => (0 : ℝ)) 0 ∧
      HasDerivAt (fun _ : ℝ => (0 : EucSpace 1))
        (-(ContinuousLinearMap.id ℝ (EucSpace 1)
          (gradient (fun _ : EucSpace 1 => (0 : ℝ)) 0))) t := by
  exact ⟨differentiableAt_const 0, by simpa using hasDerivAt_const t (0 : EucSpace 1)⟩

/-- Nonnegative modulation quadratic forms imply energy monotonicity,
as in Appendix D.1, Step 2, of arXiv:2510.22026v2. The proof uses the
mean value theorem and does not assume continuity of the modulation. -/
theorem modulated_energy_antitoneOn {N : ℕ} (E : EucSpace N → ℝ)
    (x : ℝ → EucSpace N) (M : ℝ → ParamMatrix N) (a : ℝ)
    (hE : ∀ t, a ≤ t → DifferentiableAt ℝ E (x t))
    (hx : ∀ t, a ≤ t → HasDerivAt x (-(M t (gradient E (x t)))) t)
    (hM : ∀ t, a ≤ t → 0 ≤
      inner (𝕜 := ℝ) (M t (gradient E (x t))) (gradient E (x t))) :
    AntitoneOn (fun t => E (x t)) (Set.Ici a) := by
  have hd t ht := modulated_energy_hasDerivAt E x M t (hE t ht) (hx t ht)
  apply antitoneOn_of_deriv_nonpos (convex_Ici a)
  · intro t ht
    exact (hd t ht).continuousAt.continuousWithinAt
  · intro t ht
    exact (hd t (interior_subset ht)).differentiableAt.differentiableWithinAt
  · intro t ht
    rw [(hd t (interior_subset ht)).deriv]
    exact neg_nonpos.mpr (hM t (interior_subset ht))

/-- The monotonicity hypotheses hold for a constant energy and flow;
Appendix D.1 of arXiv:2510.22026v2. -/
example :
    (∀ t : ℝ, 0 ≤ t → DifferentiableAt ℝ (fun _ : EucSpace 1 => (0 : ℝ)) 0) ∧
    (∀ t : ℝ, 0 ≤ t → HasDerivAt (fun _ : ℝ => (0 : EucSpace 1)) 0 t) ∧
    (∀ t : ℝ, 0 ≤ t → 0 ≤ inner (𝕜 := ℝ) (0 : EucSpace 1) 0) := by
  exact ⟨fun _ _ => differentiableAt_const 0, fun t _ => hasDerivAt_const t 0,
    fun _ _ => by simp⟩

/-- A continuous energy on a compact set containing an antitone path
has a finite limiting value. This is the bounded-energy step in
Appendix D.1, Step 2, of arXiv:2510.22026v2. -/
theorem compact_energy_limit {N : ℕ} (E : EucSpace N → ℝ)
    (x : ℝ → EucSpace N) (K : Set (EucSpace N)) (a : ℝ)
    (hK : IsCompact K) (hE : ContinuousOn E K)
    (hxK : ∀ t, a ≤ t → x t ∈ K)
    (hanti : AntitoneOn (fun t => E (x t)) (Set.Ici a)) :
    ∃ L : ℝ, Filter.Tendsto (fun t => E (x t)) Filter.atTop (nhds L) ∧
      ∀ t, a ≤ t → L ≤ E (x t) := by
  let f : ℝ → ℝ := fun t => E (x (max a t))
  have hmono : Antitone f := by
    intro s t hst
    exact hanti (show max a s ∈ Set.Ici a from le_max_left a s)
      (show max a t ∈ Set.Ici a from le_max_left a t) (max_le_max_left a hst)
  have hbdd : BddBelow (Set.range f) := by
    apply (hK.image_of_continuousOn hE).bddBelow.mono
    rintro _ ⟨t, rfl⟩
    exact ⟨x (max a t), hxK _ (le_max_left _ _), rfl⟩
  refine ⟨⨅ t, f t, ?_, ?_⟩
  · apply (tendsto_atTop_ciInf hmono hbdd).congr'
    filter_upwards [Filter.eventually_ge_atTop a] with t ht
    simp [f, max_eq_right ht]
  · intro t ht
    simpa [f, max_eq_right ht] using ciInf_le hbdd t

/-- The compact-energy-limit hypotheses hold for the zero trajectory
in the singleton compact set; Appendix D.1 of arXiv:2510.22026v2. -/
example : IsCompact ({0} : Set (EucSpace 1)) ∧
    ContinuousOn (fun _ : EucSpace 1 => (0 : ℝ)) {0} ∧
    (∀ t : ℝ, 0 ≤ t → (0 : EucSpace 1) ∈ ({0} : Set (EucSpace 1))) ∧
    AntitoneOn (fun _ : ℝ => (0 : ℝ)) (Set.Ici 0) := by
  exact ⟨isCompact_singleton, continuousOn_const, fun _ _ => Set.mem_singleton 0,
    fun _ _ _ _ _ => le_rfl⟩

end Transformer.Normalization

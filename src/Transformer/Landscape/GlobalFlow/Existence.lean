module

public import Transformer.Landscape.GlobalFlow.Basic

/-!
# Global flows of autonomous ODEs — existence and uniqueness for all time

An autonomous field `F` on a Banach space, Lipschitz on every ball and of
linear growth `‖F x‖ ≤ C ‖x‖`, has exactly one solution through each point,
and it is defined on all of `ℝ`. On a finite-dimensional space a locally
Lipschitz field is Lipschitz on every ball (`exists_global`,
`eq_of_hasDerivAt`).

The proof: on `[-T, T]`, Picard–Lindelöf applies to `F ∘ ballRetract R`,
which is Lipschitz and bounded everywhere; by Grönwall that solution never
leaves the ball of radius `‖x₀‖ e^{C T} < R`, where the retraction is the
identity, so it solves `x' = F x` (`exists_local`). Local uniqueness
(`eqOn_Ioo`) glues the solutions on `[-N-1, N+1]` into one on `ℝ`.

Source: the standard proof of global existence under linear growth, as used by
arXiv:2305.05465v6, `p:wellposedparticles`.
-/

@[expose] public section

open Set Metric Filter Topology
open scoped NNReal

namespace Transformer
namespace Landscape.GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

omit [NormedSpace ℝ E] in
/-- A locally Lipschitz map is Lipschitz on every closed ball. -/
theorem exists_lipschitzOnWith_closedBall [ProperSpace E] {F : E → E} (hF : LocallyLipschitz F)
    (R : ℝ) : ∃ K, LipschitzOnWith K F (closedBall 0 R) :=
  (hF.locallyLipschitzOn (s := closedBall 0 R)).exists_lipschitzOnWith_of_compact
    (isCompact_closedBall 0 R)

/-- **Uniqueness on an interval.**  Two solutions of `x' = F x` on `(a, b)`
that stay in a common ball and agree at one time agree on `(a, b)`. -/
theorem eqOn_Ioo {F : E → E} (hF : ∀ R, ∃ K, LipschitzOnWith K F (closedBall 0 R)) {f g : ℝ → E}
    {a b t₀ R : ℝ} (ht₀ : t₀ ∈ Ioo a b)
    (hf : ∀ t ∈ Ioo a b, HasDerivAt f (F (f t)) t ∧ ‖f t‖ ≤ R)
    (hg : ∀ t ∈ Ioo a b, HasDerivAt g (F (g t)) t ∧ ‖g t‖ ≤ R) (heq : f t₀ = g t₀) :
    EqOn f g (Ioo a b) := by
  obtain ⟨K, hK⟩ := hF R
  exact ODE_solution_unique_of_mem_Ioo (v := fun _ => F) (s := fun _ => closedBall 0 R)
    (fun _ _ => hK) ht₀
    (fun t ht => ⟨(hf t ht).1, mem_closedBall_zero_iff.2 (hf t ht).2⟩)
    (fun t ht => ⟨(hg t ht).1, mem_closedBall_zero_iff.2 (hg t ht).2⟩) heq

section Complete

variable [CompleteSpace E]

/-- **Existence on `(-T, T)`**, with the a priori bound. -/
theorem exists_local {F : E → E} (hF : ∀ R, ∃ K, LipschitzOnWith K F (closedBall 0 R)) {C : ℝ}
    (hC : 0 ≤ C) (hFC : ∀ x, ‖F x‖ ≤ C * ‖x‖) (x₀ : E) {T : ℝ} (hT : 0 < T) :
    ∃ α : ℝ → E, α 0 = x₀ ∧ ∀ t ∈ Ioo (-T) T,
      HasDerivAt α (F (α t)) t ∧ ‖α t‖ ≤ ‖x₀‖ * Real.exp (C * T) := by
  set M := ‖x₀‖ * Real.exp (C * T)
  set R := M + 1
  have hR : 0 < R := by positivity
  obtain ⟨K, hK⟩ := hF R
  set G : E → E := fun x => F (ballRetract R x)
  have hmaps : MapsTo (ballRetract R) univ (closedBall (0 : E) R) := fun x _ =>
    mem_closedBall_zero_iff.2 (norm_ballRetract_le hR x)
  have hGL : LipschitzWith (K * 2) G :=
    lipschitzOnWith_univ.1 (hK.comp (lipschitzWith_ballRetract hR).lipschitzOnWith hmaps)
  have hGC : ∀ x, ‖G x‖ ≤ C * ‖x‖ := fun x =>
    (hFC _).trans (mul_le_mul_of_nonneg_left (norm_ballRetract_le_norm hR x) hC)
  have hGb : ∀ x, ‖G x‖ ≤ C * R := fun x =>
    (hFC _).trans (mul_le_mul_of_nonneg_left (norm_ballRetract_le hR x) hC)
  have h0 : (0 : ℝ) ∈ Icc (-T) T := ⟨by linarith, hT.le⟩
  have hPL : IsPicardLindelof (fun _ => G) (tmin := -T) (tmax := T) ⟨0, h0⟩ x₀
      (C * R * T).toNNReal 0 (C * R).toNNReal (K * 2) :=
    IsPicardLindelof.of_time_independent
      (fun x _ => by rw [Real.coe_toNNReal _ (by positivity)]; exact hGb x)
      (hGL.lipschitzOnWith)
      (by
        rw [Real.coe_toNNReal _ (by positivity), Real.coe_toNNReal _ (by positivity)]
        simp only [NNReal.coe_zero, sub_zero, zero_sub, neg_neg, max_self]
        exact le_rfl)
  obtain ⟨α, hα0, hα⟩ := hPL.exists_eq_forall_mem_Icc_hasDerivWithinAt₀
  have hbound := norm_le_of_linearGrowth hC hGC hα
  rw [hα0] at hbound
  refine ⟨α, hα0, fun t ht => ?_⟩
  have htc : t ∈ Icc (-T) T := Ioo_subset_Icc_self ht
  have hαR : ballRetract R (α t) = α t :=
    ballRetract_of_norm_le hR ((hbound t htc).trans (by linarith))
  refine ⟨?_, hbound t htc⟩
  have := (hα t htc).hasDerivAt (Icc_mem_nhds ht.1 ht.2)
  simpa only [G, hαR] using this

/-- **Global existence** for a field of linear growth, Lipschitz on every ball. -/
theorem exists_global_of_lipschitzOnWith {F : E → E}
    (hF : ∀ R, ∃ K, LipschitzOnWith K F (closedBall 0 R)) {C : ℝ}
    (hC : 0 ≤ C) (hFC : ∀ x, ‖F x‖ ≤ C * ‖x‖) (x₀ : E) :
    ∃ γ : ℝ → E, γ 0 = x₀ ∧ ∀ t, HasDerivAt γ (F (γ t)) t := by
  choose α hα0 hα using fun N : ℕ =>
    exists_local hF hC hFC x₀ (T := N + 1) (by positivity)
  -- Any two of the local solutions agree where both are defined.
  have hagree : ∀ N N' : ℕ, ∀ s : ℝ, |s| < min (N : ℝ) N' + 1 → α N s = α N' s := by
    intro N N' s hs
    have hmin : min (N : ℝ) N' + 1 ≤ N + 1 ∧ min (N : ℝ) N' + 1 ≤ N' + 1 :=
      ⟨by simp, by simp⟩
    have hsub : ∀ M : ℕ, min (N : ℝ) N' + 1 ≤ M + 1 →
        Ioo (-(min (N : ℝ) N' + 1)) (min (N : ℝ) N' + 1) ⊆ Ioo (-((M : ℝ) + 1)) (M + 1) :=
      fun M hM => Ioo_subset_Ioo (by linarith) hM
    have hm0 : (0 : ℝ) ≤ min (N : ℝ) N' := le_min (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    refine eqOn_Ioo hF (a := -(min (N : ℝ) N' + 1)) (b := min (N : ℝ) N' + 1) (t₀ := 0)
      (R := ‖x₀‖ * Real.exp (C * (max (N : ℝ) N' + 1))) ⟨by linarith, by linarith⟩
      (fun t ht => ⟨(hα N t (hsub N hmin.1 ht)).1, (hα N t (hsub N hmin.1 ht)).2.trans ?_⟩)
      (fun t ht => ⟨(hα N' t (hsub N' hmin.2 ht)).1, (hα N' t (hsub N' hmin.2 ht)).2.trans ?_⟩)
      ((hα0 N).trans (hα0 N').symm) (abs_lt.1 hs)
    all_goals
      refine mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left ?_ hC))
        (norm_nonneg _)
      simp
  refine ⟨fun t => α ⌈|t|⌉₊ t, by simpa using hα0 0, fun t => ?_⟩
  set N := ⌈|t|⌉₊
  have htN : |t| < N + 1 := (Nat.le_ceil _).trans_lt (by linarith)
  have hIoo : t ∈ Ioo (-((N : ℝ) + 1)) (N + 1) :=
    ⟨by linarith [neg_abs_le t], by linarith [le_abs_self t]⟩
  have heq : (fun t => α ⌈|t|⌉₊ t) =ᶠ[𝓝 t] α N := by
    filter_upwards [Ioo_mem_nhds hIoo.1 hIoo.2] with s hs
    refine hagree _ _ s ?_
    rw [← min_add_add_right]
    refine lt_min_iff.2 ⟨(Nat.le_ceil _).trans_lt (by linarith), abs_lt.2 ⟨by linarith [hs.1], hs.2⟩⟩
  have hd := (hα N t hIoo).1
  rw [show α N t = α ⌈|t|⌉₊ t from rfl] at hd
  exact hd.congr_of_eventuallyEq heq

end Complete

/-- **Global uniqueness**: two solutions on `ℝ` of a field Lipschitz on every
ball that agree at `0` are equal. -/
theorem eq_of_hasDerivAt_of_lipschitzOnWith {F : E → E}
    (hF : ∀ R, ∃ K, LipschitzOnWith K F (closedBall 0 R)) {f g : ℝ → E}
    (hf : ∀ t, HasDerivAt f (F (f t)) t) (hg : ∀ t, HasDerivAt g (F (g t)) t) (h0 : f 0 = g 0) :
    f = g := by
  funext t
  set T := |t| + 1
  have hfc : Continuous f := continuous_iff_continuousAt.2 fun s => (hf s).continuousAt
  have hgc : Continuous g := continuous_iff_continuousAt.2 fun s => (hg s).continuousAt
  obtain ⟨Rf, hRf⟩ := (isCompact_Icc (a := -T) (b := T)).exists_bound_of_continuousOn
    hfc.continuousOn
  obtain ⟨Rg, hRg⟩ := (isCompact_Icc (a := -T) (b := T)).exists_bound_of_continuousOn
    hgc.continuousOn
  refine eqOn_Ioo hF (a := -T) (b := T) (R := max Rf Rg)
    ⟨by linarith [abs_nonneg t], by linarith [abs_nonneg t]⟩
    (fun s hs => ⟨hf s, (hRf s (Ioo_subset_Icc_self hs)).trans (le_max_left _ _)⟩)
    (fun s hs => ⟨hg s, (hRg s (Ioo_subset_Icc_self hs)).trans (le_max_right _ _)⟩) h0
    ⟨by linarith [neg_abs_le t], by linarith [le_abs_self t]⟩

/-- **Global existence** for a locally Lipschitz field of linear growth on a
finite-dimensional space.

Source: the standard proof of global existence under linear growth, as used by
arXiv:2305.05465v6, `p:wellposedparticles`. -/
theorem exists_global [ProperSpace E] {F : E → E} (hF : LocallyLipschitz F) {C : ℝ}
    (hC : 0 ≤ C) (hFC : ∀ x, ‖F x‖ ≤ C * ‖x‖) (x₀ : E) :
    ∃ γ : ℝ → E, γ 0 = x₀ ∧ ∀ t, HasDerivAt γ (F (γ t)) t :=
  exists_global_of_lipschitzOnWith (exists_lipschitzOnWith_closedBall hF) hC hFC x₀

/-- **Global uniqueness**: two solutions on `ℝ` of a locally Lipschitz field on
a finite-dimensional space that agree at `0` are equal. -/
theorem eq_of_hasDerivAt [ProperSpace E] {F : E → E} (hF : LocallyLipschitz F) {f g : ℝ → E}
    (hf : ∀ t, HasDerivAt f (F (f t)) t) (hg : ∀ t, HasDerivAt g (F (g t)) t) (h0 : f 0 = g 0) :
    f = g :=
  eq_of_hasDerivAt_of_lipschitzOnWith (exists_lipschitzOnWith_closedBall hF) hf hg h0

/-- The hypotheses of the results above are satisfiable: the linear field
`x ↦ x` on `ℝ`, Lipschitz on every ball and of linear growth, with its
stationary solution `0`, in `B̄(0, 0)`, on the interval `(-1, 1)` around `0`. -/
example : LocallyLipschitz (fun x : ℝ => x) ∧ (0 : ℝ) ≤ 1 ∧ (∀ x : ℝ, ‖x‖ ≤ 1 * ‖x‖) ∧
    (∀ R : ℝ, ∃ K, LipschitzOnWith K (fun x : ℝ => x) (closedBall 0 R)) ∧
    (∀ t, HasDerivAt (fun _ : ℝ => (0 : ℝ)) ((fun x : ℝ => x) 0) t) ∧ ‖(0 : ℝ)‖ ≤ 0 ∧
    (0 : ℝ) ∈ Ioo (-1) 1 ∧ (0 : ℝ) < 1 :=
  ⟨LipschitzWith.id.locallyLipschitz, zero_le_one, fun x => by simp,
    fun _ => ⟨1, LipschitzWith.id.lipschitzOnWith⟩, fun t => hasDerivAt_const t (0 : ℝ),
    by simp, ⟨by norm_num, by norm_num⟩, one_pos⟩

end Landscape.GlobalFlow
end Transformer

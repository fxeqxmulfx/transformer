module

public import Transformer.Causal.Section4_SubspaceCovectors

/-!
# The original follower field at a leading consensus prefix

ArXiv:2411.04990v2, §4, `thm2`, and the preceding discussion of critical
manifolds. The limiting original predecessor tuple consists of copies of
a leading unit vector. Its last-row field still uses the original mask,
query and key matrices, value matrix, self term and partition function.
This auxiliary limiting field does not replace a moving original prefix.

Every leading component splits into the actual predecessor mass and a
common scalar coefficient. A negative-overlap critical point has a
strictly positive coefficient, regardless of the query and key matrices.
-/

@[expose] public section

open Real
open scoped BigOperators

namespace Transformer.Causal

variable {d : ℕ}

/-- The actual last original field when all original predecessors are
equal to `u`. Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4,
the limiting critical configurations considered before `thm2`. -/
noncomputable def csaFrozenFollowerField (n : ℕ) (β : ℝ) (Q K V : ParamMatrix d)
    (u x : EucSpace d) : EucSpace d :=
  csaAmbientField β Q K V (Fin.snoc (fun _ : Idx n => u) x : Idx (n + 1) → EucSpace d) (Fin.last n)

/-- The actual self weight in the original last row, retaining all
`n` predecessor terms in its denominator.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, `thm2`. -/
noncomputable def csaFrozenSelfWeight (n : ℕ) (β : ℝ) (Q K : ParamMatrix d)
    (u x : EucSpace d) : ℝ :=
  csaTokenWeight β Q K (Fin.snoc (fun _ : Idx n => u) x : Idx (n + 1) → EucSpace d) (Fin.last n) (Fin.last n)

/-- The original common leading coefficient at a consensus prefix,
including its full original radial average.
Source: arXiv:2411.04990v2, §4, the leading component equation of `thm2`. -/
noncomputable def csaFrozenLeadingCoefficient (n : ℕ) (β : ℝ) (Q K V : ParamMatrix d)
    (lam : ℝ) (u x : EucSpace d) : ℝ :=
  lam * csaFrozenSelfWeight n β Q K u x -
    inner ℝ x (csaAverage β Q K V (Fin.snoc (fun _ : Idx n => u) x : Idx (n + 1) → EucSpace d) (Fin.last n))

/-- Inserting the varying original last token after a constant original
prefix is smooth. Source: arXiv:2411.04990v2, §4, the actual field near
its limiting critical configurations. -/
theorem contDiff_csaConsensusInsertion (n : ℕ) (u : EucSpace d) :
    ContDiff ℝ (⊤ : ℕ∞) (fun x : EucSpace d =>
      (Fin.snoc (fun _ : Idx n => u) x : Idx (n + 1) → EucSpace d)) := by
  apply contDiff_pi.mpr
  intro i
  refine Fin.lastCases ?_ (fun j => ?_) i
  · simp only [Fin.snoc_last]
    convert! (contDiff_id : ContDiff ℝ (⊤ : ℕ∞) (id : EucSpace d → _)) using 1
  · simpa only [Fin.snoc_castSucc] using
      (contDiff_const : ContDiff ℝ (⊤ : ℕ∞) (fun _ : EucSpace d => u))

/-- Varying the actual last token gives a smooth original field.
Source: arXiv:2411.04990v2, §4, the smooth critical-point analysis. -/
theorem contDiff_csaFrozenFollowerField (n : ℕ) (β : ℝ) (Q K V : ParamMatrix d)
    (u : EucSpace d) : ContDiff ℝ (⊤ : ℕ∞) (csaFrozenFollowerField n β Q K V u) := by
  exact ((contDiff_apply ℝ (fun _ : Idx (n + 1) => EucSpace d) (Fin.last n)).comp
    (contDiff_csaAmbientField β Q K V)).comp (contDiff_csaConsensusInsertion n u)

/-- The actual common leading coefficient is smooth in the original
last token. Source: arXiv:2411.04990v2, §4, `thm2`. -/
theorem contDiff_csaFrozenLeadingCoefficient (n : ℕ) (β : ℝ) (Q K V : ParamMatrix d)
    (lam : ℝ) (u : EucSpace d) :
    ContDiff ℝ (⊤ : ℕ∞) (csaFrozenLeadingCoefficient n β Q K V lam u) := by
  exact (contDiff_const.mul
    ((contDiff_csaTokenWeight β Q K (Fin.last n) (Fin.last n)).comp
      (contDiff_csaConsensusInsertion n u))).sub
    (contDiff_id.inner ℝ ((contDiff_csaAverage β Q K V (Fin.last n)).comp
      (contDiff_csaConsensusInsertion n u)))

/-- The actual leading component retains the original predecessor
mass and the original self contribution. This identity holds on the
ambient extension as well as on the sphere.
Source: arXiv:2411.04990v2, §§2–4, the component equation of `thm2`. -/
theorem inner_csaFrozenFollowerField (n : ℕ) (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (v : EucSpace d) (hv : v ∈ L) (u x : EucSpace d) :
    inner ℝ v (csaFrozenFollowerField n β Q K V u x) =
      lam * (1 - csaFrozenSelfWeight n β Q K u x) * inner ℝ v u +
        csaFrozenLeadingCoefficient n β Q K V lam u x * inner ℝ v x := by
  let Y : Idx (n + 1) → EucSpace d := Fin.snoc (fun _ : Idx n => u) x
  have hs := sum_csaTokenWeight β Q K Y (Fin.last n)
  rw [Fin.sum_univ_castSucc] at hs
  have hm : (∑ j : Idx n, csaTokenWeight β Q K Y (Fin.last n) j.castSucc) =
      1 - csaTokenWeight β Q K Y (Fin.last n) (Fin.last n) := by linarith
  have ha : inner ℝ v (csaAverage β Q K V Y (Fin.last n)) =
      (∑ j : Idx n, csaTokenWeight β Q K Y (Fin.last n) j.castSucc) *
        (lam * inner ℝ v u) +
      csaTokenWeight β Q K Y (Fin.last n) (Fin.last n) * (lam * inner ℝ v x) := by
    rw [csaAverage, inner_sum]
    simp only [real_inner_smul_right, inner_value_of_mem_leading V L lam heig hinv v hv]
    rw [Fin.sum_univ_castSucc]
    simp only [Y, Fin.snoc_castSucc, Fin.snoc_last]
    rw [Finset.sum_mul]
  simp only [csaFrozenFollowerField, csaAmbientField, Fin.snoc_last]
  change inner ℝ v (proj d x (csaAverage β Q K V Y (Fin.last n))) = _
  rw [proj, inner_sub_right, real_inner_smul_right, ha, hm]
  change _ = lam * (1 - csaTokenWeight β Q K Y (Fin.last n) (Fin.last n)) * inner ℝ v u +
    (lam * csaTokenWeight β Q K Y (Fin.last n) (Fin.last n) -
      inner ℝ x (csaAverage β Q K V Y (Fin.last n))) * inner ℝ v x
  ring

/-- A nonempty actual prefix has positive total original mass, even
before imposing unit norms. Source: arXiv:2411.04990v2, §4,
positivity of the original weights in the critical-point argument. -/
theorem csaFrozenPredecessorMass_pos (n : ℕ) (hn : 1 ≤ n) (β : ℝ)
    (Q K : ParamMatrix d) (u x : EucSpace d) :
    0 < 1 - csaFrozenSelfWeight n β Q K u x := by
  let Y : Idx (n + 1) → EucSpace d := Fin.snoc (fun _ : Idx n => u) x
  let j : Idx n := ⟨0, hn⟩
  have hp := csaTokenWeight_pos β Q K Y (Fin.last n) j.castSucc (by simp [j])
  have hl := Finset.single_le_sum (fun k _ =>
    csaTokenWeight_nonneg β Q K Y (Fin.last n) k.castSucc) (Finset.mem_univ j)
  have hs := sum_csaTokenWeight β Q K Y (Fin.last n)
  rw [Fin.sum_univ_castSucc] at hs
  change 0 < 1 - csaTokenWeight β Q K Y (Fin.last n) (Fin.last n)
  linarith

/-- A negative-overlap critical point has a positive original leading
coefficient. No derivatives of attention have been discarded.
Source: arXiv:2411.04990v2, §4, the critical-manifold obstruction
preceding `thm2`; this is an auxiliary instability input. -/
theorem csaFrozenLeadingCoefficient_pos_of_critical (n : ℕ) (hn : 1 ≤ n) (β : ℝ)
    (Q K V : ParamMatrix d) (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L) (x : EucSpace d)
    (hcrit : csaFrozenFollowerField n β Q K V u x = 0)
    (hq : inner ℝ (u : EucSpace d) x < 0) :
    0 < csaFrozenLeadingCoefficient n β Q K V lam u x := by
  have h := inner_csaFrozenFollowerField n β Q K V L lam heig hinv u hu u x
  rw [hcrit, inner_zero_right, real_inner_self_eq_norm_sq,
    mem_sphere_zero_iff_norm.mp u.2, one_pow, mul_one] at h
  have hp := mul_pos hlam (csaFrozenPredecessorMass_pos n hn β Q K u x)
  nlinarith

/-- Actual antipodal unit tokens, identity values, and zero queries and
keys jointly realize the spectral, nonempty-prefix and negative critical
point premises. Source: arXiv:2411.04990v2, §4, the exceptional critical
configurations considered in `thm2`. -/
example : ∃ (u : SSphere 2) (x : EucSpace 2), (1 : ℕ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
    (u : EucSpace 2) ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    csaFrozenFollowerField 1 1 0 0 (ContinuousLinearMap.id ℝ _) u x = 0 ∧
    inner ℝ (u : EucSpace 2) x < 0 ∧ ‖x‖ = 1 := by
  let u : SSphere 2 := basePoint 1
  refine ⟨u, -(u : EucSpace 2), le_rfl, one_pos, fun v _ => by simp,
    fun z hz => hz, Submodule.mem_top, ?_, ?_, ?_⟩
  · simp [csaFrozenFollowerField, csaAmbientField, csaAverage, csaTokenWeight,
      csaPartition, Fin.sum_univ_two, Fin.snoc, proj]
  · simp [mem_sphere_zero_iff_norm.mp u.2]
  · rw [norm_neg, mem_sphere_zero_iff_norm.mp u.2]

end Transformer.Causal

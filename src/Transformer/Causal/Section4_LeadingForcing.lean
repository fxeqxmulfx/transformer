module

public import Transformer.Causal.Section4_LeadingComponents

/-!
# Actual leading forcing from a converging original prefix

ArXiv:2411.04990v2, §4, the original component equations of `thm2`.
Every actual leading covector has the same bounded diagonal coefficient.
Along the proposed leading limit, the actual predecessor forcing is
uniformly positive on a tail. Against a covector orthogonal to this
limit, it tends to zero. These assertions use all original weights and
their original denominator; no limit of the attention weights is needed.
-/

@[expose] public section

open Real Filter
open scoped Topology BigOperators

namespace Transformer.Causal

variable {d n : ℕ}

/-- The actual original predecessor forcing for a leading covector.
Source: arXiv:2411.04990v2, §2, `eqn:csa`, and §4, `thm2`. -/
noncomputable def csaLastLeadingForcing (β : ℝ) (Q K : ParamMatrix d) (lam : ℝ)
    (X : ℝ → SphereTuple d (n + 1)) (v : EucSpace d) (t : ℝ) : ℝ :=
  lam * ∑ j : Idx n,
    csaTokenWeight β Q K (fun k => (X t k : EucSpace d)) (Fin.last n) j.castSucc *
      inner ℝ v (X t j.castSucc : EucSpace d)

/-- The exact original last component splits into its actual predecessor
forcing and its actual common radial coefficient. In particular the
self attention term has not been removed. Source: arXiv:2411.04990v2,
§2, `eqn:csa`, and §4, the leading equations of `thm2`. -/
theorem hasDerivAt_csaLast_leading_forcing (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v) (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (v : EucSpace d) (hv : v ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (t : ℝ) :
    HasDerivAt (fun s => inner ℝ v (X s (Fin.last n) : EucSpace d))
      (csaLastLeadingForcing β Q K lam X v t +
        csaLastLeadingCoefficient β Q K V lam X t *
          inner ℝ v (X t (Fin.last n) : EucSpace d)) t :=
  hasDerivAt_csaLast_leading β Q K V L lam heig hinv v hv X hX t

/-- The full actual last leading coefficient is bounded by the leading
eigenvalue magnitude and the original value operator norm.
Source: arXiv:2411.04990v2, §§2–4, the radial coefficient in `thm2`. -/
theorem abs_csaLastLeadingCoefficient_le (β : ℝ) (Q K V : ParamMatrix d) (lam : ℝ)
    (X : ℝ → SphereTuple d (n + 1)) (t : ℝ) :
    |csaLastLeadingCoefficient β Q K V lam X t| ≤ |lam| + ‖V‖ := by
  have hY : ‖fun j => (X t j : EucSpace d)‖ ≤ 1 :=
    (pi_norm_le_iff_of_nonneg zero_le_one).mpr fun j =>
      (mem_sphere_zero_iff_norm.mp (X t j).2).le
  have hr : |csaRadialCoefficient β Q K V X t (Fin.last n)| ≤ ‖V‖ := by
    apply (abs_real_inner_le_norm _ _).trans
    rw [mem_sphere_zero_iff_norm.mp (X t (Fin.last n)).2, one_mul]
    exact (norm_csaAverage_le β Q K V _ _).trans
      (by simpa only [mul_one] using mul_le_mul_of_nonneg_left hY (norm_nonneg V))
  calc
    _ ≤ |lam * csaTokenWeight β Q K (fun j => (X t j : EucSpace d))
          (Fin.last n) (Fin.last n)| + |csaRadialCoefficient β Q K V X t (Fin.last n)| :=
      abs_sub _ _
    _ ≤ |lam| + ‖V‖ := by
      rw [abs_mul, abs_of_nonneg (csaTokenWeight_nonneg β Q K _ _ _)]
      exact add_le_add (by
        simpa only [mul_one] using mul_le_mul_of_nonneg_left
          (csaTokenWeight_le_one β Q K (fun j => (X t j : EucSpace d))
            (Fin.last n) (Fin.last n)) (abs_nonneg lam)) hr

/-- A genuinely converging original prefix gives a uniformly positive
actual longitudinal forcing, for every nonempty prefix and arbitrary
queries and keys. Source: arXiv:2411.04990v2, §4, `thm2`. -/
theorem eventually_csaLastLeadingForcing_pos (β : ℝ) (Q K : ParamMatrix d)
    (lam : ℝ) (hlam : 0 < lam) (X : ℝ → SphereTuple d (n + 1)) (hn : 1 ≤ n)
    (u : SSphere d)
    (hp : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop
      (nhds (u : EucSpace d))) :
    ∃ c : ℝ, 0 < c ∧ ∀ᶠ t in atTop, c ≤ csaLastLeadingForcing β Q K lam X u t := by
  have hh : ∀ᶠ t in atTop, ∀ j : Idx n,
      (1 / 2 : ℝ) ≤ inner ℝ (u : EucSpace d) (X t j.castSucc : EucSpace d) := by
    rw [Filter.eventually_all]
    intro j
    have hj : Tendsto (fun t => inner ℝ (u : EucSpace d) (X t j.castSucc : EucSpace d))
        atTop (nhds (inner ℝ (u : EucSpace d) u)) := tendsto_const_nhds.inner (hp j)
    have hj' : Tendsto (fun t => inner ℝ (u : EucSpace d) (X t j.castSucc : EucSpace d))
        atTop (nhds 1) := by
      simpa only [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp u.2, one_pow] using hj
    exact (hj'.eventually (lt_mem_nhds (by norm_num : (1 / 2 : ℝ) < 1))).mono fun _ h => h.le
  let c := lam * csaWeightFloor β Q K (n + 1) / 2
  refine ⟨c, div_pos (mul_pos hlam (csaWeightFloor_pos β Q K (Nat.zero_lt_succ n)))
    (by norm_num), hh.mono ?_⟩
  intro t ht
  let j : Idx n := ⟨0, hn⟩
  have hw := csaWeightFloor_le β Q K (fun k => (X t k : EucSpace d))
    (fun k => mem_sphere_zero_iff_norm.mp (X t k).2) (Fin.last n) j.castSucc (by simp [j])
  have hs := Finset.single_le_sum (fun k _ => mul_nonneg
    (csaTokenWeight_nonneg β Q K (fun l => (X t l : EucSpace d)) (Fin.last n) k.castSucc)
      ((by norm_num : (0 : ℝ) ≤ 1 / 2).trans (ht k))) (Finset.mem_univ j)
  have hterm := mul_le_mul hw (ht j) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (csaTokenWeight_nonneg β Q K (fun k => (X t k : EucSpace d)) (Fin.last n) j.castSucc)
  have hmul := mul_le_mul_of_nonneg_left (hterm.trans hs) hlam.le
  dsimp [c, csaLastLeadingForcing]
  nlinarith

/-- The actual predecessor forcing against a covector orthogonal to
the original prefix limit tends to zero. A limit for the original
attention weights is unnecessary. Source: arXiv:2411.04990v2, §4,
the leading splitting in `thm2`. -/
theorem tendsto_csaLastLeadingForcing_zero (β : ℝ) (Q K : ParamMatrix d) (lam : ℝ)
    (X : ℝ → SphereTuple d (n + 1)) (u v : EucSpace d) (hv : inner ℝ v u = 0)
    (hp : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop (nhds u)) :
    Tendsto (csaLastLeadingForcing β Q K lam X v) atTop (nhds 0) := by
  have hj (j : Idx n) : Tendsto (fun t =>
      csaTokenWeight β Q K (fun k => (X t k : EucSpace d)) (Fin.last n) j.castSucc *
        inner ℝ v (X t j.castSucc : EucSpace d)) atTop (nhds 0) := by
    have hi : Tendsto (fun t => inner ℝ v (X t j.castSucc : EucSpace d)) atTop (nhds 0) := by
      simpa only [hv] using (tendsto_const_nhds.inner (hp j) :
        Tendsto (fun t => inner ℝ v (X t j.castSucc : EucSpace d)) atTop (nhds (inner ℝ v u)))
    apply squeeze_zero_norm (fun t => ?_) (show Tendsto
      (fun t => |inner ℝ v (X t j.castSucc : EucSpace d)|) atTop (nhds 0) from by
        simpa only [abs_zero] using hi.abs)
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (csaTokenWeight_nonneg β Q K _ _ _)]
    simpa only [one_mul] using mul_le_mul_of_nonneg_right
      (csaTokenWeight_le_one β Q K (fun k => (X t k : EucSpace d))
        (Fin.last n) j.castSucc) (abs_nonneg _)
  have hs := (tendsto_finsetSum Finset.univ (fun j _ => hj j)).const_mul lam
  simp only [Finset.sum_const_zero, mul_zero] at hs
  convert! hs using 1

/-- Actual nonzero consensus and an orthogonal coordinate covector
jointly realize the prefix, nonempty, positive-eigenvalue, and
orthogonality premises. Source: arXiv:2411.04990v2, §4, `thm2`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    ∃ (X : ℝ → SphereTuple 2 3) (u : SSphere 2) (v : EucSpace 2),
      CSA 2 3 β Q K (ContinuousLinearMap.id ℝ _) X ∧
      (0 : ℝ) < 1 ∧ (1 : ℕ) ≤ 2 ∧ inner ℝ v u = 0 ∧ v ≠ 0 ∧
      (∀ w ∈ (⊤ : Submodule ℝ (EucSpace 2)),
        (ContinuousLinearMap.id ℝ (EucSpace 2)) w = (1 : ℝ) • w) ∧
      (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
        (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
      v ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
      (∀ j : Idx 2, Tendsto (fun t => (X t j.castSucc : EucSpace 2)) atTop
        (nhds (u : EucSpace 2))) := by
  let u : SSphere 2 := basePoint 1
  let v : EucSpace 2 := EuclideanSpace.single (1 : Fin 2) 1
  refine ⟨fun _ _ => u, u, v, CSA_const_consensus β Q K u, one_pos, by norm_num,
    ?_, ?_, fun w _ => by simp, fun z hz => hz, Submodule.mem_top,
    fun _ => tendsto_const_nhds⟩
  · simp [u, v, basePoint, EuclideanSpace.inner_single_left]
  · intro hv
    have he := congrArg (fun x : EucSpace 2 => x (1 : Fin 2)) hv
    simp [v] at he

end Transformer.Causal

/-
# Metastability — Separated configurations are exponentially rare on the circle

The low-dimensional claim of §4: `n` i.i.d. uniform points on `𝕊^1` are
`(β, ε)`-separated with probability at most `c^n`, `c < 1`.

The proof here is not the source's heuristic, which it calls one and which
does not bound the probability of a union over the choices of centres.  It
rests on two elementary facts about `γ(β) = 1 - α - 8ε - β⁻¹ log(2n²/ε)`.

* `α ≥ -1`, so `γ(β) ≤ 2 - β⁻¹ log(2n²/ε)`, which is negative as soon as
  `n ≥ e^{2β}`: past that size no configuration is separated at all
  (`not_isSeparated_of_large`).
* Two points with `1 - 8ε ≤ ⟨x_i, x_j⟩ < 1 - 4ε` spoil every choice of
  centres: a single cap of height `1 - ε` forces `⟨x_i, x_j⟩ ≥ 1 - 4ε`, and two
  distinct caps force `α ≥ ⟨x_i, x_j⟩ ≥ 1 - 8ε`, hence `γ(β) ≤ 0`
  (`not_isSeparated_of_inner`).  The event that the first two points fall in
  two fixed small caps realizing such an inner product has a probability
  `p > 0` independent of `n` (`UniformCap`), so the probability of being
  separated is at most `1 - p` for every `n`.

`c = max(1 - p, 1/2)^{1/N}` with `N ≥ e^{2β}` then does it.

Source: arXiv:2410.06833v1, §4, "Low dimension".
-/

import Transformer.Metastability.NotSeparated

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Metastability

variable {d : ℕ}

/-- **Lemma (low-dimensional bound).**  For `β > 0` and `0 < ε < 1/16`, the
probability that `n` i.i.d. uniform points on `𝕊^1` are `(β, ε)`-separated
decays exponentially in `n`: there is `c ∈ (0, 1)`, depending on `β` and `ε`
alone, with

  `ℙ((x_1,…,x_n) is (β, ε)-separated) ≤ c^n`   for every `n ≥ 2`.

**What the source says and what is changed here.**  `β > 0` is the source's
"Fix `β > 0`", made a hypothesis; the statement stood without it and is false
for `β < 0`: then `-β⁻¹ log(2n²/ε) → +∞`, so for large `n` every
configuration is separated with `k = n` centres at the points themselves, and
the probability is `1`.  The source's argument is a heuristic ("we briefly
explain how to heuristically derive this bound"); the proof here is the one in
the module docstring.

Source: arXiv:2410.06833v1, §4, "Low dimension" (the claim). -/
theorem low_dim_decay (β ε : ℝ) (hβ : 0 < β) (hε : 0 < ε) (hε16 : ε < 1 / 16) :
    ∀ σ : ∀ d : ℕ, Measure (SSphere d), IsUniformFamily σ →
      ∃ c : ℝ, 0 < c ∧ c < 1 ∧
        ∀ n : ℕ, 2 ≤ n →
          (iidSphere 2 n (σ 2)).real { X | isSeparated 2 n β ε X } ≤ c ^ n := by
  intro σ hσ
  have hν := hσ 2 (by norm_num)
  have : IsProbabilityMeasure (σ 2) := hν.1
  have hε1 : ε < 1 := by linarith
  -- two points with `⟨e, f⟩ = 1 - 6ε`, the middle of the window
  let e : SSphere 2 := ⟨EuclideanSpace.single (⟨0, by omega⟩ : Fin 2) (1 : ℝ),
    mem_sphere_zero_iff_norm.mpr (by simp [PiLp.norm_single])⟩
  let f : SSphere 2 := ⟨Perspective.spherePt 2 le_rfl (1 - 6 * ε),
    mem_sphere_zero_iff_norm.mpr (Perspective.norm_spherePt 2 le_rfl _
      (abs_le.2 ⟨by linarith, by linarith⟩))⟩
  have hef : inner (𝕜 := ℝ) (e : EucSpace 2) (f : EucSpace 2) = 1 - 6 * ε :=
    Perspective.inner_spherePt 2 le_rfl _
  set C₁ := openCap e (ε ^ 2 / 8)
  set C₂ := openCap f (ε ^ 2 / 8)
  have hη : 0 < ε ^ 2 / 8 := by positivity
  set p := (σ 2).real C₁ * (σ 2).real C₂
  have hp : 0 < p := mul_pos
    (ENNReal.toReal_pos (measure_openCap_pos _ hν e hη).ne' (measure_ne_top _ _))
    (ENNReal.toReal_pos (measure_openCap_pos _ hν f hη).ne' (measure_ne_top _ _))
  -- every `n ≥ 2`: probability at most `1 - p`
  have hsmall : ∀ n : ℕ, 2 ≤ n →
      (iidSphere 2 n (σ 2)).real { X | isSeparated 2 n β ε X } ≤ 1 - p := by
    intro n hn
    let i₀ : Idx n := ⟨0, by omega⟩
    let i₁ : Idx n := ⟨1, by omega⟩
    have hi : i₀ ≠ i₁ := by simp [i₀, i₁, Fin.ext_iff]
    let s : Idx n → Set (SSphere 2) := fun i =>
      if i = i₀ then C₁ else if i = i₁ then C₂ else Set.univ
    have hs : ∀ i, MeasurableSet (s i) := fun i => by
      simp only [s]
      split_ifs
      exacts [(isOpen_openCap _ _).measurableSet, (isOpen_openCap _ _).measurableSet,
        MeasurableSet.univ]
    have hsub : { X | isSeparated 2 n β ε X } ⊆ (Set.pi Set.univ s)ᶜ := by
      intro X hX hE
      have h₀ : X i₀ ∈ C₁ := by simpa [s] using hE i₀ (Set.mem_univ _)
      have h₁ : X i₁ ∈ C₂ := by simpa [s, hi.symm] using hE i₁ (Set.mem_univ _)
      have ha := norm_sub_lt_of_mem_openCap hε h₀
      have hb := norm_sub_lt_of_mem_openCap hε h₁
      have habs := abs_inner_sub_inner_le (X i₀) (X i₁) e f
      rw [hef] at habs
      have := abs_lt.1 (habs.trans_lt (by linarith : _ < ε))
      exact not_isSeparated_of_inner hβ hε hε1 X i₀ i₁ (by linarith) (by linarith) hX
    have hprod : (iidSphere 2 n (σ 2)) (Set.pi Set.univ s) = σ 2 C₁ * σ 2 C₂ := by
      rw [iidSphere, Measure.pi_pi]
      have : ∀ i, σ 2 (s i) =
          (if i = i₀ then σ 2 C₁ else 1) * (if i = i₁ then σ 2 C₂ else 1) := by
        intro i
        by_cases h₀ : i = i₀ <;> by_cases h₁ : i = i₁
        · exact absurd (h₀.symm.trans h₁) hi
        · simp [s, h₀, hi]
        · simp [s, h₁, hi.symm]
        · simp [s, h₀, h₁]
      simp_rw [this, Finset.prod_mul_distrib]
      simp
    have : IsProbabilityMeasure (iidSphere 2 n (σ 2)) := by
      unfold iidSphere; infer_instance
    calc (iidSphere 2 n (σ 2)).real { X | isSeparated 2 n β ε X }
        ≤ (iidSphere 2 n (σ 2)).real (Set.pi Set.univ s)ᶜ := measureReal_mono hsub
      _ = 1 - p := by
        rw [measureReal_compl (MeasurableSet.univ_pi hs), probReal_univ,
          measureReal_def, hprod, ENNReal.toReal_mul]
        rfl
  -- `n ≥ N`: probability zero
  set N : ℕ := ⌈Real.exp (2 * β)⌉₊ + 2
  have hN : 0 < N := by omega
  set q := max (1 - p) (1 / 2)
  have hq0 : 0 < q := lt_max_of_lt_right (by norm_num)
  have hq1 : q < 1 := max_lt (by linarith) (by norm_num)
  have hc0 : 0 < q ^ ((N : ℝ)⁻¹) := Real.rpow_pos_of_pos hq0 _
  have hc1 : q ^ ((N : ℝ)⁻¹) < 1 :=
    Real.rpow_lt_one hq0.le hq1 (inv_pos.2 (by exact_mod_cast hN))
  refine ⟨q ^ ((N : ℝ)⁻¹), hc0, hc1, fun n hn => ?_⟩
  rcases le_or_gt N n with hNn | hnN
  · have hexp : Real.exp (2 * β) ≤ n := by
      have := Nat.le_ceil (Real.exp (2 * β))
      have : (⌈Real.exp (2 * β)⌉₊ : ℝ) ≤ n := by exact_mod_cast (by omega : _ ≤ n)
      linarith
    have hempty : { X | isSeparated 2 n β ε X } = ∅ :=
      Set.eq_empty_of_forall_notMem fun X => not_isSeparated_of_large hβ hε hε1 hexp X
    rw [hempty, measureReal_empty]
    positivity
  · calc (iidSphere 2 n (σ 2)).real { X | isSeparated 2 n β ε X }
        ≤ 1 - p := hsmall n hn
      _ ≤ q := le_max_left _ _
      _ = (q ^ ((N : ℝ)⁻¹)) ^ N := (Real.rpow_inv_natCast_pow hq0.le hN.ne').symm
      _ ≤ (q ^ ((N : ℝ)⁻¹)) ^ n := pow_le_pow_of_le_one hc0.le hc1.le hnN.le

/-- The hypotheses of `low_dim_decay` carried in its binders are satisfiable:
`β = 1`, `ε = 1/32`.  The uniform family is pinned down by `IsUniformFamily`,
not constructed; see `InitialUniform`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 32 ∧ (1 : ℝ) / 32 < 1 / 16 := by norm_num

end Metastability
end Transformer

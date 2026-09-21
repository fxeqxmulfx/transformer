/-
# Mean-Field Dynamics — `thm: agazzi_merge` is false as printed (§5 of 2512.01868v4)

The survey states, after Bruno, Pasqualotto & Agazzi, that for a discrete
multi-cluster start `eq: init_clust` with a unique closest pair `(ī, j̄)`, the
trajectories of `eq: SA` in the rescaled time `dt = e^{β(1 - ⟨x_ī(0), x_j̄(0)⟩)} ds`
converge as `β → ∞`, uniformly on any `[0, T]` on which the limit pair stays
`ε`-apart, to the pairing dynamics `hardmaxPair`.

`not_agazzi_merge` refutes this with the source's own hypotheses:
`eq: init_clust` allows `α_j ≥ 0`, and at `α = (0, 1)` on a great circle the
cluster of mass `1` attends only to itself and never moves, while the pairing
dynamics moves it at once.  The limit system does not see the masses; `eq: SA`
does.

The masses are not the only obstruction.  With `α_ī, α_j̄ > 0` the pair's
rescaled speed is `(α_j̄ / α_ī) e^{β(ρ(s) - ρ(0))}` to leading order, with
`ρ = ⟨x_ī, x_j̄⟩`: it carries the mass ratio, which the limit system lacks, and
it grows without bound in `β` as soon as `ρ` rises above `ρ(0)`, so the rescaled
pair closes in a time of order `1/β` rather than following `hardmaxPair`.  That
argument is not formalized here; no positive-mass restatement is put on the
books, since the survey's paraphrase is the only source in `papers/` and the
original statement of Bruno, Pasqualotto & Agazzi is not.

The condition `⟨y_ī(s), y_j̄(s)⟩ ≤ 1 - ε` is read on the limit trajectory, as in
`Merging`: on the `β`-dependent trajectory it would make `[0, T_ε]` depend on `β`.
-/

import Transformer.MeanField.MergingPaths

open scoped BigOperators
open Real

namespace Transformer
namespace MeanField

variable (d : ℕ)

/-- **`thm: agazzi_merge` is false as printed.**

In every dimension `d ≥ 2`, with `K = 2` clusters of masses `α = (0, 1)` at
orthonormal `f, e` — the only pair, so the unique closest one — and `ε = 1/2`,
`T = 1/4`: the heavy cluster is stationary under `eq: SA` for every `β`, while
the pairing dynamics `hardmaxPair` moves it to
`(cosh ½)^{-1/2}(sinh ¼ · f + cosh ¼ · e) ≠ e` by `s = 1/4`; so no `β` brings
the rescaled trajectory within `δ = ‖e - y_j̄(1/4)‖` of the limit.  Both
trajectories are the closed forms of `MeanField.MergingPaths`.

Source: arXiv:2512.01868v4, §5, `thm: agazzi_merge` and `eq: init_clust`. -/
theorem not_agazzi_merge (hd : 2 ≤ d) :
    ¬ ∀ (K : ℕ) (α : Idx K → ℝ) (X₀ : SphereTuple d K),
      (∀ j : Idx K, 0 ≤ α j) → ∑ j : Idx K, α j = 1 →
      ∀ ibar jbar : Idx K, ibar ≠ jbar →
      (∀ i j : Idx K, i ≠ j → (i, j) ≠ (ibar, jbar) → (i, j) ≠ (jbar, ibar) →
        inner (𝕜 := ℝ) ((X₀ i : EucSpace d)) ((X₀ j : EucSpace d))
          < inner (𝕜 := ℝ) ((X₀ ibar : EucSpace d)) ((X₀ jbar : EucSpace d))) →
      ∀ ε T : ℝ, 0 < ε → 0 < T →
      ∀ Y : ℝ → SphereTuple d K, Y 0 = X₀ → hardmaxPair d K ibar jbar Y →
        (∀ s ∈ Set.Icc (0 : ℝ) T,
          inner (𝕜 := ℝ) ((Y s ibar : EucSpace d)) ((Y s jbar : EucSpace d)) ≤ 1 - ε) →
        ∀ X : ℝ → ℝ → SphereTuple d K,
          (∀ β : ℝ, 0 < β → X β 0 = X₀ ∧ clusterSA d K α β (X β)) →
          ∀ δ : ℝ, 0 < δ → ∃ B : ℝ, ∀ β : ℝ, B < β →
            ∀ s ∈ Set.Icc (0 : ℝ) T, ∀ k : Idx K,
              ‖((X β (Real.exp (β * (1 - inner (𝕜 := ℝ)
                      ((X₀ ibar : EucSpace d)) ((X₀ jbar : EucSpace d)))) * s) k : EucSpace d))
                  - ((Y s k : EucSpace d))‖ < δ := by
  intro h
  set e : EucSpace d := EuclideanSpace.single ⟨0, by omega⟩ 1
  set f : EucSpace d := EuclideanSpace.single ⟨1, by omega⟩ 1
  have he : ‖e‖ = 1 := by simp [e]
  have hf : ‖f‖ = 1 := by simp [f]
  have hef : inner (𝕜 := ℝ) e f = 0 := by
    simp [e, f, EuclideanSpace.inner_single_left]
  have hfe : inner (𝕜 := ℝ) f e = 0 := by rw [real_inner_comm]; exact hef
  have hee : inner (𝕜 := ℝ) e e = 1 := by rw [real_inner_self_eq_norm_sq, he, one_pow]
  -- the paths
  set r : ℝ → ℝ := fun s => (√(cosh (2 * s)))⁻¹
  have hr : ∀ s, r s ^ 2 * (sinh s ^ 2 + cosh s ^ 2) = 1 := fun s => by
    have hc : 0 < cosh (2 * s) := cosh_pos _
    have := Real.sq_sqrt hc.le
    rw [cosh_two_mul] at this hc
    simp only [r, inv_pow, cosh_two_mul, this]
    field_simp
    ring
  have hm : ∀ t, (cosh t)⁻¹ ^ 2 * (sinh t ^ 2 + 1 ^ 2) = 1 := fun t => by
    have := cosh_sq t
    field_simp
    linarith
  have hY0 : ∀ s, r s • (sinh s • e + cosh s • f) ∈ Metric.sphere (0 : EucSpace d) 1 :=
    fun s => mem_sphere_zero_iff_norm.2 (norm_frame d e f he hf hef _ _ _ (hr s))
  have hY1 : ∀ s, r s • (sinh s • f + cosh s • e) ∈ Metric.sphere (0 : EucSpace d) 1 :=
    fun s => mem_sphere_zero_iff_norm.2 (norm_frame d f e hf he hfe _ _ _ (hr s))
  have hXm : ∀ t, (cosh t)⁻¹ • (sinh t • e + f) ∈ Metric.sphere (0 : EucSpace d) 1 :=
    fun t => mem_sphere_zero_iff_norm.2 (by
      simpa using norm_frame d e f he hf hef _ (sinh t) 1 (hm t))
  have heS : e ∈ Metric.sphere (0 : EucSpace d) 1 := mem_sphere_zero_iff_norm.2 he
  let Yt : ℝ → SphereTuple d 2 := fun s => ![⟨_, hY0 s⟩, ⟨_, hY1 s⟩]
  let Xt : ℝ → SphereTuple d 2 := fun t => ![⟨_, hXm t⟩, ⟨e, heS⟩]
  have hXY : Xt 0 = Yt 0 := by
    funext k
    fin_cases k <;> apply Subtype.ext <;> simp [Xt, Yt, r]
  -- the pairing dynamics
  have hhard : hardmaxPair d 2 0 1 Yt := by
    intro s k
    fin_cases k
    · have := hasDerivAt_pairPath d e f he hf hef s
      rw [add_comm (cosh s • e)] at this
      simpa [Yt, r] using this
    · have := hasDerivAt_pairPath d f e hf he hfe s
      rw [add_comm (cosh s • f)] at this
      simpa [Yt, r] using this
  have hcond : ∀ s ∈ Set.Icc (0 : ℝ) (1 / 4),
      inner (𝕜 := ℝ) ((Yt s 0 : EucSpace d)) ((Yt s 1 : EucSpace d)) ≤ 1 - 1 / 2 := by
    rintro s ⟨-, hs⟩
    have hin : inner (𝕜 := ℝ) ((Yt s 0 : EucSpace d)) ((Yt s 1 : EucSpace d))
        = r s ^ 2 * (2 * sinh s * cosh s) := by
      change inner (𝕜 := ℝ) (r s • (sinh s • e + cosh s • f))
        (r s • (sinh s • f + cosh s • e)) = _
      rw [add_comm (sinh s • f), real_inner_smul_left, real_inner_smul_right,
        inner_frame d e f he hf hef]
      ring
    rw [hin]
    calc r s ^ 2 * (2 * sinh s * cosh s)
        ≤ r s ^ 2 * ((cosh s ^ 2 + sinh s ^ 2) / 2) :=
          mul_le_mul_of_nonneg_left (two_sinh_mul_cosh_le s hs) (sq_nonneg _)
      _ = 1 - 1 / 2 := by linear_combination (1 / 2 : ℝ) * hr s
  -- the self-attention dynamics with a massless first cluster
  have hsum : ∀ (β : ℝ) (t : ℝ) (i : Idx 2),
      (∑ k : Idx 2, (![0, 1] : Idx 2 → ℝ) k * Real.exp (β * inner (𝕜 := ℝ)
          ((Xt t i : EucSpace d)) ((Xt t k : EucSpace d))))⁻¹ •
        ∑ j : Idx 2, ((![0, 1] : Idx 2 → ℝ) j * Real.exp (β * inner (𝕜 := ℝ)
          ((Xt t i : EucSpace d)) ((Xt t j : EucSpace d)))) • ((Xt t j : EucSpace d)) = e := by
    intro β t i
    simp [Fin.sum_univ_two, Xt, smul_smul, exp_ne_zero]
  have hX : ∀ β : ℝ, 0 < β → Xt 0 = Yt 0 ∧ clusterSA d 2 ![0, 1] β Xt := by
    refine fun β _ => ⟨hXY, fun t i => ?_⟩
    rw [hsum β t i]
    fin_cases i
    · exact hasDerivAt_massless d e f he hef t
    · change HasDerivAt (fun _ => e) (proj d e e) t
      rw [proj, hee, one_smul, sub_self]
      exact hasDerivAt_const t e
  -- the heavy cluster stands still for `X`, and moves for `Y`
  let δ := ‖e - (Yt (1 / 4) 1 : EucSpace d)‖
  have hδ : 0 < δ := by
    refine norm_pos_iff.2 (sub_ne_zero.2 fun hEq => ?_)
    have h1 : inner (𝕜 := ℝ) f ((Yt (1 / 4) 1 : EucSpace d)) = r (1 / 4) * sinh (1 / 4) := by
      change inner (𝕜 := ℝ) f (r (1 / 4) • (sinh (1 / 4) • f + cosh (1 / 4) • e)) = _
      simp only [real_inner_smul_right, inner_add_right, real_inner_self_eq_norm_sq, hf, hfe]
      ring
    rw [← hEq, hfe] at h1
    have : 0 < r (1 / 4) * sinh (1 / 4) :=
      mul_pos (inv_pos.2 (Real.sqrt_pos.2 (cosh_pos _))) (sinh_pos_iff.2 (by norm_num))
    linarith
  have hα : ∀ j : Idx 2, 0 ≤ (![0, 1] : Idx 2 → ℝ) j := fun j => by fin_cases j <;> simp
  have hα1 : ∑ j : Idx 2, (![0, 1] : Idx 2 → ℝ) j = 1 := by simp [Fin.sum_univ_two]
  have hmax : ∀ i j : Idx 2, i ≠ j → (i, j) ≠ (0, 1) → (i, j) ≠ (1, 0) →
      inner (𝕜 := ℝ) ((Yt 0 i : EucSpace d)) ((Yt 0 j : EucSpace d))
        < inner (𝕜 := ℝ) ((Yt 0 0 : EucSpace d)) ((Yt 0 1 : EucSpace d)) := by
    intro i j hij h1 h2
    fin_cases i <;> fin_cases j
    · exact absurd rfl hij
    · exact absurd rfl h1
    · exact absurd rfl h2
    · exact absurd rfl hij
  obtain ⟨B, hB⟩ := h 2 ![0, 1] (Yt 0) hα hα1 0 1 (by decide) hmax
    (1 / 2) (1 / 4) (by norm_num) (by norm_num) Yt rfl hhard hcond (fun _ => Xt) hX δ hδ
  have hlt := hB (max B 0 + 1) (by linarith [le_max_left B 0]) (1 / 4) ⟨by norm_num, le_rfl⟩ 1
  exact lt_irrefl δ hlt

/-- The hypothesis of `not_agazzi_merge` is satisfiable. -/
example : 2 ≤ 2 := le_rfl

end MeanField
end Transformer

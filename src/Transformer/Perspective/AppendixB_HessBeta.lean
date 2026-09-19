/-
# Appendix B — the Hessian of `𝖤_β` along a block rotation, proved

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The `β > 0` companion of `Perspective.hessian_at_critical`.  Rotate the tokens
indexed by `𝒮` by `e^{tB}` with `B` skew and leave the others where they are;
then

  `𝖤_β''(0) = (2β)⁻¹ · 2β Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c}
                 e^{β ⟨x_i, x_j⟩} (β ⟨B x_i, x_j⟩² + ⟨B² x_i, x_j⟩)`.

The bookkeeping is the same as at `β = 0`: a pair inside `𝒮` contributes
nothing because `B` is skew, a pair inside `𝒮^c` contributes nothing because
it does not move, and the two cross blocks contribute the same amount.  What
is new is only the chain rule through `e^{βu}`, which turns `u'' ` into
`β u'' + β² (u')²`.

The scalar is left as `(2β)⁻¹ · 2β` rather than cancelled, so that the
identity holds at every `β` — at `β = 0` both sides are `0`, since Lean reads
`(2·0)⁻¹` as `0`.  `Perspective.dr1_skew_inequality` cancels it where `β ≠ 0`.
-/

import Transformer.Perspective.AppendixB_EBeta
import Transformer.Perspective.DoubleSum

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **The Hessian of `𝖤_β` along a block rotation.**

For a skew-symmetric `B`, a subset `𝒮 ⊂ [n]`, and the perturbation
`x_i(t) = e^{tB} x_i` (`i ∈ 𝒮`), `x_i(t) = x_i` (`i ∉ 𝒮`),

  `𝖤_β''(0) = (2β)⁻¹ · 2β Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c}
                 e^{β ⟨x_i, x_j⟩} (β ⟨B x_i, x_j⟩² + ⟨B² x_i, x_j⟩)`.

Source: arXiv:2312.10794v5, Appendix B, the computation behind `eq: dr1`;
it is `e:helpcl` of Appendix A with `⟨x_i, x_j⟩` replaced by
`e^{β ⟨x_i, x_j⟩}`. -/
theorem secondDeriv_selfEnergy
    (X : SphereTuple d n) (β : ℝ) (B : ParamMatrix d) (𝒮 : Finset (Idx n))
    (hB : IsSkew d B)
    (Y : ℝ → SphereTuple d n) (hY : PerturbationBy d n B 𝒮 X Y) :
    SecondDerivEBetaAt d n β Y
      ((2 * β)⁻¹ * (2 * β * ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
        Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
          * (β * (inner (𝕜 := ℝ) (B ((X i : EucSpace d))) ((X j : EucSpace d))) ^ 2
              + inner (𝕜 := ℝ) (B (B ((X i : EucSpace d)))) ((X j : EucSpace d))))) := by
  classical
  obtain ⟨hY0, hYS, hYc⟩ := hY
  -- The velocity field of the curve: `B` on `𝒮`, and `0` off it.
  set A : Idx n → ParamMatrix d := fun i => if i ∈ 𝒮 then B else 0 with hA
  have hvel : ∀ (i : Idx n) (t : ℝ),
      HasDerivAt (fun s => (Y s i : EucSpace d)) (A i ((Y t i : EucSpace d))) t := by
    intro i t
    by_cases hi : i ∈ 𝒮
    · simpa [hA, hi] using hYS i hi t
    · have hconst : (fun s => (Y s i : EucSpace d)) = fun _ => (X i : EucSpace d) :=
        funext fun s => hYc i hi s
      have hc : HasDerivAt (fun _ : ℝ => (X i : EucSpace d)) 0 t := hasDerivAt_const _ _
      rw [hconst]
      simpa [hA, hi] using hc
  have hvel2 : ∀ (i : Idx n) (t : ℝ),
      HasDerivAt (fun s => A i ((Y s i : EucSpace d)))
        (A i (A i ((Y t i : EucSpace d)))) t := by
    intro i t
    simpa [Function.comp_def] using (A i).hasFDerivAt.comp_hasDerivAt t (hvel i t)
  -- `u_{ij}(t) = ⟨x_i(t), x_j(t)⟩` and its first two derivatives.
  set GG : Idx n → Idx n → ℝ → ℝ := fun i j t =>
    inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (A j ((Y t j : EucSpace d)))
      + inner (𝕜 := ℝ) ((Y t j : EucSpace d)) (A i ((Y t i : EucSpace d))) with hGG
  set HH : Idx n → Idx n → ℝ → ℝ := fun i j t =>
    (inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (A j (A j ((Y t j : EucSpace d))))
        + inner (𝕜 := ℝ) (A i ((Y t i : EucSpace d))) (A j ((Y t j : EucSpace d))))
      + (inner (𝕜 := ℝ) ((Y t j : EucSpace d)) (A i (A i ((Y t i : EucSpace d))))
        + inner (𝕜 := ℝ) (A j ((Y t j : EucSpace d))) (A i ((Y t i : EucSpace d)))) with hHH
  have ha : ∀ (i j : Idx n) (t : ℝ),
      HasDerivAt (fun s => inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s j : EucSpace d)))
        (GG i j t) t := by
    intro i j t
    have h := HasDerivAt.inner ℝ (hvel i t) (hvel j t)
    rw [real_inner_comm ((Y t j : EucSpace d)) (A i ((Y t i : EucSpace d)))] at h
    simpa only [hGG] using h
  have hb : ∀ (i j : Idx n) (t : ℝ), HasDerivAt (fun s => GG i j s) (HH i j t) t := by
    intro i j t
    have h1 := HasDerivAt.inner ℝ (hvel i t) (hvel2 j t)
    have h2 := HasDerivAt.inner ℝ (hvel j t) (hvel2 i t)
    simpa only [hGG, hHH] using h1.fun_add h2
  -- `(e^{β u_{ij}})' = e^{β u_{ij}} · β u'_{ij}`.
  set W : Idx n → Idx n → ℝ → ℝ := fun i j t =>
    Real.exp (β * inner (𝕜 := ℝ) ((Y t i : EucSpace d)) ((Y t j : EucSpace d)))
      * (β * GG i j t) with hW
  have hexp : ∀ (i j : Idx n) (t : ℝ),
      HasDerivAt
        (fun s => Real.exp (β * inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s j : EucSpace d))))
        (W i j t) t := by
    intro i j t
    simpa only [hW] using ((ha i j t).const_mul β).exp
  -- `(e^{β u})'' = e^{β u} (β² (u')² + β u'')`.
  set K : Idx n → Idx n → ℝ := fun i j =>
    (Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
        * (β * GG i j 0)) * (β * GG i j 0)
      + Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
        * (β * HH i j 0) with hK
  have hKd : ∀ i j : Idx n, HasDerivAt (fun s => W i j s) (K i j) 0 := by
    intro i j
    have h := (hexp i j 0).fun_mul ((hb i j 0).const_mul β)
    simp only [hW] at h
    rw [hY0] at h
    simpa only [hW, hK] using h
  refine ⟨fun t => (2 * β)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, W i j t, fun t => ?_, ?_⟩
  · have h := (hasDerivAt_double_sum n
      (fun i j s =>
        Real.exp (β * inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s j : EucSpace d))))
      (fun i j => W i j t) t (fun i j => hexp i j t)).const_mul ((2 * β)⁻¹)
    simpa only [selfEnergy, particleEnergy, ContinuousLinearMap.id_apply] using h
  · have h := (hasDerivAt_double_sum n (fun i j t => W i j t) (fun i j => K i j) 0
      (fun i j => hKd i j)).const_mul ((2 * β)⁻¹)
    refine h.congr_deriv ?_
    congr 1
    -- The four blocks: `𝒮 × 𝒮` and `𝒮^c × 𝒮^c` vanish, the two cross blocks agree.
    have hGGsymm : ∀ i j : Idx n, GG i j 0 = GG j i 0 := by
      intro i j; simp only [hGG]; ring
    have hHHsymm : ∀ i j : Idx n, HH i j 0 = HH j i 0 := by
      intro i j; simp only [hHH]; ring
    have hKsymm : ∀ i j : Idx n, K i j = K j i := by
      intro i j
      simp only [hK, hGGsymm i j, hHHsymm i j,
        real_inner_comm ((X i : EucSpace d)) ((X j : EucSpace d))]
    have hKin : ∀ i ∈ 𝒮, ∀ j ∈ 𝒮, K i j = 0 := by
      intro i hi j hj
      have hAi : A i = B := by simp [hA, hi]
      have hAj : A j = B := by simp [hA, hj]
      have hg : GG i j 0 = 0 := by
        have h1 := hB ((X i : EucSpace d)) ((X j : EucSpace d))
        have h2 := real_inner_comm ((X j : EucSpace d)) (B ((X i : EucSpace d)))
        simp only [hGG, hY0, hAi, hAj]
        linarith
      have hh : HH i j 0 = 0 := by
        have h1 := hB ((X i : EucSpace d)) (B ((X j : EucSpace d)))
        have h2 := hB ((X j : EucSpace d)) (B ((X i : EucSpace d)))
        simp only [hHH, hY0, hAi, hAj]
        linarith
      simp [hK, hg, hh]
    have hKout : ∀ i ∉ 𝒮, ∀ j ∉ 𝒮, K i j = 0 := by
      intro i hi j hj
      have hAi : A i = 0 := by simp [hA, hi]
      have hAj : A j = 0 := by simp [hA, hj]
      have hg : GG i j 0 = 0 := by simp [hGG, hY0, hAi, hAj]
      have hh : HH i j 0 = 0 := by simp [hHH, hY0, hAi, hAj]
      simp [hK, hg, hh]
    have hKcross : ∀ i ∈ 𝒮, ∀ j ∉ 𝒮, K i j
        = β * (Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
            * (β * (inner (𝕜 := ℝ) (B ((X i : EucSpace d))) ((X j : EucSpace d))) ^ 2
                + inner (𝕜 := ℝ) (B (B ((X i : EucSpace d)))) ((X j : EucSpace d)))) := by
      intro i hi j hj
      have hAi : A i = B := by simp [hA, hi]
      have hAj : A j = 0 := by simp [hA, hj]
      have hg : GG i j 0 = inner (𝕜 := ℝ) (B ((X i : EucSpace d))) ((X j : EucSpace d)) := by
        simp only [hGG, hY0, hAi, hAj, zero_apply, inner_zero_right, zero_add]
        exact (real_inner_comm ((X j : EucSpace d)) (B ((X i : EucSpace d)))).symm
      have hh : HH i j 0
          = inner (𝕜 := ℝ) (B (B ((X i : EucSpace d)))) ((X j : EucSpace d)) := by
        simp only [hHH, hY0, hAi, hAj, zero_apply, inner_zero_right, inner_zero_left,
          zero_add, add_zero]
        exact (real_inner_comm ((X j : EucSpace d)) (B (B ((X i : EucSpace d))))).symm
      rw [hK]
      simp only [hg, hh]
      ring
    set S : ℝ := ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
      Real.exp (β * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
        * (β * (inner (𝕜 := ℝ) (B ((X i : EucSpace d))) ((X j : EucSpace d))) ^ 2
            + inner (𝕜 := ℝ) (B (B ((X i : EucSpace d)))) ((X j : EucSpace d))) with hS
    have hin : ∑ i ∈ 𝒮, ∑ j : Idx n, K i j = β * S := by
      rw [hS, Finset.mul_sum]
      refine Finset.sum_congr rfl fun i hi => ?_
      rw [← Finset.sum_add_sum_compl 𝒮 (fun j => K i j),
        Finset.sum_eq_zero (fun j hj => hKin i hi j hj), zero_add, Finset.mul_sum]
      exact Finset.sum_congr rfl fun j hj => hKcross i hi j (Finset.mem_compl.mp hj)
    have hout : ∑ i ∈ 𝒮ᶜ, ∑ j : Idx n, K i j = β * S := by
      have hsplit : ∀ i ∈ (𝒮ᶜ : Finset (Idx n)), ∑ j : Idx n, K i j = ∑ j ∈ 𝒮, K i j := by
        intro i hi
        rw [← Finset.sum_add_sum_compl 𝒮 (fun j => K i j),
          Finset.sum_eq_zero
            (fun j hj => hKout i (Finset.mem_compl.mp hi) j (Finset.mem_compl.mp hj)),
          add_zero]
      rw [Finset.sum_congr rfl hsplit, Finset.sum_comm]
      rw [← hin]
      exact Finset.sum_congr rfl fun i hi => by
        rw [← Finset.sum_add_sum_compl 𝒮 (fun j => K i j),
          Finset.sum_eq_zero (fun j hj => hKin i hi j hj), zero_add]
        exact Finset.sum_congr rfl fun j _ => hKsymm j i
    calc ∑ i : Idx n, ∑ j : Idx n, K i j
        = ∑ i ∈ 𝒮, ∑ j : Idx n, K i j + ∑ i ∈ 𝒮ᶜ, ∑ j : Idx n, K i j :=
          (Finset.sum_add_sum_compl 𝒮 (fun i => ∑ j : Idx n, K i j)).symm
      _ = 2 * β * S := by rw [hin, hout]; ring

/-- The hypotheses of `secondDeriv_selfEnergy` are satisfiable: the zero
matrix is skew, and rotating none of the particles — `𝒮 = ∅` — leaves the
constant curve as the perturbation. -/
example :
    IsSkew 1 0 ∧
      PerturbationBy 1 2 0 ∅ (antipodalPair 1 northPole)
        (fun _ => antipodalPair 1 northPole) :=
  ⟨fun x y => by simp,
    rfl, fun i hi => absurd hi (Finset.notMem_empty i), fun _ _ _ => rfl⟩

end Perspective
end Transformer

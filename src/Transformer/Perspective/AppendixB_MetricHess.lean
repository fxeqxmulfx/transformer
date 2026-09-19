/-
# Appendix B — the Hessians of `𝖤_β` and `𝖤_0` differ by `O(β)`, proved

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, Appendix B, `eq: metric.hess`.

The second-order companion of `Perspective.metric_grad_comparison`: along the
block rotation `x_i(t) = e^{tB} x_i` (`i ∈ 𝒮`), `x_i(t) = x_i` (`i ∉ 𝒮`), the
second derivatives of `𝖤_β` and of `(n/2) 𝖤_0` agree up to `O(β)`.

The proof differentiates both energies twice along the curve.  Writing
`a_{ij}(t) = ⟨x_i(t), x_j(t)⟩` and `w_{ij} = e^{β a_{ij}(0)}`,

  `𝖤_β''(0) - (n/2) 𝖤_0''(0) = Σ_i Σ_j ((β/2) w_{ij} a_{ij}'(0)²
                                        + (1/2) (w_{ij} - 1) a_{ij}''(0))`,

and on the sphere `|a_{ij}'| ≤ 2‖B‖`, `|a_{ij}''| ≤ 4‖B‖²`, `w_{ij} ≤ e` and
`|w_{ij} - 1| ≤ 2β` for `β ≤ 1`.  The constant is `(2e + 4) n² ‖B‖² + 1`.

Deviation from the source: skew-symmetry of `B` is *not* assumed.  The paper
states `eq: metric.hess` for the block rotations of `e:helpcl`, whose `B` is
skew; the bound above holds for every `B`, and `IsSkew` is dropped rather than
carried as an unused hypothesis.  Skewness is what makes `PerturbationBy`
satisfiable — it is why the rotated tokens stay on the sphere — but it enters
nowhere in the estimate.
-/

import Transformer.Perspective.AppendixB_MetricGrad
import Transformer.Perspective.DoubleSum

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (eq: metric.hess).** *Comparison of Hessians.*

  `Hess_{g_β} 𝖤_β(x)[v] = Hess_g 𝖤_0(x)[v] + O(β)`,

along the block rotation `PerturbationBy d n B 𝒮 X`, and with the same
normalisation factor `n/2` as in `Perspective.metric_grad_comparison`.
Together with `hessian_at_critical_intrinsic` this is what transports the
`β = 0` saddle analysis of Appendix A to small `β > 0`.

Source: arXiv:2312.10794v5, Appendix B, `eq: metric.hess`.  Skew-symmetry of
`B` is not needed for the estimate and is not assumed; see the module
docstring. -/
theorem metric_hess_comparison
    (X : SphereTuple d n) (B : ParamMatrix d) (𝒮 : Finset (Idx n)) :
    ∃ C : ℝ, 0 < C ∧
      ∀ Y : ℝ → SphereTuple d n, PerturbationBy d n B 𝒮 X Y →
        ∀ c₀ : ℝ, SecondDerivE0At d n Y c₀ →
          ∀ β : ℝ, 0 < β → β ≤ 1 →
            ∀ cβ : ℝ, SecondDerivEBetaAt d n β Y cβ →
              |cβ - ((n : ℝ) / 2) * c₀| ≤ C * β := by
  refine ⟨(2 * Real.exp 1 + 4) * (n : ℝ) ^ 2 * ‖B‖ ^ 2 + 1, by positivity, ?_⟩
  intro Y hY c₀ hc₀ β hβ hβ1 cβ hcβ
  have hβ0 : β ≠ 0 := ne_of_gt hβ
  obtain ⟨-, hYS, hYc⟩ := hY
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
      HasDerivAt (fun s => A i ((Y s i : EucSpace d))) (A i (A i ((Y t i : EucSpace d)))) t := by
    intro i t
    simpa [Function.comp_def] using (A i).hasFDerivAt.comp_hasDerivAt t (hvel i t)
  -- `a_{ij}`, its first and second derivative, and the weight `w_{ij}`.
  set GG : Idx n → Idx n → ℝ → ℝ := fun i j t =>
    inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (A j ((Y t j : EucSpace d)))
      + inner (𝕜 := ℝ) ((Y t j : EucSpace d)) (A i ((Y t i : EucSpace d))) with hGG
  set HH : Idx n → Idx n → ℝ → ℝ := fun i j t =>
    (inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (A j (A j ((Y t j : EucSpace d))))
        + inner (𝕜 := ℝ) (A i ((Y t i : EucSpace d))) (A j ((Y t j : EucSpace d))))
      + (inner (𝕜 := ℝ) ((Y t j : EucSpace d)) (A i (A i ((Y t i : EucSpace d))))
        + inner (𝕜 := ℝ) (A j ((Y t j : EucSpace d))) (A i ((Y t i : EucSpace d)))) with hHH
  set W : Idx n → Idx n → ℝ → ℝ := fun i j t =>
    Real.exp (β * inner (𝕜 := ℝ) ((Y t i : EucSpace d)) ((Y t j : EucSpace d))) with hW
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
  -- The two derivatives of `𝖤_0`.
  have hE0 : ∀ t : ℝ, HasDerivAt (fun s => E0 d n (Y s))
      ((n : ℝ)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, GG i j t) t := by
    intro t
    have h := (hasDerivAt_double_sum n
      (fun i j s => inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s j : EucSpace d)))
      (fun i j => GG i j t) t (fun i j => ha i j t)).const_mul ((n : ℝ)⁻¹)
    simpa only [E0] using h
  have hE0' : HasDerivAt (fun t => (n : ℝ)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, GG i j t)
      ((n : ℝ)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, HH i j 0) 0 :=
    (hasDerivAt_double_sum n (fun i j t => GG i j t) (fun i j => HH i j 0) 0
      (fun i j => hb i j 0)).const_mul ((n : ℝ)⁻¹)
  -- The two derivatives of `𝖤_β`.
  have hEβ : ∀ t : ℝ, HasDerivAt (fun s => selfEnergy d n β (Y s))
      ((2 * β)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, W i j t * (β * GG i j t)) t := by
    intro t
    simpa only [hW, hGG] using
      hasDerivAt_selfEnergy d n β Y (fun i => A i ((Y t i : EucSpace d))) t (fun i => hvel i t)
  have hprod : ∀ i j : Idx n, HasDerivAt (fun t => W i j t * (β * GG i j t))
      (W i j 0 * (β * GG i j 0) * (β * GG i j 0) + W i j 0 * (β * HH i j 0)) 0 := by
    intro i j
    have hwd : HasDerivAt (fun t => W i j t) (W i j 0 * (β * GG i j 0)) 0 := by
      simpa only [hW] using ((ha i j 0).const_mul β).exp
    exact hwd.fun_mul ((hb i j 0).const_mul β)
  have hEβ' : HasDerivAt
      (fun t => (2 * β)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, W i j t * (β * GG i j t))
      ((2 * β)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
        (W i j 0 * (β * GG i j 0) * (β * GG i j 0) + W i j 0 * (β * HH i j 0))) 0 :=
    (hasDerivAt_double_sum n (fun i j t => W i j t * (β * GG i j t)) _ 0 hprod).const_mul _
  -- The two second derivatives, identified with the hypotheses.
  obtain ⟨f₀, hf₀, hf₀'⟩ := hc₀
  rw [funext fun t => (hf₀ t).unique (hE0 t)] at hf₀'
  have hc₀val : c₀ = (n : ℝ)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, HH i j 0 := hf₀'.unique hE0'
  obtain ⟨fβ, hfβ, hfβ'⟩ := hcβ
  rw [funext fun t => (hfβ t).unique (hEβ t)] at hfβ'
  have hcβval : cβ = (2 * β)⁻¹ * ∑ i : Idx n, ∑ j : Idx n,
      (W i j 0 * (β * GG i j 0) * (β * GG i j 0) + W i j 0 * (β * HH i j 0)) := hfβ'.unique hEβ'
  have hc₀half : ((n : ℝ) / 2) * c₀ = (2 : ℝ)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, HH i j 0 := by
    rcases Finset.eq_empty_or_nonempty (Finset.univ : Finset (Idx n)) with he | hne
    · rw [hc₀val, he]; simp
    · obtain ⟨i₀, -⟩ := hne
      have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Fin.pos_iff_nonempty.mpr ⟨i₀⟩).ne'
      rw [hc₀val, ← mul_assoc]
      congr 1
      field_simp
  have hdiff : cβ - ((n : ℝ) / 2) * c₀
      = ∑ i : Idx n, ∑ j : Idx n,
          ((β / 2) * (W i j 0 * (GG i j 0) ^ 2) + 2⁻¹ * ((W i j 0 - 1) * HH i j 0)) := by
    rw [hcβval, hc₀half, const_mul_double_sum, const_mul_double_sum, double_sum_sub]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    field_simp
    ring
  -- The bounds on the sphere.
  have hnormY : ∀ (i : Idx n) (t : ℝ), ‖(Y t i : EucSpace d)‖ = 1 :=
    fun i t => mem_sphere_zero_iff_norm.mp (Y t i).2
  have hAnorm : ∀ (i : Idx n) (x : EucSpace d), ‖A i x‖ ≤ ‖B‖ * ‖x‖ := by
    intro i x
    by_cases hi : i ∈ 𝒮
    · simpa [hA, hi] using B.le_opNorm x
    · have h0 : ‖A i x‖ = 0 := by simp [hA, hi]
      rw [h0]
      positivity
  have hAY : ∀ (i : Idx n) (t : ℝ), ‖A i ((Y t i : EucSpace d))‖ ≤ ‖B‖ := by
    intro i t
    simpa [hnormY i t] using hAnorm i ((Y t i : EucSpace d))
  have hAAY : ∀ (i : Idx n) (t : ℝ), ‖A i (A i ((Y t i : EucSpace d)))‖ ≤ ‖B‖ ^ 2 := by
    intro i t
    have h1 := hAnorm i (A i ((Y t i : EucSpace d)))
    have h2 := hAY i t
    nlinarith [norm_nonneg B]
  have hGGbound : ∀ i j : Idx n, |GG i j 0| ≤ 2 * ‖B‖ := by
    intro i j
    have h1 : |inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) (A j ((Y 0 j : EucSpace d)))| ≤ ‖B‖ := by
      refine (abs_real_inner_le_norm _ _).trans ?_
      rw [hnormY i 0, one_mul]
      exact hAY j 0
    have h2 : |inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (A i ((Y 0 i : EucSpace d)))| ≤ ‖B‖ := by
      refine (abs_real_inner_le_norm _ _).trans ?_
      rw [hnormY j 0, one_mul]
      exact hAY i 0
    simp only [hGG]
    refine (abs_add_le _ _).trans ?_
    linarith
  have hHHbound : ∀ i j : Idx n, |HH i j 0| ≤ 4 * ‖B‖ ^ 2 := by
    intro i j
    have e1 : |inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) (A j (A j ((Y 0 j : EucSpace d))))|
        ≤ ‖B‖ ^ 2 := by
      refine (abs_real_inner_le_norm _ _).trans ?_
      rw [hnormY i 0, one_mul]
      exact hAAY j 0
    have e3 : |inner (𝕜 := ℝ) ((Y 0 j : EucSpace d)) (A i (A i ((Y 0 i : EucSpace d))))|
        ≤ ‖B‖ ^ 2 := by
      refine (abs_real_inner_le_norm _ _).trans ?_
      rw [hnormY j 0, one_mul]
      exact hAAY i 0
    have e2 : |inner (𝕜 := ℝ) (A i ((Y 0 i : EucSpace d))) (A j ((Y 0 j : EucSpace d)))|
        ≤ ‖B‖ ^ 2 := by
      refine (abs_real_inner_le_norm _ _).trans ?_
      nlinarith [hAY i 0, hAY j 0, norm_nonneg (A i ((Y 0 i : EucSpace d))),
        norm_nonneg (A j ((Y 0 j : EucSpace d))), norm_nonneg B]
    have e4 : |inner (𝕜 := ℝ) (A j ((Y 0 j : EucSpace d))) (A i ((Y 0 i : EucSpace d)))|
        ≤ ‖B‖ ^ 2 := by
      refine (abs_real_inner_le_norm _ _).trans ?_
      nlinarith [hAY i 0, hAY j 0, norm_nonneg (A i ((Y 0 i : EucSpace d))),
        norm_nonneg (A j ((Y 0 j : EucSpace d))), norm_nonneg B]
    simp only [hHH]
    refine (abs_add_le _ _).trans ((add_le_add (abs_add_le _ _) (abs_add_le _ _)).trans ?_)
    linarith
  have hinner1 : ∀ i j : Idx n,
      |β * inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))| ≤ 1 := by
    intro i j
    have h1 : |inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))| ≤ 1 := by
      have h := abs_real_inner_le_norm ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))
      rwa [hnormY i 0, hnormY j 0, one_mul] at h
    rw [abs_mul, abs_of_pos hβ]
    nlinarith [abs_nonneg (inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d)))]
  have hWpos : ∀ i j : Idx n, 0 < W i j 0 := by
    intro i j
    simp only [hW]
    exact Real.exp_pos _
  have hWle : ∀ i j : Idx n, W i j 0 ≤ Real.exp 1 := by
    intro i j
    simp only [hW]
    refine Real.exp_le_exp.mpr ?_
    have h := abs_le.mp (hinner1 i j)
    linarith [h.2]
  have hWsub : ∀ i j : Idx n, |W i j 0 - 1| ≤ 2 * β := by
    intro i j
    simp only [hW]
    refine (Real.abs_exp_sub_one_le (hinner1 i j)).trans ?_
    rw [abs_mul, abs_of_pos hβ]
    have h1 : |inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))| ≤ 1 := by
      have h := abs_real_inner_le_norm ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d))
      rwa [hnormY i 0, hnormY j 0, one_mul] at h
    nlinarith [abs_nonneg (inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) ((Y 0 j : EucSpace d)))]
  -- The term-by-term estimate, and the sum.
  have hterm : ∀ i j : Idx n,
      |(β / 2) * (W i j 0 * (GG i j 0) ^ 2) + 2⁻¹ * ((W i j 0 - 1) * HH i j 0)|
        ≤ β * ((2 * Real.exp 1 + 4) * ‖B‖ ^ 2) := by
    intro i j
    have hg2 : (GG i j 0) ^ 2 ≤ 4 * ‖B‖ ^ 2 := by
      have h := hGGbound i j
      nlinarith [abs_nonneg (GG i j 0), sq_abs (GG i j 0), norm_nonneg B]
    have h1 : |(β / 2) * (W i j 0 * (GG i j 0) ^ 2)| ≤ β * (2 * Real.exp 1 * ‖B‖ ^ 2) := by
      have hnn : 0 ≤ (β / 2) * (W i j 0 * (GG i j 0) ^ 2) :=
        mul_nonneg (by linarith) (mul_nonneg (hWpos i j).le (sq_nonneg _))
      rw [abs_of_nonneg hnn]
      have hWG : W i j 0 * (GG i j 0) ^ 2 ≤ Real.exp 1 * (4 * ‖B‖ ^ 2) :=
        mul_le_mul (hWle i j) hg2 (sq_nonneg _) (Real.exp_pos 1).le
      nlinarith [mul_le_mul_of_nonneg_left hWG (by linarith : (0 : ℝ) ≤ β / 2)]
    have h2 : |2⁻¹ * ((W i j 0 - 1) * HH i j 0)| ≤ β * (4 * ‖B‖ ^ 2) := by
      rw [abs_mul, abs_mul]
      have h2inv : |(2 : ℝ)⁻¹| = 2⁻¹ := by norm_num
      rw [h2inv]
      linarith [mul_le_mul (hWsub i j) (hHHbound i j) (abs_nonneg (HH i j 0))
        (by linarith : (0 : ℝ) ≤ 2 * β)]
    refine (abs_add_le _ _).trans ?_
    nlinarith [h1, h2]
  rw [hdiff]
  refine (abs_double_sum_le n _ _ hterm).trans ?_
  nlinarith [hβ.le, sq_nonneg ((n : ℝ))]

/-- The hypotheses inside the statement are satisfiable: the constant curve at
`X` is the perturbation by `B = 0` along `𝒮 = ∅`, and both second derivatives
along it vanish. -/
example (X : SphereTuple d n) :
    PerturbationBy d n 0 ∅ X (fun _ => X) ∧ SecondDerivE0At d n (fun _ => X) 0 ∧
      SecondDerivEBetaAt d n 1 (fun _ => X) 0 :=
  ⟨⟨rfl, fun i hi => absurd hi (Finset.notMem_empty i), fun _ _ _ => rfl⟩,
    ⟨fun _ => 0, fun _ => hasDerivAt_const _ _, hasDerivAt_const _ _⟩,
    ⟨fun _ => 0, fun _ => hasDerivAt_const _ _, hasDerivAt_const _ _⟩⟩

end Perspective
end Transformer

/-
# Appendix A — the Hessian of `𝖤_0` along a block rotation, proved

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`e:helpcl`: rotate the tokens indexed by `𝒮` by `e^{tB}` with `B` skew and
leave the others where they are; then

  `𝖤_0''(0) = (2/n) Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} ⟨B² x_i, x_j⟩`.

The whole content is that the pairs inside `𝒮` and the pairs inside `𝒮^c`
contribute nothing.  The second kind is constant in `t`; the first kind is
constant because `B` is skew — `⟨B x_i, B x_j⟩ = -⟨x_i, B² x_j⟩` cancels the
term it sits next to — which is the only place skewness is used, and the
reason `e^{tB}` is a rotation in the first place.  What is left is the cross
terms, counted twice by the symmetry of the double sum.

The perturbation is described by the differential equation `ẋ_i = B x_i` its
rotated tokens solve, not by the matrix exponential; on the sphere the two are
the same and the `HasDerivAt` API is the one the rest of the development
speaks.
-/

import Transformer.Perspective.AppendixA_Beta0
import Transformer.Perspective.RussianTrick
import Transformer.Perspective.DoubleSum

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The perturbation of `X` used in `e:helpcl`:

  `x_i(t) = e^{tB} x_i` for `i ∈ 𝒮`,   `x_i(t) = x_i` otherwise.

The rotated particles are described by the differential equation
`ẋ_i(t) = B x_i(t)` they solve rather than by the matrix exponential, which is
the same thing for the initial condition `Y 0 = X` and keeps the statement
inside the `HasDerivAt` API used everywhere else here. -/
def PerturbationBy
    (B : ParamMatrix d) (𝒮 : Finset (Idx n))
    (X : SphereTuple d n) (Y : ℝ → SphereTuple d n) : Prop :=
  Y 0 = X ∧
  (∀ i ∈ 𝒮, ∀ t : ℝ,
    HasDerivAt (fun s => (Y s i : EucSpace d)) (B ((Y t i : EucSpace d))) t) ∧
  (∀ i ∉ 𝒮, ∀ t : ℝ, (Y t i : EucSpace d) = (X i : EucSpace d))

/-- `c` is the second derivative of `t ↦ 𝖤_0(Y(t))` at `t = 0`: the energy is
differentiable along the whole curve, and its derivative is again
differentiable at `0`, with derivative `c`. -/
def SecondDerivE0At (Y : ℝ → SphereTuple d n) (c : ℝ) : Prop :=
  ∃ f' : ℝ → ℝ,
    (∀ t : ℝ, HasDerivAt (fun s => E0 d n (Y s)) (f' t) t) ∧ HasDerivAt f' c 0

/-- **Equation (e:helpcl).** *Hessian of `𝖤_0` at a critical point.*

For a skew-symmetric `B`, a subset `𝒮 ⊂ [n]`, and the perturbation
`x_i(t) = e^{tB} x_i` (`i ∈ 𝒮`), `x_i(t) = x_i` (`i ∉ 𝒮`),

  `𝖤_0''(0) = (2/n) Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} ⟨B² x_i, x_j⟩`.

**What the source says and what is changed here.**  The survey states
`e:helpcl` at a critical point of `𝖤_0`, because that is where it uses it:
the first-order term of the expansion vanishes there, and the second-order
term is the whole of it.  The identity above is about the second derivative
alone, and it holds at every `X`: criticality enters the proof nowhere, so it
is not assumed.  `Perspective.yury_lemma`, which is where the paper's use of
`e:helpcl` lives, keeps the hypothesis.

Source: arXiv:2312.10794v5, Appendix A, `e:helpcl`. -/
theorem hessian_at_critical
    (X : SphereTuple d n) (B : ParamMatrix d) (𝒮 : Finset (Idx n))
    (hB : IsSkew d B)
    (Y : ℝ → SphereTuple d n) (hY : PerturbationBy d n B 𝒮 X Y) :
    SecondDerivE0At d n Y
      ((2 * (n : ℝ)⁻¹) * ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
        inner (𝕜 := ℝ) (B (B ((X i : EucSpace d)))) ((X j : EucSpace d))) := by
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
  -- `a_{ij}(t) = ⟨x_i(t), x_j(t)⟩` and its first two derivatives.
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
  refine ⟨fun t => (n : ℝ)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, GG i j t, fun t => ?_, ?_⟩
  · have h := (hasDerivAt_double_sum n
      (fun i j s => inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s j : EucSpace d)))
      (fun i j => GG i j t) t (fun i j => ha i j t)).const_mul ((n : ℝ)⁻¹)
    simpa only [E0] using h
  · have h := (hasDerivAt_double_sum n (fun i j t => GG i j t) (fun i j => HH i j 0) 0
      (fun i j => hb i j 0)).const_mul ((n : ℝ)⁻¹)
    refine h.congr_deriv ?_
    -- One half of `HH`, the other being the same with `i` and `j` exchanged.
    set P : Idx n → Idx n → ℝ := fun i j =>
      inner (𝕜 := ℝ) ((X j : EucSpace d)) (A i (A i ((X i : EucSpace d))))
        + inner (𝕜 := ℝ) (A j ((X j : EucSpace d))) (A i ((X i : EucSpace d))) with hP
    have hHH0 : ∀ i j : Idx n, HH i j 0 = P j i + P i j := by
      intro i j
      simp only [hHH, hP, hY0]
    -- Only the pairs `i ∈ 𝒮`, `j ∈ 𝒮ᶜ` survive.
    have hPout : ∀ i ∈ (𝒮ᶜ : Finset (Idx n)), ∑ j : Idx n, P i j = 0 := by
      intro i hi
      have hAi : A i = 0 := by simp [hA, Finset.mem_compl.mp hi]
      simp [hP, hAi]
    have hPin : ∀ i ∈ 𝒮, ∑ j : Idx n, P i j
        = ∑ j ∈ 𝒮ᶜ,
            inner (𝕜 := ℝ) (B (B ((X i : EucSpace d)))) ((X j : EucSpace d)) := by
      intro i hi
      have hAi : A i = B := by simp [hA, hi]
      rw [← Finset.sum_add_sum_compl 𝒮 (fun j => P i j)]
      have hin : ∑ j ∈ 𝒮, P i j = 0 := by
        refine Finset.sum_eq_zero fun j hj => ?_
        have hAj : A j = B := by simp [hA, hj]
        have hskew := hB ((X j : EucSpace d)) (B ((X i : EucSpace d)))
        simp only [hP, hAi, hAj]
        linarith [hskew]
      rw [hin, zero_add]
      refine Finset.sum_congr rfl fun j hj => ?_
      have hAj : A j = 0 := by simp [hA, Finset.mem_compl.mp hj]
      simp only [hP, hAi, hAj]
      rw [zero_apply, inner_zero_left, add_zero,
        real_inner_comm ((X j : EucSpace d)) (B (B ((X i : EucSpace d))))]
    have hsumP : ∑ i : Idx n, ∑ j : Idx n, P i j
        = ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
            inner (𝕜 := ℝ) (B (B ((X i : EucSpace d)))) ((X j : EucSpace d)) := by
      rw [← Finset.sum_add_sum_compl 𝒮 (fun i => ∑ j : Idx n, P i j),
        Finset.sum_eq_zero hPout, add_zero]
      exact Finset.sum_congr rfl hPin
    have hswap : ∑ i : Idx n, ∑ j : Idx n, P j i = ∑ i : Idx n, ∑ j : Idx n, P i j :=
      Finset.sum_comm
    have hkey : ∑ i : Idx n, ∑ j : Idx n, HH i j 0
        = 2 * ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
            inner (𝕜 := ℝ) (B (B ((X i : EucSpace d)))) ((X j : EucSpace d)) := by
      calc ∑ i : Idx n, ∑ j : Idx n, HH i j 0
          = ∑ i : Idx n, ∑ j : Idx n, (P j i + P i j) :=
            Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => hHH0 i j
        _ = (∑ i : Idx n, ∑ j : Idx n, P j i) + ∑ i : Idx n, ∑ j : Idx n, P i j := by
            rw [← Finset.sum_add_distrib]
            exact Finset.sum_congr rfl fun i _ => Finset.sum_add_distrib
        _ = 2 * ∑ i : Idx n, ∑ j : Idx n, P i j := by rw [hswap]; ring
        _ = _ := by rw [hsumP]
    rw [hkey]; ring

/-- The hypotheses of `hessian_at_critical` are satisfiable: the zero matrix is
skew, and rotating none of the particles of the antipodal pair — `𝒮 = ∅` —
leaves the constant curve as the perturbation. -/
example :
    IsSkew 1 0 ∧
      PerturbationBy 1 2 0 ∅ (antipodalPair 1 northPole)
        (fun _ => antipodalPair 1 northPole) :=
  ⟨fun x y => by simp,
    rfl, fun i hi => absurd hi (Finset.notMem_empty i), fun _ _ _ => rfl⟩

end Perspective
end Transformer

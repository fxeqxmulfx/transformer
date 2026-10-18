/-
# Phase model over a finite abelian group — the landscape below `|G|²/8`

As for `ℤ/n` (`PhaseModel.Landscape`), with `n = |G|`: some character `χ`
has `n Σ_x (1 - cos (ψ_x - arg χ(x))) ≤ V_full(ψ)`, so below `n²/8` the
phases lie within `n/8` of a character (`exists_sum_one_sub_cos_lt`), where
for `n ≥ 4` every critical point is the character up to periods or a strict
saddle (`Abelian.Characters`).  Hence, for `n ≥ 4`, every local minimizer of
`V_full` below `n²/8` is the phases of a character up to periods
(`exists_sub_characterPhase_mem_periods_of_isLocalMin`), a global minimizer
(`divisionEnergy_eq_zero_of_isLocalMin`); every critical point with
`0 < V_full < n²/8` is a strict saddle (`isStrictSaddle_of_divisionEnergy_lt`);
and gradient descent with `0 < η`, `9 n² η < 1`, started below `n²/8`,
almost surely converges to the phases of a character up to periods whenever
it converges (`ae_tendsto_characterPhase`).  Whether the orbits converge,
and the critical points above `n²/8`, are not settled here.
-/

import Transformer.PhaseModel.Abelian.Characters
import Transformer.PhaseModel.Abelian.Fourier
import Transformer.SaddleAvoidance.Minimizers

open Filter MeasureTheory Real
open scoped Topology

namespace Transformer.PhaseModel.Abelian

open SaddleAvoidance (gradientStep)

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {G : Type*} [AddCommGroup G] [Fintype G] [DecidableEq G]

/-- **Below `|G|²/8`, the phases lie within `|G|/8` of a character**: if
`V_full(ψ) < |G|²/8`, some character `χ` has
`Σ_x (1 - cos (ψ_x - arg χ(x))) < |G|/8`. -/
theorem exists_sum_one_sub_cos_lt {ψ : EuclideanSpace ℝ G}
    (hV : divisionEnergy (Finset.univ : Finset (G × G)) ψ < Fintype.card G ^ 2 / 8) :
    ∃ χ : AddChar G ℂ,
      ∑ x, (1 - cos (ψ.ofLp x - (characterPhase χ).ofLp x)) < Fintype.card G / 8 := by
  obtain ⟨χ, hχ⟩ := exists_mul_sum_one_sub_cos_le ψ
  have hn : (0 : ℝ) < Fintype.card G := by exact_mod_cast Fintype.card_pos
  refine ⟨χ, ?_⟩
  by_contra h
  push Not at h
  nlinarith

/-- **Below `|G|²/8`, every local minimizer of `V_full` is a character**: for
`|G| ≥ 4`, a local minimizer `ψ` of `V_full` with `V_full(ψ) < |G|²/8` has
`ψ - (arg χ(x))_x ∈ 2πℤ^G` for some character `χ`. -/
theorem exists_sub_characterPhase_mem_periods_of_isLocalMin (hn : 4 ≤ Fintype.card G)
    {ψ : EuclideanSpace ℝ G}
    (hmin : IsLocalMin (divisionEnergy (Finset.univ : Finset (G × G))) ψ)
    (hV : divisionEnergy (Finset.univ : Finset (G × G)) ψ < Fintype.card G ^ 2 / 8) :
    ∃ χ : AddChar G ℂ, ψ - characterPhase χ ∈ periods G := by
  obtain ⟨χ, hχ⟩ := exists_sum_one_sub_cos_lt hV
  exact ⟨χ, sub_characterPhase_mem_periods_of_isLocalMin hn hmin hχ⟩

/-- **Below `|G|²/8`, `V_full` has no spurious local minimizer**: for
`|G| ≥ 4`, a local minimizer `ψ` of `V_full` with `V_full(ψ) < |G|²/8` is a
zero, a global minimizer. -/
theorem divisionEnergy_eq_zero_of_isLocalMin (hn : 4 ≤ Fintype.card G) {ψ : EuclideanSpace ℝ G}
    (hmin : IsLocalMin (divisionEnergy (Finset.univ : Finset (G × G))) ψ)
    (hV : divisionEnergy (Finset.univ : Finset (G × G)) ψ < Fintype.card G ^ 2 / 8) :
    divisionEnergy (Finset.univ : Finset (G × G)) ψ = 0 :=
  (divisionEnergy_univ_eq_zero_iff ψ).2
    (exists_sub_characterPhase_mem_periods_of_isLocalMin hn hmin hV)

/-- **Below `|G|²/8`, every critical point of `V_full` that is not a zero is
a strict saddle**: for `|G| ≥ 4`, at a critical point `ψ` with
`0 < V_full(ψ) < |G|²/8` the Hessian has a negative eigenvalue. -/
theorem isStrictSaddle_of_divisionEnergy_lt (hn : 4 ≤ Fintype.card G) {ψ : EuclideanSpace ℝ G}
    (hg : gradient (divisionEnergy (Finset.univ : Finset (G × G))) ψ = 0)
    (hV0 : divisionEnergy (Finset.univ : Finset (G × G)) ψ ≠ 0)
    (hV : divisionEnergy (Finset.univ : Finset (G × G)) ψ < Fintype.card G ^ 2 / 8) :
    SaddleAvoidance.IsStrictSaddle (divisionEnergy (Finset.univ : Finset (G × G))) ψ := by
  obtain ⟨χ, hχ⟩ := exists_sum_one_sub_cos_lt hV
  exact isStrictSaddle_of_sum_one_sub_cos_lt hn hg hχ fun h =>
    hV0 ((divisionEnergy_univ_eq_zero_iff ψ).2 ⟨χ, h⟩)

/-- **Gradient descent on `V_full` started below `|G|²/8` converges, if at
all, to the phases of a character**, for almost every start.  Let
`|G| ≥ 4` and `0 < η` with `9 |G|² η < 1`.  For a start `x` drawn from a law
`ν` absolutely continuous with respect to a Haar measure, almost surely: if
`V_full(x) < |G|²/8` and the orbit converges to `w`, then `w` is the phases
of a character up to periods.  The orbit decreases `V_full`, so
`V_full(w) < |G|²/8`; `w` is critical, and almost surely not a strict saddle
(`SaddleAvoidance.ae_not_tendsto_of_isStrictSaddle`). -/
theorem ae_tendsto_characterPhase (hn : 4 ≤ Fintype.card G) (μ : Measure (EuclideanSpace ℝ G))
    [μ.IsAddHaarMeasure] {ν : Measure (EuclideanSpace ℝ G)} (hν : ν ≪ μ) {η : ℝ} (hη : 0 < η)
    (hηn : 9 * Fintype.card G ^ 2 * η < 1) :
    ∀ᵐ x ∂ν, divisionEnergy (Finset.univ : Finset (G × G)) x < Fintype.card G ^ 2 / 8 →
      ∀ w, Tendsto (fun k => (gradientStep (divisionEnergy (Finset.univ :
        Finset (G × G))) η)^[k] x) atTop (𝓝 w) →
        ∃ χ : AddChar G ℂ, w - characterPhase χ ∈ periods G := by
  set V := divisionEnergy (Finset.univ : Finset (G × G)) with hVdef
  have hL := lipschitzWith_gradient_phaseEnergy
    (fun p : (Finset.univ : Finset (G × G)) => intVec (divisionRow p.1))
  have hηL : η * ((∑ p : (Finset.univ : Finset (G × G)),
      ‖intVec (divisionRow p.1)‖₊ ^ 2 : NNReal) : ℝ) < 1 := by
    push_cast
    have h := sum_norm_intVec_divisionRow_sq_le (Finset.univ : Finset (G × G))
    have hcard : ((Finset.univ : Finset (G × G)).card : ℝ) = Fintype.card G ^ 2 := by
      simp [sq]
    rw [hcard] at h
    nlinarith
  have hfd : Differentiable ℝ V := (contDiff_phaseEnergy _ (k := 1)).differentiable one_ne_zero
  filter_upwards [SaddleAvoidance.ae_not_tendsto_of_isStrictSaddle μ hν (contDiff_phaseEnergy _)
    hL hη hηL] with x hx hx0 w hw
  have hg : gradient V w = 0 := SaddleAvoidance.gradient_eq_zero_of_tendsto hL.continuous hη.ne' hw
  have hanti := SaddleAvoidance.antitone_iterate hfd hL hη.le hηL.le x
  have hle : V w ≤ V x :=
    le_of_tendsto' ((hfd.continuous.tendsto w).comp hw) fun k => by
      simpa using hanti (Nat.zero_le k)
  by_contra hne
  exact hx w (isStrictSaddle_of_divisionEnergy_lt hn hg
    (fun h => hne ((divisionEnergy_univ_eq_zero_iff w).1 h)) (hle.trans_lt hx0)) hw

/-- The hypotheses of the first four theorems hold in the Klein four-group
`ℤ/2 × ℤ/2` at the phases of every character `χ`, a global minimizer with
`V_full = 0 < 4²/8`. -/
example (χ : AddChar (ZMod 2 × ZMod 2) ℂ) :
    let V := divisionEnergy (Finset.univ : Finset ((ZMod 2 × ZMod 2) × (ZMod 2 × ZMod 2)))
    4 ≤ Fintype.card (ZMod 2 × ZMod 2) ∧ IsLocalMin V (characterPhase χ) ∧
      V (characterPhase χ) < (Fintype.card (ZMod 2 × ZMod 2) : ℝ) ^ 2 / 8 := by
  intro V
  have h0 : V (characterPhase χ) = 0 := divisionEnergy_characterPhase _ χ
  refine ⟨by simp, Filter.Eventually.of_forall fun y => ?_, by rw [h0]; positivity⟩
  rw [h0]
  exact phaseEnergy_nonneg _ y

/-- The hypotheses of `isStrictSaddle_of_divisionEnergy_lt` hold in every
group `G` of order at least `49`, such as `ℤ/7 × ℤ/7`, at `ψ = π e_0`: every
relation of `ψ` is a multiple of `π`, so `ψ` is critical; the relation of
the pair `(0, 0)` is `-π`, so `V_full(ψ) ≥ 2`; and only the pairs `(x, y)`
with `x = 0`, `y = 0` or `x = y` have a nonzero relation, each adding at
most `2`, so `V_full(ψ) ≤ 6 |G| < |G|²/8`. -/
example (hG : 49 ≤ Fintype.card G) :
    let V := divisionEnergy (Finset.univ : Finset (G × G))
    let ψ : EuclideanSpace ℝ G := PiLp.single 2 0 π
    4 ≤ Fintype.card G ∧ gradient V ψ = 0 ∧ V ψ ≠ 0 ∧ V ψ < Fintype.card G ^ 2 / 8 := by
  intro V ψ
  have hψ : ∀ x : G, ψ.ofLp x = if x = 0 then π else 0 := fun x => by
    simp [ψ]
  have hrel : ∀ a : G → ℤ, ⟪intVec a, ψ⟫ = (a 0 : ℤ) * π := fun a => by
    simp [ψ, inner_intVec]
  have hV : V ψ = ∑ p : G × G,
      (1 - cos (ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2))) := divisionEnergy_apply _ ψ
  refine ⟨by omega, ?_, ?_, ?_⟩
  · simp only [V, divisionEnergy]
    rw [gradient_phaseEnergy]
    exact Finset.sum_eq_zero fun r _ => by rw [hrel, sin_int_mul_pi, zero_smul]
  · have hnn : ∀ p : G × G, 0 ≤ 1 - cos (ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2)) :=
      fun p => sub_nonneg.2 (cos_le_one _)
    have h0 := Finset.single_le_sum (fun p _ => hnn p) (Finset.mem_univ ((0 : G), (0 : G)))
    rw [← hV] at h0
    simp only [sub_self, zero_sub, cos_neg, hψ, ite_true, cos_pi] at h0
    intro h
    linarith
  · have hb : ∀ p : G × G, 1 - cos (ψ.ofLp p.1 - ψ.ofLp p.2 - ψ.ofLp (p.1 - p.2)) ≤
        2 * ((if p.1 = 0 then 1 else 0) + (if p.2 = 0 then 1 else 0) +
          (if p.1 = p.2 then 1 else 0)) := by
      rintro ⟨x, y⟩
      dsimp only
      have hc := neg_one_le_cos (ψ.ofLp x - ψ.ofLp y - ψ.ofLp (x - y))
      by_cases hx : x = 0
      · rw [ite_eq_left hx]
        split_ifs <;> linarith
      by_cases hy : y = 0
      · rw [ite_eq_left hy]
        split_ifs <;> linarith
      by_cases hxy : x = y
      · rw [ite_eq_left hxy]
        split_ifs
        linarith
      simp [hψ, hx, hy, hxy, sub_eq_zero]
    have hsum : ∑ p : G × G, 2 * ((if p.1 = 0 then (1 : ℝ) else 0) +
        (if p.2 = 0 then 1 else 0) + (if p.1 = p.2 then 1 else 0)) = 6 * Fintype.card G := by
      rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_add_distrib, Fintype.sum_prod_type,
        Finset.sum_comm, Fintype.sum_prod_type, Fintype.sum_prod_type]
      simp only [Finset.sum_ite_eq, Finset.sum_ite_eq', Finset.mem_univ, ite_true,
        Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
      ring
    have hle : V ψ ≤ 6 * Fintype.card G := by
      rw [hV, ← hsum]
      exact Finset.sum_le_sum fun p _ => hb p
    have hn' : (49 : ℝ) ≤ Fintype.card G := by exact_mod_cast hG
    nlinarith

/-- The hypotheses of `ae_tendsto_characterPhase` hold in the Klein
four-group `ℤ/2 × ℤ/2` with Lebesgue measure and `η = 1/145`, as
`9 · 4² = 144`. -/
example : 4 ≤ Fintype.card (ZMod 2 × ZMod 2) ∧
    (volume : Measure (EuclideanSpace ℝ (ZMod 2 × ZMod 2))).IsAddHaarMeasure ∧
    (volume : Measure (EuclideanSpace ℝ (ZMod 2 × ZMod 2))) ≪ volume ∧ (0 : ℝ) < 1 / 145 ∧
    9 * (Fintype.card (ZMod 2 × ZMod 2) : ℝ) ^ 2 * (1 / 145) < 1 :=
  ⟨by simp, inferInstance, Measure.AbsolutelyContinuous.rfl, by norm_num,
    by norm_num [Fintype.card_prod, ZMod.card]⟩

end Transformer.PhaseModel.Abelian

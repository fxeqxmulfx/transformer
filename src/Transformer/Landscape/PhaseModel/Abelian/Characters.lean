module

public import Transformer.Landscape.PhaseModel.Abelian.Basin

/-!
# Phase model over a finite abelian group — critical points near a character

The phases `(arg χ(x))_x` of a character `χ` of `G` leave the energy of every
set of rows invariant (`divisionEnergy_add_characterPhase`), so the
estimates near the zero `0` hold near every character, as for `ℤ/n`
(`Landscape.PhaseModel.Characters`). For `|G| ≥ 4`, a local minimizer of `V_full` with
`Σ_x (1 - cos (ψ_x - arg χ(x))) < |G|/8` is the character `χ` up to periods
(`sub_characterPhase_mem_periods_of_isLocalMin`), and a critical point there
that is not is a strict saddle (`isStrictSaddle_of_sum_one_sub_cos_lt`): the
Hessian has a negative diagonal entry. The last example exhibits such a
saddle in `ℤ/5 × ℤ/5` near the trivial character: the phases `π e_0` keep
every relation in `πℤ`, so they are critical, at
`Σ_x (1 - cos ψ_x) = 2 < 25/8`.
-/

@[expose] public section

open Real

namespace Transformer.Landscape.PhaseModel.Abelian

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable {G : Type*} [AddCommGroup G] [Fintype G] [DecidableEq G]

/-- **A local minimizer of `V_full` near a character is that character** up
to periods: for `|G| ≥ 4`, if `ψ` is a local minimizer of `V_full` with
`Σ_x (1 - cos (ψ_x - arg χ(x))) < |G|/8`, then `ψ - (arg χ(x))_x ∈ 2πℤ^G`. -/
theorem sub_characterPhase_mem_periods_of_isLocalMin (hn : 4 ≤ Fintype.card G)
    {ψ : EuclideanSpace ℝ G} (hmin : IsLocalMin (divisionEnergy (Finset.univ : Finset (G × G))) ψ)
    {χ : AddChar G ℂ}
    (hE : ∑ x, (1 - cos (ψ.ofLp x - (characterPhase χ).ofLp x)) < Fintype.card G / 8) :
    ψ - characterPhase χ ∈ periods G := by
  have hcomp : divisionEnergy (Finset.univ : Finset (G × G)) ∘ (· + characterPhase χ) =
      divisionEnergy Finset.univ := funext (divisionEnergy_add_characterPhase _ χ)
  have hmin' : IsLocalMin (divisionEnergy (Finset.univ : Finset (G × G)))
      (ψ - characterPhase χ) := by
    have h : IsLocalMin (divisionEnergy (Finset.univ : Finset (G × G)))
        ((· + characterPhase χ) (ψ - characterPhase χ)) := by
      simpa only [sub_add_cancel] using hmin
    have h2 := IsLocalMin.comp_continuous (g := fun x => x + characterPhase χ) h
      (by fun_prop : Continuous fun x : EuclideanSpace ℝ G => x + characterPhase χ).continuousAt
    rwa [hcomp] at h2
  have hg := Landscape.SaddleAvoidance.gradient_eq_zero_of_isLocalMin hmin'
  refine mem_periods_of_forall_half_lt_cos hg (half_lt_cos_of_gradient_eq_zero hn hg
    (fun t => Landscape.SaddleAvoidance.inner_nonneg_of_isLocalMin
      ((contDiff_phaseEnergy _ (k := 1)).differentiable one_ne_zero)
      (hasFDerivAt_gradient_phaseEnergy _ _).differentiableAt.hasFDerivAt hmin' _) ?_)
  simpa only [WithLp.ofLp_sub, Pi.sub_apply] using hE

/-- **A critical point of `V_full` near a character that is not that
character is a strict saddle**: for `|G| ≥ 4`, if `∇V_full(ψ) = 0`,
`Σ_x (1 - cos (ψ_x - arg χ(x))) < |G|/8` and `ψ - (arg χ(x))_x ∉ 2πℤ^G`, then
the Hessian at `ψ` has a negative eigenvalue: a negative diagonal entry. -/
theorem isStrictSaddle_of_sum_one_sub_cos_lt (hn : 4 ≤ Fintype.card G) {ψ : EuclideanSpace ℝ G}
    (hg : gradient (divisionEnergy (Finset.univ : Finset (G × G))) ψ = 0) {χ : AddChar G ℂ}
    (hE : ∑ x, (1 - cos (ψ.ofLp x - (characterPhase χ).ofLp x)) < Fintype.card G / 8)
    (hψ : ψ - characterPhase χ ∉ periods G) :
    Landscape.SaddleAvoidance.IsStrictSaddle (divisionEnergy (Finset.univ : Finset (G × G))) ψ := by
  have hV := divisionEnergy_add_characterPhase (Finset.univ : Finset (G × G)) χ
  have hgrad := Landscape.SaddleAvoidance.gradient_add_eq hV (ψ - characterPhase χ)
  rw [sub_add_cancel] at hgrad
  have hH : fderiv ℝ (gradient (divisionEnergy (Finset.univ : Finset (G × G)))) ψ =
      fderiv ℝ (gradient (divisionEnergy Finset.univ)) (ψ - characterPhase χ) := by
    have h1 := fderiv_comp_add_right (𝕜 := ℝ)
      (f := gradient (divisionEnergy (Finset.univ : Finset (G × G))))
      (x := ψ - characterPhase χ) (characterPhase χ)
    rw [sub_add_cancel, funext (Landscape.SaddleAvoidance.gradient_add_eq hV)] at h1
    exact h1.symm
  by_contra hs
  refine hψ (mem_periods_of_forall_half_lt_cos (hgrad ▸ hg)
    (half_lt_cos_of_gradient_eq_zero hn (hgrad ▸ hg) (fun t => ?_) ?_))
  · by_contra ht
    push Not at ht
    exact hs (Landscape.SaddleAvoidance.isStrictSaddle_of_inner_neg
      (f := divisionEnergy (Finset.univ : Finset (G × G))) (contDiff_phaseEnergy _) hg
      (h := PiLp.single 2 t 1) (by rw [hH]; exact ht))
  · simpa only [WithLp.ofLp_sub, Pi.sub_apply] using hE

/-- The hypotheses of `sub_characterPhase_mem_periods_of_isLocalMin` hold in
the Klein four-group `ℤ/2 × ℤ/2` at the phases of every character `χ`, a
global minimizer. -/
example (χ : AddChar (ZMod 2 × ZMod 2) ℂ) :
    let V := divisionEnergy (Finset.univ : Finset ((ZMod 2 × ZMod 2) × (ZMod 2 × ZMod 2)))
    4 ≤ Fintype.card (ZMod 2 × ZMod 2) ∧ IsLocalMin V (characterPhase χ) ∧
      ∑ x, (1 - cos ((characterPhase χ).ofLp x - (characterPhase χ).ofLp x)) <
        (Fintype.card (ZMod 2 × ZMod 2) : ℝ) / 8 := by
  intro V
  refine ⟨by simp, Filter.Eventually.of_forall fun y => ?_, by simp⟩
  rw [show V (characterPhase χ) = 0 from divisionEnergy_characterPhase _ χ]
  exact phaseEnergy_nonneg _ y

/-- The hypotheses of `isStrictSaddle_of_sum_one_sub_cos_lt` hold in
`ℤ/5 × ℤ/5`, of order `25`, for the trivial character `0` and `ψ = π e_0`:
every relation of `ψ` is a multiple of `π`, so `ψ` is critical;
`Σ_x (1 - cos ψ_x) = 2 < 25/8`; and `π ∉ 2πℤ`. -/
example : let V := divisionEnergy (Finset.univ : Finset ((ZMod 5 × ZMod 5) × (ZMod 5 × ZMod 5)))
    let ψ : EuclideanSpace ℝ (ZMod 5 × ZMod 5) := PiLp.single 2 0 π
    4 ≤ Fintype.card (ZMod 5 × ZMod 5) ∧ gradient V ψ = 0 ∧
      ∑ x, (1 - cos (ψ.ofLp x - (characterPhase (0 : AddChar (ZMod 5 × ZMod 5) ℂ)).ofLp x)) <
        (Fintype.card (ZMod 5 × ZMod 5) : ℝ) / 8 ∧
      ψ - characterPhase (0 : AddChar (ZMod 5 × ZMod 5) ℂ) ∉ periods (ZMod 5 × ZMod 5) := by
  intro V ψ
  have hrel : ∀ a : ZMod 5 × ZMod 5 → ℤ, ⟪intVec a, ψ⟫ = (a 0 : ℤ) * π := fun a => by
    simp [ψ, inner_intVec]
  refine ⟨by simp, ?_, ?_, ?_⟩
  · simp only [V, divisionEnergy]
    rw [gradient_phaseEnergy]
    exact Finset.sum_eq_zero fun r _ => by rw [hrel, sin_int_mul_pi, zero_smul]
  · have h : ∀ x : ZMod 5 × ZMod 5,
        1 - cos (ψ.ofLp x - (characterPhase (0 : AddChar (ZMod 5 × ZMod 5) ℂ)).ofLp x) =
          if x = 0 then 2 else 0 := fun x => by
      by_cases hx : x = 0
      · simp only [ψ, hx, characterPhase_zero, PiLp.ofLp_single, Pi.single_eq_same,
          WithLp.ofLp_zero, Pi.zero_apply, sub_zero, cos_pi, ite_true]
        norm_num
      · simp [ψ, hx, characterPhase_zero]
    simp only [h, Finset.sum_ite_eq', Finset.mem_univ, ite_true, Fintype.card_prod, ZMod.card]
    norm_num
  · rw [characterPhase_zero, sub_zero]
    intro h
    obtain ⟨k, hk⟩ := h 0
    simp only [ψ, PiLp.ofLp_single, Pi.single_eq_same] at hk
    have h1 : (1 : ℝ) = 2 * k := by nlinarith [pi_pos]
    have h2 : (1 : ℤ) = 2 * k := by exact_mod_cast h1
    omega

end Transformer.Landscape.PhaseModel.Abelian

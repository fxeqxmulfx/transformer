/-
# Homogenized Transformers — the drift vanishes at Gaussian initialization

Consequence (ii) of assumption (G) of arXiv:2604.01978v1: "because `V` is
centered, the drift vanishes: `b_{ρ*}[μ] ≡ 0`".

Under (G) the head is `(V, A) = (V, W W'ᵀ)` with `V, W, W'` independent i.i.d.
Gaussian matrices, and a centered Gaussian is symmetric, so the law of the
head is invariant under `(V, A) ↦ (-V, A)` (`measurePreserving_negValue`).
The attention field `B_θ[μ](x)` is linear in `V` and its weights see only `A`,
so it is odd under that flip (`attnFieldOf_negValue`).  Its mean is therefore
its own negative, and vanishes (`meanFieldOf_gaussian`); no integrability is
needed, since an integral and the integral of its negative are negatives of
each other whether or not the integrand is integrable.

Source: arXiv:2604.01978v1, §2.3.3, item (ii).
-/

import Transformer.Homogenized.HeadLawRotation

open scoped NNReal
open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- Under assumption (G), the head law is the image under `headOfBlocks` of the
i.i.d. Gaussian law of the entries of `(V, W, W')`.
arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
theorem map_headOfBlocks_pi {d : ℕ} {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) :
    (Measure.pi fun p => gaussianReal 0 (blockSigma (d := d) σV σA p ^ 2)).map headOfBlocks =
      ρ := by
  obtain ⟨Ω, _, P, Vr, Wr, Wr', _, hV, hW, hW', hind, hmV, hmW, hmW', rfl⟩ := hρ
  set F : Ω → Fin 3 × Fin d × Fin d → ℝ := fun ω p =>
    if p.1 = 0 then Vr ω p.2.1 p.2.2 else if p.1 = 1 then Wr ω p.2.1 p.2.2 else Wr' ω p.2.1 p.2.2
  have hFp : ∀ p, Measurable fun ω => F ω p := fun p => by
    simp only [F]
    split_ifs
    exacts [(hV.eval).eval, (hW.eval).eval, (hW'.eval).eval]
  have hlaw : P.map F = Measure.pi fun p => gaussianReal 0 (blockSigma σV σA p ^ 2) := by
    rw [show F = fun ω p => F ω p from rfl,
      hind.map_fun_eq_pi_map fun p => (hFp p).aemeasurable]
    congr 1; funext p
    obtain ⟨a, i, j⟩ := p
    fin_cases a
    · simpa [F, blockSigma] using hmV i j
    · simpa [F, blockSigma] using hmW i j
    · simpa [F, blockSigma] using hmW' i j
  rw [← hlaw, Measure.map_map measurable_headOfBlocks (Measurable.of_eval hFp)]
  rfl

/-- `(V, A) ↦ (-V, A)`, as a measurable equivalence of head parameters. -/
noncomputable def negValue (d : ℕ) : HeadParam d ≃ᵐ HeadParam d :=
  MeasurableEquiv.prodCongr (MeasurableEquiv.neg _) (MeasurableEquiv.refl _)

theorem negValue_apply {d : ℕ} (θ : HeadParam d) : negValue d θ = (-θ.1, θ.2) := rfl

/-- `V ↦ -V` on the value block of `(V, W, W')`, the identity on the two
query-key blocks. -/
def blockNeg {d : ℕ} (z : Fin 3 × Fin d × Fin d → ℝ) : Fin 3 × Fin d × Fin d → ℝ :=
  fun p => if p.1 = 0 then -z p else z p

theorem measurable_blockNeg {d : ℕ} : Measurable (blockNeg (d := d)) := by
  refine Measurable.of_eval fun p => ?_
  by_cases h : p.1 = 0
  · simp only [blockNeg, h, ite_true]
    fun_prop
  · simpa only [blockNeg, h, ite_false] using measurable_pi_apply p

/-- Flipping the value block is flipping the value matrix.
arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
theorem negValue_headOfBlocks {d : ℕ} (z : Fin 3 × Fin d × Fin d → ℝ) :
    negValue d (headOfBlocks z) = headOfBlocks (blockNeg z) := by
  rw [negValue_apply, headOfBlocks, headOfBlocks]
  refine Prod.ext ?_ ?_
  · ext i j
    simp [blockNeg]
  · simp [blockNeg, show (1 : Fin 3) ≠ 0 by decide, show (2 : Fin 3) ≠ 0 by decide]

/-- The Gaussian law of `(V, W, W')` of assumption (G) is invariant under
`V ↦ -V`: a centered Gaussian is symmetric.
arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
theorem map_blockNeg_pi {d : ℕ} (σV σA : ℝ≥0) :
    (Measure.pi fun p => gaussianReal 0 (blockSigma (d := d) σV σA p ^ 2)).map blockNeg =
      Measure.pi fun p => gaussianReal 0 (blockSigma (d := d) σV σA p ^ 2) := by
  set γ := fun p => gaussianReal 0 (blockSigma (d := d) σV σA p ^ 2)
  set g : Fin 3 × Fin d × Fin d → ℝ → ℝ := fun p t => if p.1 = 0 then -t else t
  have hg : ∀ p, Measurable (g p) := fun p => by
    by_cases h : p.1 = 0
    · simpa only [g, h, ite_true] using measurable_neg
    · simpa only [g, h, ite_false] using measurable_id'
  have hmap : ∀ p, (γ p).map (g p) = γ p := fun p => by
    by_cases h : p.1 = 0
    · simp only [g, h, ite_true, γ]
      rw [gaussianReal_map_neg, neg_zero]
    · simp only [g, h, ite_false]
      exact Measure.map_id'
  have : ∀ p, SigmaFinite ((γ p).map (g p)) := fun p => by
    rw [hmap p]
    infer_instance
  rw [show (blockNeg : (Fin 3 × Fin d × Fin d → ℝ) → _) = fun z p => g p (z p) from rfl,
    Measure.pi_map_pi fun p => (hg p).aemeasurable]
  congr 1
  funext p
  exact hmap p

/-- **The head law is invariant under `V ↦ -V`.**  Under assumption (G), `V` is
a centered Gaussian matrix independent of `A`, so `(V, A)` and `(-V, A)` have
the same law.

Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
theorem measurePreserving_negValue {d : ℕ} {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) : MeasurePreserving (negValue d) ρ ρ := by
  refine ⟨(negValue d).measurable, ?_⟩
  rw [← map_headOfBlocks_pi hρ,
    Measure.map_map (negValue d).measurable measurable_headOfBlocks,
    show negValue d ∘ headOfBlocks = headOfBlocks ∘ blockNeg from
      funext negValue_headOfBlocks,
    ← Measure.map_map measurable_headOfBlocks measurable_blockNeg, map_blockNeg_pi]

/-- The hypothesis of `measurePreserving_negValue` is satisfiable. -/
example (d : ℕ) : IsGaussianHeadLaw d 0 0 (Measure.dirac (0 : HeadParam d)) :=
  isGaussianHeadLaw_dirac_zero d

/-- The attention field is odd in the value matrix: its weights see only `A`,
and it is linear in `V`.  arXiv:2604.01978v1,
`EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
theorem attnFieldOf_negValue {d : ℕ} (β : ℝ) (θ : HeadParam d) (μ : Measure (EucSpace d))
    (x : EucSpace d) : attnFieldOf β (negValue d θ) μ x = -attnFieldOf β θ μ x := by
  have hw : ∀ y, attnWeight β (negValue d θ) x y = attnWeight β θ x y := fun _ => rfl
  have hv : ∀ y, valueMap (negValue d θ) y = -valueMap θ y := fun y => by
    simp [valueMap, negValue_apply]
  simp only [attnFieldOf, hw, hv, smul_neg, integral_neg]

/-- **Consequence (ii) of (G), `b_{ρ*}[μ] ≡ 0`.**  Under
`eq: tformers.at.initialization` the mean field vanishes identically, at every
measure `μ` of tokens and every point `x`: the head law is invariant under
`V ↦ -V` and `B_θ[μ](x)` is odd under it.

Source: arXiv:2604.01978v1, §2.3.3, item (ii). -/
theorem meanFieldOf_gaussian {d : ℕ} (β : ℝ) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (μ : Measure (EucSpace d)) (x : EucSpace d) :
    meanFieldOf β ρ μ x = 0 := by
  have h := (measurePreserving_negValue hρ).integral_comp' fun θ => attnFieldOf β θ μ x
  simp only [attnFieldOf_negValue, integral_neg] at h
  have h2 : (2 : ℝ) • meanFieldOf β ρ μ x = 0 := by
    rw [two_smul, meanFieldOf]
    nth_rw 1 [← h]
    exact neg_add_cancel _
  exact (smul_eq_zero.mp h2).resolve_left two_ne_zero

/-- The hypothesis of `meanFieldOf_gaussian` is satisfiable. -/
example (d : ℕ) : IsGaussianHeadLaw d 0 0 (Measure.dirac (0 : HeadParam d)) :=
  isGaussianHeadLaw_dirac_zero d

/-- **Consequence (ii) of (G), at a configuration.**  The mean field of the
tokens vanishes: `b_{ρ*}[μ_X](z) = 0` for every `z`.

Source: arXiv:2604.01978v1, §2.3.3, item (ii). -/
theorem meanField_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0} {ρ : Measure (HeadParam d)}
    (hρ : IsGaussianHeadLaw d σV σA ρ) (x : Idx n → EucSpace d) (z : EucSpace d) :
    meanField β ρ x z = 0 := by
  rw [← meanFieldOf_empMeasure, meanFieldOf_gaussian β hρ]

/-- The hypothesis of `meanField_gaussian` is satisfiable. -/
example (d : ℕ) : IsGaussianHeadLaw d 0 0 (Measure.dirac (0 : HeadParam d)) :=
  isGaussianHeadLaw_dirac_zero d

/-- **Consequence (ii) of (G), the projected drift.**  Because `V` is centered
and independent of `A`, the drift of `eq: first.sde` vanishes identically:
`b(X)_i = Proj_{x_i} b_{ρ*}[μ_X](x_i) = 0`.  The diffusive regime
`αηL = Θ(1)` is therefore driven entirely by the random fluctuations, and the
ballistic regime is static.

The source's claim is the unprojected `b_{ρ*}[μ] ≡ 0`, which is
`meanFieldOf_gaussian`; this is its projection at the empirical measure.

Source: arXiv:2604.01978v1, §2.3.3, item (ii). -/
theorem bField_gaussian {d n : ℕ} (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (x : Idx n → EucSpace d) (i : Idx n) :
    bField β ρ x i = 0 := by
  rw [bField, meanField_gaussian β hρ, proj, inner_zero_right, zero_smul, sub_zero]

/-- The hypothesis of `bField_gaussian` is satisfiable. -/
example (d : ℕ) : IsGaussianHeadLaw d 0 0 (Measure.dirac (0 : HeadParam d)) :=
  isGaussianHeadLaw_dirac_zero d

end Homogenized
end Transformer

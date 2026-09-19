/-
# Causal attention — Counting strong R'enyi centers (§ "R'enyi centers vs
strong R'enyi centers" of 2411.04990v2)

The meta-stable cluster nuclei of `Causal/Metastability.lean` are the *strong*
R'enyi centers of the initial configuration, and the survey counts them: for an
i.i.d. sequence on `𝕊^{d-1}` whose spherical caps all carry the same mass, the
expected number of strong centers at separation `δ` is the reciprocal of that
cap mass.

The unlabelled lemma of that subsection is proved here: by independence, the
`k`-th point is a strong centre exactly when its `k` predecessors all avoid its
cap, an event of probability `(1 - σ^{d-1}(B_δ))^k`, and the expected count is
the geometric series of those probabilities.  The cap masses it specializes to
in dimensions two and three are `Transformer.Causal.CapMass`.
-/

import Transformer.Basic
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Constructions.HaarToSphere
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Causal

variable (d : ℕ)

/-- The **geodesic distance** on the unit sphere: the angle between two unit
vectors, `arccos ⟨x, y⟩`.  This is the `dist` of the survey's R'enyi parking
definitions, as opposed to the ambient distance used in
`Transformer.Causal.isStrongRenyiCenters`.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
noncomputable def geoDist (x y : SSphere d) : ℝ :=
  Real.arccos (inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))

/-- The geodesic distance is continuous, hence measurable: it is `arccos` of
the inner product of two continuously varying unit vectors. -/
theorem continuous_geoDist :
    Continuous fun p : SSphere d × SSphere d => geoDist d p.1 p.2 :=
  Real.continuous_arccos.comp
    ((continuous_subtype_val.comp continuous_fst).inner
      (continuous_subtype_val.comp continuous_snd))

/-- The **probability that the `k`-th point of an i.i.d. sequence is a strong
R'enyi center** at separation `δ`: it is farther than `δ` from every one of its
predecessors.

The event involves only the first `k + 1` points, so it is written against the
finite product measure `μ^{⊗(k+1)}` rather than an infinite product.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
noncomputable def strongCenterProb
    (μ : Measure (SSphere d)) (δ : ℝ) (k : ℕ) : ℝ≥0∞ :=
  (Measure.pi fun _ : Fin (k + 1) => μ)
    { x : Fin (k + 1) → SSphere d |
        ∀ i : Fin (k + 1), (i : ℕ) < k → δ < geoDist d (x (Fin.last k)) (x i) }

/-- **Lemma (strong R'enyi parking count).**

For an infinite i.i.d. sequence on `𝕊^{d-1}` sampled from a distribution `μ`
all of whose geodesic caps of radius `δ` carry the same mass
`σ^{d-1}(B_δ) > 0` — the survey's "spherically harmonic" distributions, the
uniform measure among them — the average number of points chosen by strong
R'enyi parking is

  `1 / σ^{d-1}(B_δ)`.

The average is the sum over `k` of the probabilities that the `k`-th point is
chosen, which is what the survey's proof computes; the constancy of the cap
mass is carried as the hypothesis `hharm` rather than derived from a symmetry
assumption on `μ`.

Source: arXiv:2411.04990v2, §"R'enyi centers vs strong R'enyi centers". -/
theorem strong_renyi_expected_count
    (μ : Measure (SSphere d)) [IsProbabilityMeasure μ] (δ capMass : ℝ)
    (hcap : 0 < capMass)
    (hharm : ∀ x : SSphere d,
      μ { y : SSphere d | geoDist d x y ≤ δ } = ENNReal.ofReal capMass) :
    ∑' k : ℕ, strongCenterProb d μ δ k = ENNReal.ofReal capMass⁻¹ := by
  classical
  set c : ℝ≥0∞ := ENNReal.ofReal capMass with hc
  -- a cap carries at most the whole mass, so `1 - (1 - c) = c` below
  have hne : Nonempty (SSphere d) := by
    by_contra h
    rw [not_nonempty_iff] at h
    have huniv : μ Set.univ = 1 := measure_univ
    rw [Set.univ_eq_empty_iff.mpr h, measure_empty] at huniv
    exact zero_ne_one huniv
  obtain ⟨x₀⟩ := hne
  have hcle : c ≤ 1 := by rw [← hharm x₀]; exact prob_le_one
  -- the mass outside a cap
  have hcompl : ∀ x : SSphere d, μ { y : SSphere d | δ < geoDist d x y } = 1 - c := by
    intro x
    have hmeas : MeasurableSet { y : SSphere d | geoDist d x y ≤ δ } :=
      measurableSet_le ((continuous_geoDist d).comp
        (continuous_const.prodMk continuous_id)).measurable measurable_const
    have hset : { y : SSphere d | δ < geoDist d x y } = { y | geoDist d x y ≤ δ }ᶜ := by
      ext y; simp [not_le]
    rw [hset, prob_compl_eq_one_sub hmeas, hharm x, hc]
  -- the `k`-th point is a strong centre exactly when all `k` predecessors miss its cap
  have hstep : ∀ k : ℕ, strongCenterProb d μ δ k = (1 - c) ^ k := by
    intro k
    set T : Set (SSphere d × (Fin k → SSphere d)) :=
      { p | ∀ j : Fin k, δ < geoDist d p.1 (p.2 j) } with hT
    have hTmeas : MeasurableSet T := by
      have hinter : T = ⋂ j : Fin k, { p : SSphere d × (Fin k → SSphere d) |
          δ < geoDist d p.1 (p.2 j) } := by ext p; simp [hT]
      rw [hinter]
      refine MeasurableSet.iInter fun j => measurableSet_lt measurable_const ?_
      exact ((continuous_geoDist d).comp
        (continuous_fst.prodMk ((continuous_apply j).comp continuous_snd))).measurable
    have hpre : { x : Fin (k + 1) → SSphere d |
        ∀ i : Fin (k + 1), (i : ℕ) < k → δ < geoDist d (x (Fin.last k)) (x i) } =
          MeasurableEquiv.piFinSuccAbove (fun _ : Fin (k + 1) => SSphere d) (Fin.last k) ⁻¹' T := by
      ext x
      simp only [Set.mem_ofPred_eq, Set.mem_preimage, hT,
        MeasurableEquiv.piFinSuccAbove_apply, Fin.insertNthEquiv_symm_apply, Fin.removeNth,
        Fin.succAbove_last]
      constructor
      · intro h j
        exact h j.castSucc (by simp [j.isLt])
      · intro h i hi
        have := h ⟨(i : ℕ), hi⟩
        rwa [show (Fin.castSucc ⟨(i : ℕ), hi⟩ : Fin (k + 1)) = i from Fin.ext rfl] at this
    rw [strongCenterProb, hpre,
      (measurePreserving_piFinSuccAbove (fun _ : Fin (k + 1) => μ)
        (Fin.last k)).measure_preimage hTmeas.nullMeasurableSet,
      Measure.prod_apply hTmeas]
    have hsec : ∀ x : SSphere d, Prod.mk x ⁻¹' T
        = Set.univ.pi fun _ : Fin k => { y : SSphere d | δ < geoDist d x y } := by
      intro x; ext y; simp [hT]
    calc ∫⁻ x, (Measure.pi fun _ : Fin k => μ) (Prod.mk x ⁻¹' T) ∂μ
        = ∫⁻ _ : SSphere d, (1 - c) ^ k ∂μ := by
          refine lintegral_congr fun x => ?_
          rw [hsec x, Measure.pi_pi]
          simp [hcompl x]
      _ = (1 - c) ^ k := by simp
  calc ∑' k : ℕ, strongCenterProb d μ δ k = ∑' k : ℕ, (1 - c) ^ k := tsum_congr hstep
    _ = (1 - (1 - c))⁻¹ := ENNReal.tsum_geometric _
    _ = c⁻¹ := by rw [ENNReal.sub_sub_cancel ENNReal.one_ne_top hcle]
    _ = ENNReal.ofReal capMass⁻¹ := by rw [hc, ENNReal.ofReal_inv_of_pos hcap]

/-- The unit vector `e_0` of `EucSpace 1`, as a point of the sphere. -/
noncomputable def northPole : SSphere 1 :=
  ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩

/-- The hypotheses of `strong_renyi_expected_count` are satisfiable: at
`δ = π` every geodesic cap is the whole sphere, so any probability measure —
here a Dirac mass on `𝕊^0` — has constant cap mass `1`. -/
example :
    (0 : ℝ) < 1 ∧ ∀ x : SSphere 1,
      (Measure.dirac northPole) { y : SSphere 1 | geoDist 1 x y ≤ Real.pi }
        = ENNReal.ofReal 1 := by
  refine ⟨one_pos, fun x => ?_⟩
  have hset : { y : SSphere 1 | geoDist 1 x y ≤ Real.pi } = Set.univ := by
    ext y
    simp [geoDist, Real.arccos_le_pi]
  rw [hset, ENNReal.ofReal_one]
  exact measure_univ

end Causal
end Transformer

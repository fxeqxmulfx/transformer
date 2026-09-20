/-
# The number of modes of a Gaussian KDE — a field satisfying Kac-Rice

The hypotheses of `thm:kac-rice` are satisfiable: the random line
`Ψ_z(t) = z₁ + z₂ t` with `z` a standard Gaussian vector in the plane meets
all four of them at level `u = 0` over `T = {0}`.

The source quotes `thm:kac-rice` without a witness; this file supplies one, so
that the four conditions of `IsKacRiceField` are known to be consistent.  The
line is the simplest non-degenerate choice: at `t = 0` the pair
`(Ψ(0), Ψ'(0))` is `z` itself, so the joint density asked for by condition 3
is the Gaussian density of `z`, and `Ψ'` is constant in `t`, so the modulus of
continuity of condition 4 vanishes identically.

Source: arXiv:2412.09080v3, `thm:kac-rice`, items 1-4.
-/

import Transformer.Modes.Section2_KacRice

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory Filter Topology Asymptotics

namespace Transformer
namespace Modes

/-- The random line `Ψ_z(t) = z₁ + z₂ t`, the simplest non-degenerate random
field: at `t = 0` the pair `(Ψ(0), Ψ'(0))` is `z` itself, so its joint law is
whatever law `z` is given. -/
noncomputable def randomLine (z : ℝ × ℝ) (t : ℝ) : ℝ := z.1 + z.2 * t

/-- A standard Gaussian vector in the plane. -/
noncomputable def gaussianPair : Measure (ℝ × ℝ) :=
  (gaussianReal 0 1).prod (gaussianReal 0 1)

instance : IsProbabilityMeasure gaussianPair := by
  unfold gaussianPair; infer_instance

theorem deriv_randomLine (z : ℝ × ℝ) : deriv (randomLine z) = fun _ => z.2 := by
  funext t
  have h : HasDerivAt (fun t : ℝ => z.1 + z.2 * t) z.2 t := by
    simpa using ((hasDerivAt_id t).const_mul z.2).const_add z.1
  exact h.deriv

theorem randomLine_zero (z : ℝ × ℝ) : randomLine z 0 = z.1 := by simp [randomLine]

theorem map_fst_gaussianPair : Measure.map Prod.fst gaussianPair = gaussianReal 0 1 :=
  (MeasureTheory.measurePreserving_fst (μ := gaussianReal 0 1) (ν := gaussianReal 0 1)).map_eq

theorem map_snd_gaussianPair : Measure.map Prod.snd gaussianPair = gaussianReal 0 1 :=
  (MeasureTheory.measurePreserving_snd (μ := gaussianReal 0 1) (ν := gaussianReal 0 1)).map_eq

theorem memLp_fst_gaussianPair : MemLp (Prod.fst : ℝ × ℝ → ℝ) 2 gaussianPair := by
  rw [gaussianPair]
  simpa using (memLp_id_gaussianReal (μ := 0) (v := 1) 2).comp_measurePreserving
    (MeasureTheory.measurePreserving_fst (μ := gaussianReal 0 1) (ν := gaussianReal 0 1))

theorem memLp_snd_gaussianPair : MemLp (Prod.snd : ℝ × ℝ → ℝ) 2 gaussianPair := by
  rw [gaussianPair]
  simpa using (memLp_id_gaussianReal (μ := 0) (v := 1) 2).comp_measurePreserving
    (MeasureTheory.measurePreserving_snd (μ := gaussianReal 0 1) (ν := gaussianReal 0 1))

/-- The density of `gaussianPair` is the product of the two Gaussian
densities. -/
theorem gaussianPair_eq_withDensity : gaussianPair = volume.withDensity
    fun z : ℝ × ℝ => ENNReal.ofReal (gaussianPDFReal 0 1 z.1 * gaussianPDFReal 0 1 z.2) := by
  rw [gaussianPair, gaussianReal_of_var_ne_zero 0 one_ne_zero,
    prod_withDensity (measurable_gaussianPDF 0 1) (measurable_gaussianPDF 0 1),
    ← Measure.volume_eq_prod]
  refine withDensity_congr_ae (Filter.Eventually.of_forall fun z => ?_)
  show ENNReal.ofReal (gaussianPDFReal 0 1 z.1) * ENNReal.ofReal (gaussianPDFReal 0 1 z.2)
      = ENNReal.ofReal (gaussianPDFReal 0 1 z.1 * gaussianPDFReal 0 1 z.2)
  rw [ENNReal.ofReal_mul (gaussianPDFReal_nonneg 0 1 z.1)]

theorem continuous_gaussianPDFReal_std : Continuous (gaussianPDFReal 0 1) := by
  unfold gaussianPDFReal
  fun_prop

/-- **The hypotheses of `kacRice` are satisfiable.**  The random line above,
at level `u = 0` over the compact `T = {0}`, has `Ψ(0) = z₁` standard Gaussian
and `(Ψ(0), Ψ'(0)) = z` standard Gaussian in the plane; its derivative is
constant in `t`, so its modulus of continuity vanishes. -/
example : IsCompact ({0} : Set ℝ) ∧
    IsKacRiceField gaussianPair randomLine 0 {0} (fun _ x => gaussianPDFReal 0 1 x)
      (fun _ x y => gaussianPDFReal 0 1 x * gaussianPDFReal 0 1 y) := by
  refine ⟨isCompact_singleton, ?_⟩
  have hfst : (fun z : ℝ × ℝ => randomLine z 0) = Prod.fst := funext randomLine_zero
  have hsnd : (fun z : ℝ × ℝ => deriv (randomLine z) 0) = Prod.snd := by
    funext z; rw [deriv_randomLine]
  constructor
  · exact Filter.Eventually.of_forall fun _ =>
      contDiff_const.add (contDiff_const.mul contDiff_id)
  · intro t ht
    rw [Set.mem_singleton_iff] at ht; subst ht
    rw [hfst]; exact memLp_fst_gaussianPair
  · intro t ht
    rw [Set.mem_singleton_iff] at ht; subst ht
    rw [hsnd]; exact memLp_snd_gaussianPair
  · intro t ht
    rw [Set.mem_singleton_iff] at ht; subst ht
    rw [hfst, map_fst_gaussianPair, gaussianReal_of_var_ne_zero 0 one_ne_zero]
    rfl
  · exact ⟨Set.univ, Filter.univ_mem,
      (continuous_gaussianPDFReal_std.comp continuous_snd).continuousOn⟩
  · intro t ht
    rw [Set.mem_singleton_iff] at ht; subst ht
    have hid : (fun z : ℝ × ℝ => (randomLine z 0, deriv (randomLine z) 0)) = id := by
      funext z; simp [randomLine_zero, deriv_randomLine]
    rw [hid, Measure.map_id]
    exact gaussianPair_eq_withDensity
  · have hc : Continuous fun q : ℝ × ℝ × ℝ =>
        gaussianPDFReal 0 1 q.2.1 * gaussianPDFReal 0 1 q.2.2 :=
      (continuous_gaussianPDFReal_std.comp (continuous_fst.comp continuous_snd)).mul
        (continuous_gaussianPDFReal_std.comp (continuous_snd.comp continuous_snd))
    exact ⟨Set.univ, Filter.univ_mem, hc.continuousOn⟩
  · intro ε hε
    have hset : ∀ η : ℝ, {z : ℝ × ℝ |
        ENNReal.ofReal ε < modulusOfContinuity (deriv (randomLine z)) η} = ∅ := by
      intro η
      simp [deriv_randomLine]
    have : (fun η : ℝ => (gaussianPair
        {z | ENNReal.ofReal ε < modulusOfContinuity (deriv (randomLine z)) η}).toReal)
        = fun _ => (0 : ℝ) := by
      funext η; rw [hset η]; simp
    rw [this]
    exact isBigO_zero _ _

end Modes
end Transformer

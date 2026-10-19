module

public import Transformer.Wiener.Defs
public import Mathlib.MeasureTheory.Function.L2Space
public import Mathlib.Probability.BrownianMotion.Basic
public import Mathlib.Probability.Distributions.Gaussian.HasGaussianLaw.Basic
public import Mathlib.Probability.ConditionalExpectation
public import Mathlib.Probability.Independence.Integration
public import Mathlib.Probability.Martingale.Basic

/-!
# Brownian motion and the cylindrical Wiener process: moments and martingale property

*Homogenized Transformers*, arXiv:2604.01978v1, App. A.2 `sec: cylindrical.wiener`: for an
`(ℱ t)`-Brownian motion `B` of variance `v` (`IsBrownian`, `Wiener/Defs.lean`), `B t` is a
centred Gaussian of variance `v t`, `𝔼[B s B t] = v (s ∧ t)`, and `B` is an `ℱ`-martingale. With
`v = 1` these are exactly Mathlib's Brownian motions `IsBrownianReal` whose increments are
independent of `ℱ`, and `c • B` is Brownian of variance `c² v`. For a cylindrical Wiener process
`W`, `W t 0 = 0` almost surely and `𝔼[W(t)[h] W(s)[g]] = (s ∧ t) ⟪h, g⟫`, the source's
"Equivalently" (l. 3378).
-/

@[expose] public section

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal InnerProductSpace

namespace Transformer.Wiener

variable {Ω Θ : Type*} {m : MeasurableSpace Ω} {mΘ : MeasurableSpace Θ} {P : Measure Ω}
  {ℱ : Filtration ℝ≥0 m} {ρ : Measure Θ} {v : ℝ≥0} {B : ℝ≥0 → Ω → ℝ}

namespace IsBrownian

theorem measurable (hB : IsBrownian ℱ P v B) (t : ℝ≥0) : Measurable (B t) :=
  ((hB.adapted t).mono (ℱ.le t)).measurable

/-- App. A.2: `B t` is a centred Gaussian of variance `v t`. -/
theorem hasLaw_eval (hB : IsBrownian ℱ P v B) (t : ℝ≥0) :
    HasLaw (B t) (gaussianReal 0 (v * t)) P := by
  simpa only [tsub_zero] using (hB.hasLaw 0 t zero_le).congr (hB.zero.mono fun ω hω => by simp [hω])

theorem memLp_two (hB : IsBrownian ℱ P v B) (t : ℝ≥0) : MemLp (B t) 2 P :=
  (hB.hasLaw_eval t).hasGaussianLaw.memLp_two

theorem integrable (hB : IsBrownian ℱ P v B) (t : ℝ≥0) : Integrable (B t) P :=
  (hB.hasLaw_eval t).hasGaussianLaw.integrable

/-- App. A.2: `𝔼[B t] = 0`. -/
theorem integral_eq_zero (hB : IsBrownian ℱ P v B) (t : ℝ≥0) : ∫ ω, B t ω ∂P = 0 := by
  rw [(hB.hasLaw_eval t).integral_eq, integral_id_gaussianReal]

/-- App. A.2: `𝔼[(B t)²] = Var(B t) = v t`. -/
theorem integral_sq (hB : IsBrownian ℱ P v B) (t : ℝ≥0) : ∫ ω, B t ω ^ 2 ∂P = v * t := by
  rw [← variance_of_integral_eq_zero (hB.hasLaw_eval t).aemeasurable (hB.integral_eq_zero t),
    (hB.hasLaw_eval t).variance_eq, variance_id_gaussianReal, NNReal.coe_mul]

/-- App. A.2: an increment after `s` is orthogonal to every `ℱ s`-measurable variable. -/
theorem integral_sub_mul (hB : IsBrownian ℱ P v B) {s t : ℝ≥0} (hst : s ≤ t) {Y : Ω → ℝ}
    (hY : StronglyMeasurable[ℱ s] Y) : ∫ ω, (B t ω - B s ω) * Y ω ∂P = 0 := by
  have hind : IndepFun (B t - B s) Y P := (IndepFun_iff_Indep _ _ _).2
    (indep_of_indep_of_le_right (hB.indep s t hst) hY.measurable.comap_le)
  calc ∫ ω, (B t ω - B s ω) * Y ω ∂P = (∫ ω, (B t - B s) ω ∂P) * ∫ ω, Y ω ∂P :=
        hind.integral_mul_eq_mul_integral
          ((hB.measurable t).sub (hB.measurable s)).aestronglyMeasurable
          (hY.mono (ℱ.le s)).aestronglyMeasurable
    _ = 0 := by rw [(hB.hasLaw s t hst).integral_eq, integral_id_gaussianReal, zero_mul]

/-- App. A.2: `𝔼[B s B t] = v (s ∧ t)`. -/
theorem integral_mul (hB : IsBrownian ℱ P v B) (s t : ℝ≥0) :
    ∫ ω, B s ω * B t ω ∂P = v * min s t := by
  wlog hst : s ≤ t generalizing s t
  · rw [min_comm, ← this t s (le_of_not_ge hst)]
    simp_rw [mul_comm (B s _)]
  calc ∫ ω, B s ω * B t ω ∂P = ∫ ω, (B t ω - B s ω) * B s ω + B s ω ^ 2 ∂P :=
        integral_congr_ae (ae_of_all _ fun _ => by ring)
    _ = (∫ ω, (B t ω - B s ω) * B s ω ∂P) + ∫ ω, B s ω ^ 2 ∂P :=
        integral_add (((hB.memLp_two t).sub (hB.memLp_two s)).integrable_mul (hB.memLp_two s))
          (hB.memLp_two s).integrable_sq
    _ = v * min s t := by
        rw [hB.integral_sub_mul hst (hB.adapted s), zero_add, hB.integral_sq, min_eq_left hst]

/-- App. A.2: an `(ℱ t)`-Brownian motion is an `ℱ`-martingale, its increments after `s` being
centred and independent of `ℱ s`. -/
theorem martingale (hB : IsBrownian ℱ P v B) : Martingale B ℱ P := by
  have := hB.isProbabilityMeasure
  refine ⟨hB.adapted, fun s t hst => ?_⟩
  have hind : P[B t - B s | ℱ s] =ᵐ[P] fun _ => ∫ ω, (B t - B s) ω ∂P :=
    condExp_indep_eq ((hB.measurable t).sub (hB.measurable s)).comap_le (ℱ.le s)
      (comap_measurable (B t - B s)).stronglyMeasurable (hB.indep s t hst)
  have hadd := condExp_add ((hB.integrable t).sub (hB.integrable s)) (hB.integrable s) (ℱ s)
  rw [sub_add_cancel, condExp_of_stronglyMeasurable (ℱ.le s) (hB.adapted s)
    (hB.integrable s)] at hadd
  filter_upwards [hadd, hind] with ω h1 h2
  rw [h1, Pi.add_apply, h2, (hB.hasLaw s t hst).integral_eq, integral_id_gaussianReal, zero_add]

/-- App. A.2: the increments of an `(ℱ t)`-Brownian motion are independent, each one being
independent of `ℱ` at its left end, which carries the earlier ones. -/
theorem hasIndepIncrements (hB : IsBrownian ℱ P v B) : HasIndepIncrements B P := by
  have := hB.isProbabilityMeasure
  refine hasIndepIncrements_iff_nat.2 fun τ hτ =>
    iIndepFun_iff_measure_inter_preimage_eq_mul.2 fun S sets hsets => ?_
  induction S using Finset.induction_on_max with
  | empty => simp
  | insert a S hlt ih =>
    rw [Finset.prod_insert fun ha => lt_irrefl a (hlt a ha), Finset.set_biInter_insert,
      ← ih fun i hi => hsets i (Finset.mem_insert_of_mem hi)]
    refine (Indep_iff _ _ P).1 (hB.indep (τ a) (τ (a + 1)) (hτ a.le_succ)) _ _
      ⟨sets a, hsets a (Finset.mem_insert_self a S), rfl⟩
      (Finset.measurableSet_biInter S fun i hi => ?_)
    have hle : τ (i + 1) ≤ τ a := hτ (Nat.succ_le_of_lt (hlt i hi))
    exact (((hB.adapted _).measurable.mono (ℱ.mono hle) le_rfl).sub
      ((hB.adapted _).measurable.mono (ℱ.mono ((hτ i.le_succ).trans hle)) le_rfl))
      (hsets i (Finset.mem_insert_of_mem hi))

/-- App. A.2, scaling: `c • B` is an `(ℱ t)`-Brownian motion of variance `c² v`. -/
theorem smul (hB : IsBrownian ℱ P v B) (c : ℝ) : IsBrownian ℱ P (‖c‖₊ ^ 2 * v) (c • B) where
  zero := hB.zero.mono fun ω hω => by simp [hω]
  continuous := hB.continuous.mono fun ω hω => hω.const_smul c
  adapted t := (hB.adapted t).const_smul c
  indep s t hst := by
    rw [show (c • B) t - (c • B) s = (c * ·) ∘ (B t - B s) from funext fun ω => by simp [mul_sub]]
    exact indep_of_indep_of_le_left (hB.indep s t hst)
      ((measurable_const_mul c).comp (comap_measurable _)).comap_le
  hasLaw s t hst := by
    have h := gaussianReal_const_mul (hB.hasLaw s t hst) c
    rw [mul_zero, ← mul_assoc, show NNReal.mk (c ^ 2) (sq_nonneg c) = ‖c‖₊ ^ 2 from
      NNReal.eq (by simp)] at h
    exact h.congr (ae_of_all _ fun ω => by simp [mul_sub])

end IsBrownian

/-- App. A.2: an `(ℱ t)`-Brownian motion of variance `1` is a Brownian motion in Mathlib's sense
(`IsBrownianReal`, through `HasIndepIncrements.isPreBrownianReal_of_hasLaw`) which is
`ℱ`-adapted with increments independent of `ℱ`, and conversely. -/
theorem isBrownian_one_iff : IsBrownian ℱ P 1 B ↔ IsBrownianReal B P ∧ StronglyAdapted ℱ B ∧
    ∀ s t, s ≤ t → Indep (MeasurableSpace.comap (B t - B s) inferInstance) (ℱ s) P := by
  refine ⟨fun hB => ⟨⟨hB.hasIndepIncrements.isPreBrownianReal_of_hasLaw fun t => ?_,
    hB.continuous⟩, hB.adapted, hB.indep⟩, fun ⟨hB, hadapt, hindep⟩ =>
    ⟨hB.eval_zero_ae_eq_zero, hB.cont, hadapt, hindep, fun s t hst => ?_⟩⟩
  · simpa using hB.hasLaw_eval t
  · convert hB.hasLaw_sub t s using 2
    ext
    simp [NNReal.coe_sub hst, Real.dist_eq, abs_of_nonneg (sub_nonneg.2 (NNReal.coe_le_coe.2 hst))]

namespace IsCylindricalWiener

variable {W : ℝ≥0 → Lp ℝ 2 ρ → Ω → ℝ}

/-- App. A.2: `W(t)[0] = 0` almost surely, by linearity. -/
theorem zero_ae (hW : IsCylindricalWiener ρ ℱ P W) (t : ℝ≥0) : W t 0 =ᵐ[P] 0 :=
  (hW.linear t 1 0 0).mono fun ω hω => by simpa using hω

/-- App. A.2: `𝔼[W(r)[h] W(r)[g]] = r ⟪h, g⟫`, polarizing `Var(W(r)[h]) = r ‖h‖²`. -/
theorem integral_mul_same_time (hW : IsCylindricalWiener ρ ℱ P W) (r : ℝ≥0)
    (h g : Lp ℝ 2 ρ) : ∫ ω, W r h ω * W r g ω ∂P = r * ⟪h, g⟫_ℝ := by
  have hh := (hW.brownian h).memLp_two r
  have hg := (hW.brownian g).memLp_two r
  have hsum : ∫ ω, W r (h + g) ω ^ 2 ∂P =
      ∫ ω, W r h ω ^ 2 + 2 * (W r h ω * W r g ω) + W r g ω ^ 2 ∂P :=
    integral_congr_ae <| (hW.linear r 1 h g).mono fun ω hω => by
      simp only [one_smul, Pi.add_apply] at hω
      simp only [hω]
      ring
  have i1 : Integrable (fun ω => W r h ω ^ 2) P := hh.integrable_sq
  have i2 : Integrable (fun ω => 2 * (W r h ω * W r g ω)) P := (hh.integrable_mul hg).const_mul 2
  rw [integral_add (i1.fun_add i2) hg.integrable_sq, integral_add i1 i2, integral_const_mul,
    (hW.brownian (h + g)).integral_sq, (hW.brownian h).integral_sq,
    (hW.brownian g).integral_sq] at hsum
  push_cast at hsum
  linear_combination (-1 / 2 : ℝ) * hsum + ((r : ℝ) / 2) * norm_add_sq_real h g

/-- App. A.2, the source's "Equivalently": `𝔼[W(t)[h] W(s)[g]] = (s ∧ t) ⟪h, g⟫`. -/
theorem integral_mul (hW : IsCylindricalWiener ρ ℱ P W) (s t : ℝ≥0) (h g : Lp ℝ 2 ρ) :
    ∫ ω, W t h ω * W s g ω ∂P = min s t * ⟪h, g⟫_ℝ := by
  wlog hst : s ≤ t generalizing s t h g
  · rw [min_comm, real_inner_comm, ← this t s g h (le_of_not_ge hst)]
    simp_rw [mul_comm (W t h _)]
  have hL := fun k r => (hW.brownian k).memLp_two r
  calc ∫ ω, W t h ω * W s g ω ∂P
      = ∫ ω, (W t h ω - W s h ω) * W s g ω + W s h ω * W s g ω ∂P :=
        integral_congr_ae (ae_of_all _ fun _ => by ring)
    _ = (∫ ω, (W t h ω - W s h ω) * W s g ω ∂P) + ∫ ω, W s h ω * W s g ω ∂P :=
        integral_add (((hL h t).sub (hL h s)).integrable_mul (hL g s))
          ((hL h s).integrable_mul (hL g s))
    _ = min s t * ⟪h, g⟫_ℝ := by
        rw [(hW.brownian h).integral_sub_mul hst ((hW.brownian g).adapted s), zero_add,
          hW.integral_mul_same_time, min_eq_left hst]

end IsCylindricalWiener

/-- The hypotheses of every result above hold for the zero process and the zero noise on `L²(0)`
under the Dirac mass on `Unit`, with the constant filtration. -/
example (h g : Lp ℝ 2 (0 : Measure ℝ)) :=
  have hB := isBrownian_zero (P := Measure.dirac ()) (Filtration.const ℝ≥0 _ le_rfl)
  have hW := isCylindricalWiener_zero (Θ := ℝ) (P := Measure.dirac ())
    (Filtration.const ℝ≥0 _ le_rfl)
  And.intro (And.intro (hB.measurable 1) (hB.hasLaw_eval 1)) <| And.intro (hB.memLp_two 1) <|
  And.intro (And.intro (hB.integrable 1) (hB.integral_eq_zero 1)) <| And.intro (hB.integral_sq 1) <|
  And.intro (hB.integral_sub_mul (zero_le_one' ℝ≥0) (stronglyMeasurable_const (b := 1))) <|
  And.intro (And.intro (hB.integral_mul 1 2) hB.martingale) <|
  And.intro (And.intro hB.hasIndepIncrements (hB.smul 2)) <| And.intro (hW.zero_ae 1) <|
  And.intro (hW.integral_mul_same_time 1 h g) (hW.integral_mul 2 1 h g)

end Transformer.Wiener

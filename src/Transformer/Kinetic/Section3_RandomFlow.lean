/-
# Observables of a random flow

ArXiv:2605.09213v1, §3, `lem:correl`: the angles at time `t` are the flow map of the
initial angles (`eq_flowMap`), and independent initial angles have the product of
their laws as joint law (`map_initial_eq_pi`). The covariance and the third cumulant
of observables of the random flow are therefore those of the flow map under the
product law (`covariance_randomFlow`, `thirdCumulant_randomFlow`).
-/

import Transformer.Kinetic.Fluctuations
import Transformer.Kinetic.Section3_Observables

open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}

/-- The third cumulant of an image law is that of the composed variables. -/
theorem thirdCumulant_map {Ω' : Type*} [MeasurableSpace Ω'] {Z : Ω → Ω'}
    (hZ : AEMeasurable Z P) {F G H : Ω' → ℝ} (hF : Measurable F) (hG : Measurable G)
    (hH : Measurable H) :
    thirdCumulant (P.map Z) F G H = thirdCumulant P (F ∘ Z) (G ∘ Z) (H ∘ Z) := by
  unfold thirdCumulant
  rw [integral_map hZ hF.aestronglyMeasurable, integral_map hZ hG.aestronglyMeasurable,
    integral_map hZ hH.aestronglyMeasurable]
  exact integral_map hZ (((hF.sub_const _).fun_mul (hG.sub_const _)).fun_mul
    (hH.sub_const _)).aestronglyMeasurable

omit [MeasurableSpace Ω] in
/-- An observable of a flow is that of the flow map at its initial angles. -/
theorem obs_eq_comp (lam β : ℝ) (N : ℕ) {ϑ : Ω → ℝ → Idx N → ℝ}
    (hflow : ∀ ω, IsGPTFlow lam β N (ϑ ω)) (φ : ℝ → ℝ) (t : ℝ) (i : Idx N) :
    (fun ω => φ (ϑ ω t i)) = (fun x => φ (flowMap lam β N t x i)) ∘ fun ω => ϑ ω 0 :=
  funext fun ω => by rw [Function.comp_apply, ← eq_flowMap lam β N (hflow ω) t]

/-- Independent initial angles have the product law. -/
theorem map_initial_eq_pi [IsProbabilityMeasure P] {N : ℕ} {ϑ : Ω → ℝ → Idx N → ℝ}
    (hind : iIndepFun (fun (j : Idx N) (ω : Ω) => ϑ ω 0 j) P)
    (hmeas : ∀ j : Idx N, Measurable fun ω => ϑ ω 0 j) :
    P.map (fun ω => ϑ ω 0) = Measure.pi fun j => P.map fun ω => ϑ ω 0 j :=
  (iIndepFun_iff_map_fun_eq_pi_map fun j => (hmeas j).aemeasurable).1 hind

/-- **The covariance of a random flow** is that of the flow map under the product law
of the initial angles. -/
theorem covariance_randomFlow [IsProbabilityMeasure P] (lam β : ℝ) {N : ℕ}
    {ϑ : Ω → ℝ → Idx N → ℝ} (hflow : ∀ ω, IsGPTFlow lam β N (ϑ ω))
    (hind : iIndepFun (fun (j : Idx N) (ω : Ω) => ϑ ω 0 j) P)
    (hmeas : ∀ j : Idx N, Measurable fun ω => ϑ ω 0 j) {φ dφ : ℝ → ℝ}
    (hφ : ∀ x, HasDerivAt φ (dφ x) x) {t t' : ℝ} (ht : 0 ≤ t) (ht' : 0 ≤ t') (i j : Idx N) :
    covariance (fun ω => φ (ϑ ω t i)) (fun ω => φ (ϑ ω t' j)) P =
      cov[fun x => φ (flowMap lam β N t x i), fun x => φ (flowMap lam β N t' x j);
        Measure.pi fun l => P.map fun ω => ϑ ω 0 l] := by
  rw [← map_initial_eq_pi hind hmeas,
    covariance_map (measurable_obs lam β N hφ ht i).aestronglyMeasurable
      (measurable_obs lam β N hφ ht' j).aestronglyMeasurable
      (Measurable.of_eval hmeas).aemeasurable,
    obs_eq_comp lam β N hflow φ t i, obs_eq_comp lam β N hflow φ t' j]

/-- **The third cumulant of a random flow** is that of the flow map under the product
law of the initial angles. -/
theorem thirdCumulant_randomFlow [IsProbabilityMeasure P] (lam β : ℝ) {N : ℕ}
    {ϑ : Ω → ℝ → Idx N → ℝ} (hflow : ∀ ω, IsGPTFlow lam β N (ϑ ω))
    (hind : iIndepFun (fun (j : Idx N) (ω : Ω) => ϑ ω 0 j) P)
    (hmeas : ∀ j : Idx N, Measurable fun ω => ϑ ω 0 j) {φ dφ : ℝ → ℝ}
    (hφ : ∀ x, HasDerivAt φ (dφ x) x) {t t' t'' : ℝ} (ht : 0 ≤ t) (ht' : 0 ≤ t')
    (ht'' : 0 ≤ t'') (i j k : Idx N) :
    thirdCumulant P (fun ω => φ (ϑ ω t i)) (fun ω => φ (ϑ ω t' j))
        (fun ω => φ (ϑ ω t'' k)) =
      thirdCumulant (Measure.pi fun l => P.map fun ω => ϑ ω 0 l)
        (fun x => φ (flowMap lam β N t x i)) (fun x => φ (flowMap lam β N t' x j))
        (fun x => φ (flowMap lam β N t'' x k)) := by
  rw [← map_initial_eq_pi hind hmeas,
    thirdCumulant_map (Measurable.of_eval hmeas).aemeasurable
      (measurable_obs lam β N hφ ht i) (measurable_obs lam β N hφ ht' j)
      (measurable_obs lam β N hφ ht'' k),
    obs_eq_comp lam β N hflow φ t i, obs_eq_comp lam β N hflow φ t' j,
    obs_eq_comp lam β N hflow φ t'' k]

/-- The hypotheses of `thirdCumulant_map` hold for the identity observables of a constant. -/
example := thirdCumulant_map (P := Measure.dirac ()) (Z := fun _ : Unit => (0 : ℝ))
  measurable_const.aemeasurable (F := id) (G := id) (H := id) measurable_id measurable_id
  measurable_id

/-- The hypotheses of `obs_eq_comp` hold for three tokens resting at `0`. -/
example (lam β : ℝ) := obs_eq_comp (Ω := Unit) lam β 3 (ϑ := fun _ _ _ => 0)
  (fun _ => isGPTFlow_constant_zero lam β 3) Real.sin 0 0

/-- The hypotheses of `map_initial_eq_pi` hold on the one-point probability space. -/
example := map_initial_eq_pi (P := Measure.dirac ()) (N := 3)
  (ϑ := fun (_ : Unit) _ _ => (0 : ℝ)) (Homogenized.iIndepFun_of_unit _)
  (fun _ => measurable_const)

/-- The hypotheses of `covariance_randomFlow` hold for three tokens resting at `0` on the
one-point probability space, and the test `sin`. -/
example (lam β : ℝ) := covariance_randomFlow (P := Measure.dirac ()) lam β (N := 3)
  (ϑ := fun (_ : Unit) _ _ => (0 : ℝ)) (fun _ => isGPTFlow_constant_zero lam β 3)
  (Homogenized.iIndepFun_of_unit _) (fun _ => measurable_const) Real.hasDerivAt_sin le_rfl
  le_rfl 1 0

/-- The hypotheses of `thirdCumulant_randomFlow` hold for three tokens resting at `0` on
the one-point probability space, and the test `sin`. -/
example (lam β : ℝ) := thirdCumulant_randomFlow (P := Measure.dirac ()) lam β (N := 3)
  (ϑ := fun (_ : Unit) _ _ => (0 : ℝ)) (fun _ => isGPTFlow_constant_zero lam β 3)
  (Homogenized.iIndepFun_of_unit _) (fun _ => measurable_const) Real.hasDerivAt_sin le_rfl
  le_rfl le_rfl 2 1 0

end Kinetic
end Transformer

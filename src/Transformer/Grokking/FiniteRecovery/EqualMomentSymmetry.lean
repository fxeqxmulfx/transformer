module

public import Transformer.Grokking.FiniteRecovery.EqualMomentRelabeling
public import Mathlib.MeasureTheory.Measure.Real

/-! If swapping two answer tokens fixes every training batch and a held-out
prompt, an ordinary equal-moment trajectory cannot prefer either token with
probability greater than one half at a fixed step or on its entire tail. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory Set Filter
open scoped NNReal Topology

variable {A : Type*} [Fintype A] {H d L T : ℕ}
local notation "Draw" => (ComplexParameter A H d L → ℂ)

/-- Argand's score for a single hidden prompt after ordinary training. -/
noncomputable def equalMomentScore (r : ℕ) (decay : ℝ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (x : Word A T) (i : Fin T)
    (z : Draw) (t : ℕ) (a : A) : ℝ :=
  (Model.fromWeights (equalMomentRun r decay batch rate z t).weights).logits r x i a

theorem continuous_equalMomentScore (r : ℕ) (decay : ℝ) (rate : ℕ → ℝ)
    (batch : ℕ → List (LabeledPosition A)) (x : Word A T) (i : Fin T) (t : ℕ) (a : A) :
    Continuous (fun z : Draw => equalMomentScore r decay rate batch x i z t a) :=
  (continuous_logits_fromWeights r x i a).comp
    ((EqualMoment.continuous_run decay rate (fun k => batchGradient r (batch k)) (9/10)
      (by norm_num : (0:ℝ) < 1/100000000) (fun k => continuous_batchGradient r (batch k)) t).1.comp
      (continuous_sgdInitialWeights true))

/-- Two equally probable disjoint events each have mass at most one half. -/
theorem mass_le_half_of_relabel {U V : Set Draw} (variance : ℝ≥0) (e : Equiv.Perm A)
    (hV : MeasurableSet V) (hd : Disjoint U V) (ht : relabelDraw e ⁻¹' U = V) :
    (independentGaussianDraws (fun _ => variance)).real U ≤ 1/2 := by
  let μ := independentGaussianDraws (fun _ : ComplexParameter A H d L => variance)
  have he : μ U = μ V := (gaussian_relabel_preimage e (fun _ => variance) U).trans (congrArg μ ht)
  have hu : μ.real (U ∪ V) = μ.real U + μ.real U := by
    rw [measureReal_union hd hV,show μ.real V = μ.real U from congrArg ENNReal.toReal he.symm]
  have := measureReal_le_one (μ := μ) (s := U ∪ V)
  change μ.real U ≤ 1/2
  linarith

/-- Uniform native initialization is exchangeable between two token rows.
All targets, prompts and multiplicities in the batch sequence are preserved
by the specified swap. The claims apply to arbitrary schedules and decay. -/
theorem equalMoment_swap_bound (e : Equiv.Perm A) (a b : A) (ha : e a = b) (hb : e b = a)
    (r : ℕ) (decay : ℝ) (rate : ℕ → ℝ) (batch : ℕ → List (LabeledPosition A))
    (hbatch : ∀ t, (batch t).map (relabelPosition e) = batch t) (x : Word A T)
    (hx : e ∘ x = x) (i : Fin T) (variance : ℝ≥0) :
    let score := equalMomentScore (H := H) (d := d) (L := L) r decay rate batch x i
    let μ := independentGaussianDraws (fun _ : ComplexParameter A H d L => variance)
    (∀ t, μ.real {z | score z t b < score z t a} ≤ 1/2) ∧
      μ.real {z | ∀ᶠ t in atTop, score z t b < score z t a} ≤ 1/2 := by
  intro score μ
  have hs (z : Draw) (t : ℕ) (c : A) : score (relabelDraw e z) t (e c) = score z t c := by
    simpa only [funext hbatch,hx,score,equalMomentScore] using equalMoment_logits_relabel e r t decay rate batch z x i c
  have hswap (z : Draw) (t : ℕ) : score (relabelDraw e z) t b < score (relabelDraw e z) t a ↔
      score z t a < score z t b := by
    rw [← ha,hs,ha,← hb,hs]
  have hopen (t : ℕ) : IsOpen {z | score z t a < score z t b} :=
    isOpen_lt (continuous_equalMomentScore _ _ _ _ _ _ _ _) (continuous_equalMomentScore _ _ _ _ _ _ _ _)
  constructor
  · intro t
    apply mass_le_half_of_relabel variance e (hopen t).measurableSet
    · apply Set.disjoint_left.mpr
      intro z h₁ h₂
      change score z t b < score z t a at h₁
      change score z t a < score z t b at h₂
      exact lt_asymm h₁ h₂
    · exact Set.ext fun z => hswap z t
  · let V := {z : Draw | ∀ᶠ t in atTop, score z t a < score z t b}
    have hv : V = ⋃ n : ℕ, ⋂ k : ℕ, {z | score z (n+k) a < score z (n+k) b} := by
      ext z
      simp only [V,mem_ofPred_eq,eventually_atTop,mem_iUnion,mem_iInter]
      exact ⟨fun ⟨n,h⟩ => ⟨n,fun k => h (n+k) (by omega)⟩,fun ⟨n,h⟩ =>
        ⟨n,fun k hk => by simpa [Nat.add_sub_of_le hk] using h (k-n)⟩⟩
    apply mass_le_half_of_relabel (V := V) variance e
    · rw [hv]; exact MeasurableSet.iUnion fun n => MeasurableSet.iInter fun k => (hopen _).measurableSet
    · apply Set.disjoint_left.mpr
      intro z hz h
      obtain ⟨_,h₁,h₂⟩ := (hz.and h).exists
      exact lt_asymm h₁ h₂
    · exact Set.ext fun z => eventually_congr (Eventually.of_forall (hswap z))

end Transformer.Grokking.FiniteRecovery.Argand

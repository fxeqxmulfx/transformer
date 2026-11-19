module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms.Joint
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BoxHead

/-!
# Complete complex realization of joint latent decoder coefficients

Source: `review/convex/moments.md`, CM53. Every input-program block retains the
full complex code and raw head. Joint masses become actual conjugate-head
coefficients linearly; all real and imaginary coordinates are charged.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {J X S C : Type*} [Fintype J] [Fintype S] [DecidableEq S]

/-- Full realified source rows in every fixed input-program block, CM53.
Repeated latent states do not assert a flat empirical query-feature matrix. -/
def features (U : Matrix S S ℂ) (program : J → X → S) : J × (S × Bool) → X → ℝ :=
  fun k x => complexFeatures U k.2 (program k.1 x)

/-- Actual complete conjugate heads from the joint raw masses, CM53.
There is no product of independently learned program and decoder parameters. -/
noncomputable def head (U : Matrix S S ℂ) (margin : ℝ) (q : Parameters J S C) :
    J × (S × Bool) → C → ℝ :=
  fun k c => complexCoefficients
    (tableHead U (fun s d => margin * q.2 k.1 s d)) k.2 c

/-- Expanding the actual complete complex products gives a linear logit
map of the joint tensor, CM53. The raw vocabulary may differ from latent states. -/
theorem logits (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (margin : ℝ) (q : Parameters J S C) (x : X) (c : C) :
    headLogits (features U program) (head U margin q) x c =
      margin * ∑ j, q.2 j (program j x) c := by
  unfold headLogits
  rw [Fintype.sum_prod_type]
  change (∑ j, headLogits (complexFeatures U)
    (complexCoefficients (tableHead U (fun s d => margin * q.2 j s d)))
      (program j x) c) = _
  calc
    _ = ∑ j, margin * q.2 j (program j x) c := by
      apply Finset.sum_congr rfl
      intro j _
      exact congrFun (tableHead_logits U hU (fun s d => margin * q.2 j s d)
        (program j x)) c
    _ = _ := by rw [Finset.mul_sum]

omit [Fintype J] in
/-- Flat-source entry bounds hold in every program block, CM53. -/
theorem features_bound (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (x : X) (k : J × (S × Bool)) :
    |features U program k x| ≤ 1 := by
  exact complexFeatures_bound U hU k.2 (program k.1 x)

/-- Ordinary linear combinations of joint coordinates produce the same
linear combinations of actual full logits, CM53. -/
theorem logits_linear (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (margin : ℝ) (a b : Parameters J S C) (t u : ℝ) (x : X) :
    headLogits (features U program) (head U margin (t • a + u • b)) x =
      t • headLogits (features U program) (head U margin a) x +
        u • headLogits (features U program) (head U margin b) x := by
  funext c
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, logits U hU]
  change margin * (∑ j, (t * a.2 j (program j x) c + u * b.2 j (program j x) c)) = _
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  ring

variable [Fintype C] [DecidableEq C]

omit [DecidableEq S] [DecidableEq C] in
/-- The complete realified head stores2|J||S||C| coordinates, CM53.
An aggregate norm bound independent of |J| does not remove this storage cost. -/
theorem coefficient_count : Fintype.card ((J × (S × Bool)) × C) =
    2 * Fintype.card J * Fintype.card S * Fintype.card C := by
  simp only [Fintype.card_prod, Fintype.card_bool]
  ring

/-- The entire multi-program head has budget2|margin||S|, CM53. Every
block is charged; shared masses sum to1, so this norm bound has no hidden |J|.
Storage and the number of actual products still grow with |J|. -/
theorem head_budget (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (margin : ℝ)
    {q : Parameters J S C} (hq : q ∈ domain program order samples) :
    head U margin q ∈ headBudget (2 * |margin| * Fintype.card S) := by
  intro c
  rw [Fintype.sum_prod_type]
  calc
    _ ≤ ∑ j, 2 * (|margin| * q.1 j) * (Fintype.card S : ℝ) := by
      apply Finset.sum_le_sum
      intro j _
      apply tableHead_budget U hU (fun s d => margin * q.2 j s d)
        (mul_nonneg (abs_nonneg margin) (hq.1.1 j)) _ c
      intro s d
      rw [abs_mul, abs_of_nonneg (hq.2.1 j s d)]
      exact mul_le_mul_of_nonneg_left
        (coefficient_le_weight program order samples hq j s d) (abs_nonneg margin)
    _ = (2 * |margin| * Fintype.card S) * ∑ j, q.1 j := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j _
      ring
    _ = _ := by rw [hq.1.2, mul_one]

/-- The complete column budget and flat-source coordinate bound control
every raw logit, CM53. This is a capacity bound, not target identification. -/
theorem logit_bound (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (margin : ℝ)
    {q : Parameters J S C} (hq : q ∈ domain program order samples) (x : X) (c : C) :
    |headLogits (features U program) (head U margin q) x c| ≤
      2 * |margin| * Fintype.card S := by
  have h := headLogits_bound (features U program)
    (head_budget U hU program order samples margin hq) (by norm_num : (0 : ℝ) ≤ 1)
    x (features_bound U hU program x) c
  simpa only [mul_one] using h

/-- Ordinary full-vocabulary cross-entropy is convex in all joint decoder
and program coefficients, CM53. This does not train bilinear factors. -/
theorem ce_convex (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (margin : ℝ)
    (x : X) (y : C) : ConvexOn ℝ (domain program order samples)
      (fun q => crossEntropy (headLogits (features U program) (head U margin q) x) y) := by
  refine ⟨domain_convex program order samples, ?_⟩
  intro a _ b _ t u ht hu htu
  change crossEntropy (headLogits (features U program) (head U margin (t • a + u • b)) x) y ≤ _
  rw [logits_linear U hU]
  exact (ConvexRecall.convex_crossEntropy y).2
    (Set.mem_univ _) (Set.mem_univ _) ht hu htu

example := logits (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram 24 examplePoint (true, true) true

example := features_bound (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram (true, true) (true, (false, true))

-- The feature-bound hypotheses also admit the genuine complex Fourier code,
-- including an imaginary source coordinate used by a nonconstant program.
example := features_bound fourier4 fourier4_unitary SharedPrograms.witnessProgram
  (0, 3) (true, (1, true))

example := logits_linear (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram 24 examplePoint 0 (1 / 2) (1 / 2) (true, true)

example := head_budget (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram [false, true] exampleSamples 24 examplePoint_mem

example := coefficient_count (J := Bool) (S := Fin 4) (C := Bool)

example := logit_bound (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram [false, true] exampleSamples 24 examplePoint_mem (true, true) false

example := ce_convex (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram [false, true] exampleSamples 24 (true, true) true

end Transformer.Grokking.FlatSpectrum.ConvexDepth.LatentPrograms

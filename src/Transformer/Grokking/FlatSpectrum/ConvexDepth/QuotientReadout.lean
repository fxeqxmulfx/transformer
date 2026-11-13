module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.QuotientHead

/-!
# Preserve the declared pooled readout with the actual complex logits

Source: `review/flat_spectrum.md`, CM31. Original pooled metrics sum softmax
mass over raw classes sharing a readout. The common positive normalizer
does not affect their ordering. A sufficiently large assigned-class margin
keeps its readout group strictly ahead despite unequal group sizes. This
is an exact-real statement, separate from numerical replay and code discovery.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators

variable {C D : Type*} [Fintype C] [DecidableEq C] [DecidableEq D]

/-- Unnormalized softmax mass of an original declared readout group.
Using the same positive normalizer preserves every group comparison, CM31. -/
noncomputable def quotientScore (quotient : C → D) (logits : C → ℝ) (d : D) : ℝ :=
  ∑ c ∈ Finset.univ.filter (fun c => quotient c = d), Real.exp (logits c)

/-- The assigned raw class alone outweighs every competing readout group
when exp(margin) exceeds the vocabulary size, CM31. -/
theorem quotientScore_margin_strict (quotient : C → D) (label : C) (a : ℝ)
    (ha : (Fintype.card C : ℝ) < Real.exp a) {d : D} (hd : d ≠ quotient label) :
    quotientScore quotient (fun c => if c = label then a else 0) d <
      quotientScore quotient (fun c => if c = label then a else 0) (quotient label) := by
  have hsmall : quotientScore quotient (fun c => if c = label then a else 0) d ≤
      (Fintype.card C : ℝ) := by
    calc
      _ = ((Finset.univ.filter (fun c => quotient c = d)).card : ℝ) := by
        rw [quotientScore]
        calc
          _ = ∑ c ∈ Finset.univ.filter (fun c => quotient c = d), (1 : ℝ) := by
            apply Finset.sum_congr rfl
            intro c hc
            have hcq : quotient c = d := (Finset.mem_filter.mp hc).2
            have hne : c ≠ label := by
              intro h
              exact hd (h ▸ hcq.symm)
            simp [hne]
          _ = _ := by simp
      _ ≤ _ := by
        have h := Finset.card_filter_le (Finset.univ : Finset C) (fun c => quotient c = d)
        exact_mod_cast h
  have hbig : Real.exp a ≤
      quotientScore quotient (fun c => if c = label then a else 0) (quotient label) := by
    have h := Finset.single_le_sum
      (s := Finset.univ.filter (fun c => quotient c = quotient label))
      (f := fun c => Real.exp (if c = label then a else 0))
      (fun c _ => (Real.exp_pos _).le) (by simp : label ∈ _)
    simpa [quotientScore] using h
  exact hsmall.trans_lt (ha.trans_le hbig)

omit [DecidableEq C] [DecidableEq D] in
/-- Margin24 exceeds the grouping requirement for the native vocabulary
cap1024, without a decimal approximation to exp24, CM31. -/
theorem native_margin_exceeds_vocabulary (hcard : Fintype.card C ≤ 1024) :
    (Fintype.card C : ℝ) < Real.exp 24 := by
  have h : (2 : ℝ) ≤ Real.exp 1 := by
    have h := Real.add_one_le_exp 1
    norm_num at h
    exact h
  calc
    (Fintype.card C : ℝ) ≤ 1024 := by exact_mod_cast hcard
    _ < (2 : ℝ)^24 := by norm_num
    _ ≤ (Real.exp 1)^24 := pow_le_pow_left₀ (by norm_num) h 24
    _ = Real.exp 24 := by rw [← Real.exp_nat_mul]; norm_num

variable {X Z : Type*} [DecidableEq X] [Fintype Z] [Nonempty Z] [DecidableEq Z]

omit [DecidableEq D] in
/-- The explicit native margin has ordinary CE strictly below the shared
two-label floor. This compares actual losses, not only argmax labels, CM31. -/
theorem native_margin_ce_lt_log_two (label : C) (hcard : Fintype.card C ≤ 1024) :
    Landscape.Neural.crossEntropy (fun c => if c = label then 24 else 0) label < Real.log 2 := by
  rw [fixed_margin_ce]
  have hn : (1 : ℝ) ≤ Fintype.card C := by
    exact_mod_cast (Fintype.card_pos_iff.mpr (⟨label⟩ : Nonempty C))
  have hc : 0 ≤ (Fintype.card C : ℝ) - 1 := by linarith
  have hp : Real.exp 24 * Real.exp (-24) = 1 := by
    rw [← Real.exp_add]
    norm_num
  have h := mul_lt_mul_of_pos_right (native_margin_exceeds_vocabulary hcard)
    (Real.exp_pos (-24))
  rw [hp] at h
  apply Real.log_lt_log (by positivity)
  nlinarith [Real.exp_pos (-24)]

omit [DecidableEq D] in
/-- Every observed raw-label loss of the actual refined head is below log2,
including pairs for which the unrefined head has mean loss at least log2. -/
theorem refinedComplex_ce_small_seen (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (hcard : Fintype.card C ≤ 1024)
    {x : X} (hx : x ∈ observed) :
    Landscape.Neural.crossEntropy
      (refinedComplexLogits observed latent labels representative 24 x) (labels ⟨x, hx⟩) <
        Real.log 2 := by
  rw [refinedComplexLogits_eq, refinedPrediction_seen _ _ _ _ hx]
  exact native_margin_ce_lt_log_two _ hcard

/-- The actual initialized complex head preserves the compatible declared
quotient even when the metric pools multiple raw classes, CM31. -/
theorem refinedComplex_pooled_strict (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (quotient : C → D) (code : Z → D)
    (hlabels : ∀ x : ↥observed, quotient (labels x) = code (latent x))
    (hrepresentative : ∀ z, quotient (representative z) = code z)
    (hcard : Fintype.card C ≤ 1024) (x : X) {d : D} (hd : d ≠ code (latent x)) :
    quotientScore quotient (refinedComplexLogits observed latent labels representative 24 x) d <
      quotientScore quotient (refinedComplexLogits observed latent labels representative 24 x)
        (code (latent x)) := by
  have hq := refinedPrediction_quotient observed latent labels representative quotient code
    hlabels hrepresentative x
  rw [refinedComplexLogits_eq]
  rw [← hq] at hd ⊢
  exact quotientScore_margin_strict _ _ _ (native_margin_exceeds_vocabulary hcard) hd

/-- Conditional decoded correctness transfers through the pooled metric,
not only through the raw argmax. The identifying premise remains explicit. -/
theorem refinedComplex_pooled_decoded (observed : Finset X) (latent : X → Z)
    (labels : ↥observed → C) (representative : Z → C) (quotient : C → D) (code : Z → D)
    (hlabels : ∀ x : ↥observed, quotient (labels x) = code (latent x))
    (hrepresentative : ∀ z, quotient (representative z) = code z)
    (hcard : Fintype.card C ≤ 1024) (target : X → C) (x : X)
    (hidentified : code (latent x) = quotient (target x)) {d : D}
    (hd : d ≠ quotient (target x)) :
    quotientScore quotient (refinedComplexLogits observed latent labels representative 24 x) d <
      quotientScore quotient (refinedComplexLogits observed latent labels representative 24 x)
        (quotient (target x)) := by
  rw [← hidentified] at hd ⊢
  exact refinedComplex_pooled_strict observed latent labels representative quotient code
    hlabels hrepresentative hcard x hd

example := native_margin_exceeds_vocabulary (C := Bool × Bool) (by decide)

example := native_margin_ce_lt_log_two (false, true) (by decide)

example := refinedComplex_ce_small_seen ({false} : Finset Bool) id
  (fun x => (x.val, true)) (fun x => (x, false)) (by decide) (x := false) (by simp)

example := quotientScore_margin_strict (fun c : Bool × Bool => c.1) (false, true) 24
  (native_margin_exceeds_vocabulary (by decide)) (d := true) (by decide)

example := refinedComplex_pooled_strict ({false} : Finset Bool) id
  (fun x => (x.val, true)) (fun x => (x, false)) Prod.fst id
  (fun _ => rfl) (fun _ => rfl) (by decide) true (d := false) (by decide)

example := refinedComplex_pooled_decoded ({false} : Finset Bool) id
  (fun x => (x.val, true)) (fun x => (x, false)) Prod.fst id
  (fun _ => rfl) (fun _ => rfl) (by decide) (fun x => (x, true)) true rfl
  (d := false) (by decide)

end Transformer.Grokking.FlatSpectrum.ConvexDepth

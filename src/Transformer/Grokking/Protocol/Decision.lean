module

public import Mathlib.Algebra.Order.Archimedean.Real.Basic
public import Mathlib.Data.Fintype.Lattice
public import Mathlib.Data.Fintype.Order
public import Mathlib.Order.Filter.AtTopBot.Basic
public import Mathlib.Topology.Order.Basic
public import Mathlib.Tactic

/-!
# Prediction with a tie rule, competitor margins and accuracy

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*, "loss
and accuracy"; openai/grok at 3d64b1d8, `grok/training.py`, `_accuracy`, which takes
`torch.max(…).indices`; `experiments/grok_division/PROTOCOL.md`, *What a run records*.

A classifier gives every class a logit. Its prediction is the first, in the order of the
classes, of the classes with the largest logit (`predict`, `predict_eq_iff`): a deterministic
tie rule, over all classes. The competitor margin of a class is its logit less the largest
logit of another class (`margin`). A positive margin forces the prediction
(`predict_eq_of_margin_pos`), a prediction forces a nonnegative margin
(`margin_nonneg_of_predict_eq`), and at margin `0` the tie rule decides: both outcomes occur.

The accuracy on a finite set of inputs is the fraction whose label is predicted (`accuracy`).
It is `1` exactly when every label is predicted (`accuracy_eq_one_iff`), and one wrong
prediction keeps it at most `1 - 1/|H|` (`accuracy_le_of_ne`). So for a sequence of classifiers
the accuracy tends to `1` exactly when from some index on every label is predicted
(`tendsto_accuracy_one_iff`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology

variable {C X : Type*} [Fintype C] [LinearOrder C] [Nonempty C]

open scoped Classical in
/-- The classes with the largest logit. -/
noncomputable def maximizers (z : C → ℝ) : Finset C := Finset.univ.filter fun c => ∀ c', z c' ≤ z c

omit [LinearOrder C] in
lemma maximizers_nonempty (z : C → ℝ) : (maximizers z).Nonempty := by
  classical
  obtain ⟨c, hc⟩ := Finite.exists_max z
  exact ⟨c, Finset.mem_filter.2 ⟨Finset.mem_univ c, hc⟩⟩

/-- **The predicted class**: the first, in the order of the classes, of those with the largest
logit. -/
noncomputable def predict (z : C → ℝ) : C := (maximizers z).min' (maximizers_nonempty z)

lemma le_predict (z : C → ℝ) (c : C) : z c ≤ z (predict z) := by
  classical
  exact (Finset.mem_filter.1 (Finset.min'_mem _ (maximizers_nonempty z))).2 c

lemma predict_le {z : C → ℝ} {c : C} (h : ∀ c', z c' ≤ z c) : predict z ≤ c := by
  classical
  exact Finset.min'_le _ _ (Finset.mem_filter.2 ⟨Finset.mem_univ c, h⟩)

/-- **The tie rule**: `y` is predicted exactly when no logit is larger than its own and every
earlier class has a smaller one. -/
theorem predict_eq_iff {z : C → ℝ} {y : C} :
    predict z = y ↔ (∀ c, z c ≤ z y) ∧ ∀ c, c < y → z c < z y := by
  constructor
  · rintro rfl
    refine ⟨le_predict z, fun c hc => lt_of_le_of_ne (le_predict z c) fun e => ?_⟩
    exact absurd (predict_le fun c' => (le_predict z c').trans e.ge) (not_le.2 hc)
  · rintro ⟨h1, h2⟩
    refine le_antisymm (predict_le h1) (not_lt.1 fun hlt => ?_)
    exact absurd (le_predict z y) (not_le.2 (h2 _ hlt))

/-- **The competitor margin** of the class `y`: its logit less the largest logit of another
class. -/
noncomputable def margin (z : C → ℝ) (y : C) : ℝ := ⨅ c : {c // c ≠ y}, (z y - z c)

omit [LinearOrder C] [Nonempty C] in
lemma margin_le (z : C → ℝ) {y c : C} (h : c ≠ y) : margin z y ≤ z y - z c :=
  ciInf_le (Finite.bddBelow_range _) (⟨c, h⟩ : {c // c ≠ y})

/-- **A positive margin forces the prediction.** -/
theorem predict_eq_of_margin_pos {z : C → ℝ} {y : C} (h : 0 < margin z y) : predict z = y := by
  by_contra hne
  linarith [margin_le z hne, le_predict z y]

/-- **A predicted class has a nonnegative margin.** -/
theorem margin_nonneg_of_predict_eq {z : C → ℝ} {y : C} (h : predict z = y) :
    0 ≤ margin z y := by
  rcases isEmpty_or_nonempty {c // c ≠ y} with he | hn
  · rw [margin, Real.iInf_of_isEmpty]
  · exact le_ciInf fun c => sub_nonneg.2 (h ▸ le_predict z c)

open scoped Classical in
/-- **The accuracy** of the classifier `f` on the inputs `H` with labels `y`: the fraction
whose label is predicted. -/
noncomputable def accuracy (f : X → C → ℝ) (y : X → C) (H : Finset X) : ℝ :=
  ((H.filter fun x => predict (f x) = y x).card : ℝ) / H.card

variable {f : X → C → ℝ} {y : X → C} {H : Finset X}

lemma accuracy_nonneg : 0 ≤ accuracy f y H := by
  unfold accuracy
  positivity

lemma accuracy_le_one : accuracy f y H ≤ 1 :=
  div_le_one_of_le₀ (Nat.cast_le.2 (Finset.card_filter_le _ _)) (Nat.cast_nonneg _)

/-- **Accuracy `1` is every label predicted.** -/
theorem accuracy_eq_one_iff (hH : H.Nonempty) :
    accuracy f y H = 1 ↔ ∀ x ∈ H, predict (f x) = y x := by
  have hc : (H.card : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hH.card_pos.ne'
  rw [accuracy, div_eq_one_iff_eq hc, Nat.cast_inj, Finset.card_filter_eq_iff]

/-- **One wrong prediction costs one input**: the accuracy is then at most `1 - 1/|H|`. -/
theorem accuracy_le_of_ne {x : X} (hx : x ∈ H) (hne : predict (f x) ≠ y x) :
    accuracy f y H ≤ 1 - 1 / H.card := by
  classical
  have hH : (0 : ℝ) < H.card := Nat.cast_pos.2 (Finset.card_pos.2 ⟨x, hx⟩)
  have hlt : ((H.filter fun x => predict (f x) = y x).card : ℝ) + 1 ≤ H.card := by
    exact_mod_cast Finset.card_lt_card (Finset.filter_ssubset.2 ⟨x, hx, hne⟩)
  rw [accuracy, div_le_iff₀ hH, sub_mul, one_mul, one_div, inv_mul_cancel₀ hH.ne']
  linarith

/-- **The accuracy tends to `1` exactly when from some index on every label is predicted.** -/
theorem tendsto_accuracy_one_iff {f : ℕ → X → C → ℝ} (hH : H.Nonempty) :
    Tendsto (fun t => accuracy (f t) y H) atTop (𝓝 1) ↔
      ∀ᶠ t in atTop, ∀ x ∈ H, predict (f t x) = y x := by
  have hpos : (0 : ℝ) < 1 / H.card := one_div_pos.2 (Nat.cast_pos.2 hH.card_pos)
  constructor
  · intro h
    filter_upwards [h.eventually (lt_mem_nhds (sub_lt_self 1 hpos))] with t ht
    by_contra hne
    simp only [not_forall] at hne
    obtain ⟨x, hx, hxne⟩ := hne
    exact absurd (accuracy_le_of_ne hx hxne) (not_le.2 ht)
  · intro h
    exact tendsto_const_nhds.congr' (h.mono fun t ht => ((accuracy_eq_one_iff hH).2 ht).symm)

/-! The witnesses: two classes `false < true`. At equal logits the first class is predicted,
and both classes have margin `0`: the margin does not decide a tie. -/

lemma predict_const_bool : predict (fun _ : Bool => (0 : ℝ)) = false :=
  predict_eq_iff.2 ⟨fun _ => le_rfl, fun c hc => absurd hc (by cases c <;> simp)⟩

lemma predict_indicator_bool (x : Bool) :
    predict (fun c : Bool => if c = x then (1 : ℝ) else 0) = x :=
  predict_eq_iff.2 ⟨fun c => by by_cases h : c = x <;> simp [h], fun c hc => by simp [hc.ne]⟩

example : margin (fun _ : Bool => (0 : ℝ)) true = 0 ∧
    margin (fun _ : Bool => (0 : ℝ)) false = 0 := by
  constructor <;> simp [margin]

example := le_predict (fun _ : Bool => (0 : ℝ)) true

example := predict_le (z := fun _ : Bool => (0 : ℝ)) (c := true) fun _ => le_rfl

example := margin_le (fun _ : Bool => (0 : ℝ)) (y := true) (c := false) (by simp)

/-- The class `true` with logits `(0, 1)` has margin `1`, so it is predicted. -/
example : predict (fun c : Bool => if c = true then (1 : ℝ) else 0) = true := by
  have : Nonempty {c : Bool // c ≠ true} := ⟨⟨false, by simp⟩⟩
  refine predict_eq_of_margin_pos (lt_of_lt_of_le one_half_pos (le_ciInf fun c => ?_))
  obtain ⟨c, hc⟩ := c
  cases c
  · norm_num
  · exact absurd rfl hc

example : 0 ≤ margin (fun _ : Bool => (0 : ℝ)) false :=
  margin_nonneg_of_predict_eq predict_const_bool

example : accuracy (fun (_ _ : Bool) => (0 : ℝ)) id Finset.univ ≤ 1 :=
  accuracy_le_one

example : 0 ≤ accuracy (fun (_ _ : Bool) => (0 : ℝ)) id Finset.univ := accuracy_nonneg

/-- The classifier that marks its input has accuracy `1`. -/
example : accuracy (fun (x c : Bool) => if c = x then (1 : ℝ) else 0) id Finset.univ = 1 :=
  (accuracy_eq_one_iff Finset.univ_nonempty).2 fun x _ => predict_indicator_bool x

/-- The classifier with equal logits is wrong on the input `true` of two inputs: its accuracy is
at most `1/2`. -/
example : accuracy (fun (_ _ : Bool) => (0 : ℝ)) id Finset.univ ≤ 1 - 1 / 2 := by
  have h := accuracy_le_of_ne (f := fun (_ _ : Bool) => (0 : ℝ)) (y := id) (Finset.mem_univ true)
    (by rw [predict_const_bool]; simp)
  simpa using h

/-- A sequence of classifiers that marks its input from the index `3` on has accuracy tending
to `1`. -/
example : Tendsto (fun t : ℕ => accuracy
    (fun (x c : Bool) => if 3 ≤ t → c = x then (1 : ℝ) else 0) id Finset.univ) atTop (𝓝 1) :=
  (tendsto_accuracy_one_iff Finset.univ_nonempty).2 <| by
    filter_upwards [eventually_ge_atTop 3] with t ht x _
    simpa [ht] using predict_indicator_bool x

end Transformer.Grokking.Protocol

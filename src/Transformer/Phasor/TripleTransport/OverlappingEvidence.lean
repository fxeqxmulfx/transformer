module

public import Transformer.Phasor.TripleTransport

/-!
# Pooling overlapping observations in a learned prototype bank

Source: “Shared witnesses across contexts” in the triple-transport experiment.
Each context pair supplies observed pairs of learned output slots. Shared
evidence can identify a common relation before its possible images are
intersected across different peer contexts. Squared shifts give an example
in which several outputs can be consistent with a single context.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.OverlappingEvidence

variable {R Y M : Type*}

/-- Two finite observations share at least n distinct elements. -/
def witnesses (n : ℕ) (A B : Set Y) : Prop :=
  ∃ f : Fin n → Y, Function.Injective f ∧ ∀ i, f i ∈ A ∧ f i ∈ B

theorem witnesses_mono (n : ℕ) {A B C D : Set Y} (left : A ⊆ C) (right : B ⊆ D)
    (h : witnesses n A B) : witnesses n C D := by
  obtain ⟨f, inj, both⟩ := h
  exact ⟨f, inj, fun i => ⟨left (both i).1, right (both i).2⟩⟩

/-- If distinct underlying relations cannot overlap in n pairs, the partition
built from n shared observed pairs never identifies two different relations.
A relation can describe many possible images of one source value. -/
theorem common_mode (n : ℕ) (seen : R → Set Y) (relation : M → Set Y) (mode : R → M)
    (sound : ∀ r, seen r ⊆ relation (mode r))
    (separates : ∀ a b, witnesses n (relation a) (relation b) → a = b)
    {r s : R} (path : Relation.EqvGen (fun i j => witnesses n (seen i) (seen j)) r s) :
    mode r = mode s := by
  induction path with
  | rel i j h => exact separates _ _ (witnesses_mono n (sound i) (sound j) h)
  | refl => rfl
  | symm _ _ _ ih => exact ih.symm
  | trans _ _ _ _ _ a b => exact a.trans b

/-- Relating observations permits pooling their output pairs. Under the same
separation hypothesis every pooled pair remains in the destination relation. -/
theorem pooled_sound (n : ℕ) (seen : R → Set Y) (relation : M → Set Y) (mode : R → M)
    (sound : ∀ r, seen r ⊆ relation (mode r))
    (separates : ∀ a b, witnesses n (relation a) (relation b) → a = b)
    {r s : R} (p : Relation.EqvGen (fun i j => witnesses n (seen i) (seen j)) r s)
    {y : Y} (h : y ∈ seen r) : y ∈ relation (mode s) := by
  rw [← common_mode n seen relation mode sound separates p]
  exact sound r h

/-- Several ambiguous context reads can still distinguish the hidden target.
The condition is necessary and sufficient once every active set contains it. -/
theorem unique_intersection_iff (active : R → Prop) (allowed : R → Y → Prop) (target : Y)
    (covers : ∀ r, active r → allowed r target) :
    (∃! y, ∀ r, active r → allowed r y) ↔
    ∀ y, y ≠ target → ∃ r, active r ∧ ¬ allowed r y := by
  constructor
  · rintro ⟨winner, _, only⟩ y unequal
    by_contra! h
    exact unequal ((only y h).trans (only target covers).symm)
  · intro rules
    refine ⟨target, covers, fun y accepted => ?_⟩
    by_contra unequal
    obtain ⟨r, act, reject⟩ := rules y unequal
    exact reject (accepted r act)

private def fiber (m : Bool) : Set (Bool × Fin 3) := {p | p.1 = m}

private def observations (r : Bool × Fin 3) : Set (Bool × Fin 3) :=
  {p | p.1 = r.1 ∧ p.2 ≠ r.2}

private theorem fiber_separates (a b : Bool) (h : witnesses 1 (fiber a) (fiber b)) : a = b := by
  obtain ⟨f, _, overlap⟩ := h
  obtain ⟨left, right⟩ := overlap 0
  exact left.symm.trans right

private theorem example_path : Relation.EqvGen
    (fun i j => witnesses 1 (observations i) (observations j)) (true, 0) (true, 1) := by
  apply Relation.EqvGen.rel
  refine ⟨fun _ => (true, 2), fun i j _ => Subsingleton.elim i j, ?_⟩
  intro i
  change (true = true ∧ (2 : Fin 3) ≠ 0) ∧ (true = true ∧ (2 : Fin 3) ≠ 1)
  decide

example : (true, (1 : Fin 3)) ∈ fiber ((true, (1 : Fin 3)).1) :=
  pooled_sound 1 observations fiber Prod.fst (fun _ _ h => h.1) fiber_separates
    example_path (show (true, (1 : Fin 3)) ∈ observations (true, 0) from ⟨rfl, by decide⟩)

private def views (side : Bool) (v : Fin 3) : Prop := if side then v ≠ 0 else v ≠ 2

/-- Neither peer resolves this example on its own. Their intersection is a
singleton, witnessed with concrete finite, nonconstant sets. -/
example : (∃! v, ∀ r, views r v) ∧ (views true 2 ∧ views false 0) := by
  refine ⟨⟨1, ?_, ?_⟩, ?_, ?_⟩
  · intro r
    cases r <;> norm_num [views]
  · intro v all
    have left := all true
    have right := all false
    simp [views] at left right
    fin_cases v <;> simp_all
  · norm_num [views]
  · norm_num [views]

variable {K : Type*} [Field K]

/-- The unordered latent shift is visible through a curve in two squared
outputs. This model of the evidence is used only for the soundness proof. -/
def squareCurve (d : K) (p : K × K) : Prop := (p.2 - p.1 - d) ^ 2 = 4 * d * p.1

theorem observed_square_shift (x shift : K) :
    squareCurve (shift ^ 2) (x ^ 2, (x + shift) ^ 2) := by
  unfold squareCurve
  ring

private theorem intersection_sum {d e u v : K} (ne : d ≠ e)
    (hd : squareCurve d (u, v)) (he : squareCurve e (u, v)) :
    2 * (u + v) = d + e := by
  change (v - u - d) ^ 2 = 4 * d * u at hd
  change (v - u - e) ^ 2 = 4 * e * u at he
  have h : (d - e) * ((d + e) - 2 * (u + v)) = 0 := by linear_combination hd - he
  exact (sub_eq_zero.mp ((mul_eq_zero.mp h).resolve_left (sub_ne_zero.mpr ne))).symm

private theorem coords_of_sum_diff {u v a b : K} (two : (2 : K) ≠ 0)
    (sum : u + v = a + b) (diff : v - u = b - a) : (u, v) = (a, b) := by
  apply Prod.ext
  · apply mul_left_cancel₀ two
    change 2 * u = 2 * a
    linear_combination sum - diff
  · apply mul_left_cancel₀ two
    change 2 * v = 2 * b
    linear_combination sum + diff

/-- When the squared shifts differ, any two common ordered witnesses are
equal or reversed. Thus their intersection has at most two elements. -/
theorem common_square_pairs {d e : K} (two : (2 : K) ≠ 0) (ne : d ≠ e)
    (p q : K × K) (pd : squareCurve d p) (pe : squareCurve e p)
    (qd : squareCurve d q) (qe : squareCurve e q) : p = q ∨ p = q.swap := by
  obtain ⟨u, v⟩ := p
  obtain ⟨a, b⟩ := q
  have sums := mul_left_cancel₀ two
    ((intersection_sum ne pd pe).trans (intersection_sum ne qd qe).symm)
  change (v - u - d) ^ 2 = 4 * d * u at pd
  change (b - a - d) ^ 2 = 4 * d * a at qd
  have sq : (v - u) ^ 2 = (b - a) ^ 2 := by linear_combination pd - qd + 2 * d * sums
  rcases sq_eq_sq_iff_eq_or_eq_neg.mp sq with same | reverse
  · exact .inl (coords_of_sum_diff two sums same)
  · apply Or.inr
    apply coords_of_sum_diff two
    · simpa [add_comm] using sums
    · linear_combination reverse

/-- Three distinct shared pairs identify the squared shift even when individual
outputs, and individual ordered pairs, are ambiguous. -/
theorem three_square_witnesses (two : (2 : K) ≠ 0) {d e : K}
    (overlap : witnesses 3 {p | squareCurve d p} {p | squareCurve e p}) : d = e := by
  by_contra ne
  obtain ⟨f, inj, both⟩ := overlap
  have related (i : Fin 3) := common_square_pairs two ne (f i) (f 0)
    (both i).1 (both i).2 (both 0).1 (both 0).2
  have h₁ := (related 1).resolve_left (inj.ne (by decide : (1 : Fin 3) ≠ 0))
  have h₂ := (related 2).resolve_left (inj.ne (by decide : (2 : Fin 3) ≠ 0))
  exact (inj.ne (by decide : (1 : Fin 3) ≠ 2)) (h₁.trans h₂.symm)

example : squareCurve (1 : ℚ) (1, 4) ∧ squareCurve (9 : ℚ) (1, 4) ∧
    squareCurve (1 : ℚ) (4, 1) ∧ squareCurve (9 : ℚ) (4, 1) := by
  norm_num [squareCurve]

example : witnesses 3 {p | squareCurve (1 : ℚ) p} {p | squareCurve (1 : ℚ) p} := by
  let f : Fin 3 → ℚ × ℚ := ![(0, 1), (1, 4), (4, 9)]
  refine ⟨f, ?_, ?_⟩
  · intro a b eq
    fin_cases a <;> fin_cases b <;> norm_num [f] at *
  · intro i
    fin_cases i <;> norm_num [f, squareCurve, Set.mem_ofPred_eq]

end Transformer.Phasor.TripleTransport.OverlappingEvidence

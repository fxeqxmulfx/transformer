module

public import Transformer.Phasor.DisplacementGain

/-!
# Agreement of completed successors in a causal memory

Source: the successor-agreement trial in `experiments/phasor_value_transport/README.md`.
The hidden residual can amplify a retrieved value only while every completed
write to its key agrees. Each alphabet and every prompt length use the same rule.
-/

@[expose] public section

namespace Transformer.Phasor

variable {α β : Type*} {D d H F p : ℕ}

/-- Values observed after key `k` on the first `n` completed edges. -/
def completedSuccessors [DecidableEq α] (xs : ℕ → α) : ℕ → α → Finset α
  | 0, _ => ∅
  | n+1, k => if xs n = k then insert (xs (n+1)) (completedSuccessors xs n k)
    else completedSuccessors xs n k

/-- A value is in the query's history exactly when a completed directed
edge wrote that value to that key. -/
theorem mem_completedSuccessors [DecidableEq α] (xs : ℕ → α) (n : ℕ) (k v : α) :
    v ∈ completedSuccessors xs n k ↔ ∃ j < n, xs j = k ∧ xs (j+1) = v := by
  induction n with
  | zero => simp [completedSuccessors]
  | succ n ih =>
    by_cases h : xs n = k
    · rw [completedSuccessors, ite_eq_left h, Finset.mem_insert, ih]
      constructor
      · rintro (hv | ⟨j, hj, hk, hv⟩)
        · exact ⟨n, by omega, h, hv.symm⟩
        · exact ⟨j, by omega, hk, hv⟩
      · rintro ⟨j, hj, hk, hv⟩
        rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hj | rfl
        · exact Or.inr ⟨j, hj, hk, hv⟩
        · exact Or.inl hv.symm
    · rw [completedSuccessors, ite_eq_right h, ih]
      constructor
      · rintro ⟨j, hj, hk, hv⟩
        exact ⟨j, by omega, hk, hv⟩
      · rintro ⟨j, hj, hk, hv⟩
        have hne : j ≠ n := fun he => h (he ▸ hk)
        exact ⟨j, by omega, hk, hv⟩

/-- The ambiguity flag is permanent. A new write sets it exactly when some
earlier value disagrees with the new one or the key was already ambiguous. -/
theorem conflict_after_write [DecidableEq α] (s : Finset α) (v : α) :
    1 < (insert v s).card ↔ 1 < s.card ∨ ∃ old ∈ s, old ≠ v := by
  simp only [Finset.one_lt_card, Finset.mem_insert]
  constructor
  · rintro ⟨a, ha, b, hb, hab⟩
    rcases ha with rfl | ha <;> rcases hb with rfl | hb
    · exact (hab rfl).elim
    · exact Or.inr ⟨b, hb, Ne.symm hab⟩
    · exact Or.inr ⟨a, ha, hab⟩
    · exact Or.inl ⟨a, ha, b, hb, hab⟩
  · rintro (⟨a, ha, b, hb, hab⟩ | ⟨a, ha, hav⟩)
    · exact ⟨a, Or.inr ha, b, Or.inr hb, hab⟩
    · exact ⟨a, Or.inr ha, v, Or.inl rfl, hav⟩

/-- A missing last write means that the history is empty. An existing last
write always names an observed successor. -/
theorem latestSuccessor_history [DecidableEq α] (xs : ℕ → α) (n : ℕ) (k : α) :
    match latestSuccessor xs n k with
    | none => completedSuccessors xs n k = ∅
    | some v => v ∈ completedSuccessors xs n k := by
  induction n with
  | zero => rfl
  | succ n ih =>
    by_cases h : xs n = k
    · simp [latestSuccessor, completedSuccessors, h]
    · simpa only [latestSuccessor, completedSuccessors, ite_eq_right h] using ih

/-- One last-value register and one ambiguity bit implement the whole test.
When a key is written, a previous conflict persists and a disagreement with
its last value is sufficient to detect every new conflict. -/
theorem conflict_with_latest [DecidableEq α] (xs : ℕ → α) (n : ℕ) (k v : α) :
    1 < (insert v (completedSuccessors xs n k)).card ↔
      1 < (completedSuccessors xs n k).card ∨
        ∃ old, latestSuccessor xs n k = some old ∧ old ≠ v := by
  rw [conflict_after_write]
  have history := latestSuccessor_history xs n k
  cases hl : latestSuccessor xs n k with
  | none =>
    simp only [hl] at history
    simp [history]
  | some y =>
    simp only [hl] at history
    constructor
    · rintro (h | ⟨z, hz, hzv⟩)
      · exact Or.inl h
      · by_cases hy : y = v
        · exact Or.inl (Finset.one_lt_card.mpr ⟨z, hz, y, history, by simpa [hy] using hzv⟩)
        · exact Or.inr ⟨y, rfl, hy⟩
    · rintro (h | ⟨z, hz, hzv⟩)
      · exact Or.inl h
      · cases hz
        exact Or.inr ⟨y, history, hzv⟩

/-- Injectively renaming the alphabet renames exactly the stored values. -/
theorem completedSuccessors_map [DecidableEq α] [DecidableEq β] (f : α → β)
    (hf : Function.Injective f) (xs : ℕ → α) (n : ℕ) (k : α) :
    completedSuccessors (fun j => f (xs j)) n (f k) = (completedSuccessors xs n k).image f := by
  induction n with
  | zero => simp [completedSuccessors]
  | succ n ih =>
    by_cases h : xs n = k <;> simp [completedSuccessors, hf.eq_iff, h, ih]

/-- Apply the ordinary retrieval multiplier only on a single distinct
completed successor. The retrieved value and all learned tables stay shared. -/
def agreementGain [DecidableEq α] (gain factor : ℝ) (xs : ℕ → α) (i : ℕ) : ℝ :=
  if (completedSuccessors xs (i-1) (xs i)).card = 1 then retrievalGain gain factor xs i else gain

/-- Agreement and the resulting gain are preserved under an injective
renaming of the entire prefix, even into a larger vocabulary. -/
theorem agreementGain_map [DecidableEq α] [DecidableEq β] (f : α → β)
    (hf : Function.Injective f) (gain factor : ℝ) (xs : ℕ → α) (i : ℕ) :
    agreementGain gain factor (fun j => f (xs j)) i = agreementGain gain factor xs i := by
  simp only [agreementGain, completedSuccessors_map f hf, Finset.card_image_of_injective _ hf,
    retrievalGain_map f hf]

/-- Put the agreement-dependent gain on the normalized hidden residual. -/
noncomputable def Model.agreementGainLogit [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    (strength gain factor eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (c : Fin p) : ℝ :=
  M.displacedSuccessorLogit T strength (agreementGain gain factor xs i) eps scale xs L i c

/-- Two conflicting completed writes recover the constant-gain model's
entire score vector for arbitrary weights. -/
theorem Model.agreementGainLogit_eq_of_conflict [NeZero p] (M : Model D d H F p)
    (T : PairTables D p) (strength gain factor eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ)
    {u v : Fin p} (hu : u ∈ completedSuccessors xs (i-1) (xs i))
    (hv : v ∈ completedSuccessors xs (i-1) (xs i)) (hne : u ≠ v) (c : Fin p) :
    M.agreementGainLogit T strength gain factor eps scale xs L i c =
      M.displacedSuccessorLogit T strength gain eps scale xs L i c := by
  have hcard := Finset.one_lt_card.mpr ⟨u, hu, v, hv, hne⟩
  have h : (completedSuccessors xs (i-1) (xs i)).card ≠ 1 := by omega
  simp [agreementGainLogit, agreementGain, h]

example (xs : ℕ → ℕ) (i : ℕ) :
    agreementGain 1 2 (fun j => xs j + 5) i = agreementGain 1 2 xs i :=
  agreementGain_map (fun x : ℕ => x+5) (by intro a b h; dsimp at h; omega) 1 2 xs i

example (M : Model 64 16 4 86 4) (T : PairTables 64 4) (L : ℕ) (c : Fin 4) :
    let xs := fun i => (![0,1,0,2,0] : Fin 5 → Fin 4) (Fin.ofNat 5 i)
    M.agreementGainLogit T 1 (1/2) 2 1 1 xs L 4 c =
      M.displacedSuccessorLogit T 1 (1/2) 1 1 xs L 4 c :=
  M.agreementGainLogit_eq_of_conflict T _ _ _ _ _ _ L 4
    (u := 1) (v := 2) (by decide) (by decide) (by decide) c

end Transformer.Phasor

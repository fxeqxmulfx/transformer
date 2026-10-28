module

public import Transformer.Grokking.FlatSpectrum.ConnectedRecovery

/-!
# Two diagonals minus a cell

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel*. Its Numerical check
lists "diagonals `{0, 1}` minus one cell | `2p − 1`" among the patterns from which ADMM
recovers `ψ(a + b)`, and *Patterns* says of "two diagonals minus a cell" that Theorem E applies:
the pattern is rigid and contains a permutation graph.

Added, for every `N ≥ 2` and offsets `s₁`, `s₂` of `ℤ/N` with `d = s₂ − s₁` a unit:

* The diagonals `b = a + s₁` and `b = a + s₂` without a cell `c` off the first one still
  connect the rows with the columns (`connected_of_diagonals_diff`). Row `a ≠ c.1` reaches row
  `a + d` through the column `a + s₂`; the rows `c.1 + d, c.1 + 2d, …, c.1 + N d = c.1` are all
  the rows, since `d` is a unit; and column `b` meets row `b − s₁`.
* Whichever cell is removed, one diagonal survives whole, the graph of a permutation; swapping
  `s₁` and `s₂` reduces a cell of the first diagonal to the case above
  (`connected_and_perm_of_diagonals_diff`).
* So M recovers every `N × N` matrix `u vᵀ` with `|u_i v_j| = 1` from every pattern that holds
  the two diagonals but at most one cell (`forall_isMinOn_vecMulVec_of_diagonals_diff`),
  `ψ(a + b)` among them (`forall_isMinOn_addKernel_of_diagonals_diff`); for prime `p`, from any
  two distinct diagonals but one cell (`forall_isMinOn_addKernel_of_diagonals_diff_of_ne`). The
  two diagonals have `2N` cells (`ncard_twoDiagonals`), so one cell less is `2N − 1`, as few as
  any recovering set has (`two_mul_card_le_ncard_add_one_of_forall_isMinOn`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix SimpleGraph

/-- The diagonals `b = a + s₁` and `b = a + s₂` of `ℤ/p × ℤ/p`. -/
def twoDiagonals {p : ℕ} (s₁ s₂ : ZMod p) : Set (ZMod p × ZMod p) :=
  {ij | ij.2 = ij.1 + s₁ ∨ ij.2 = ij.1 + s₂}

lemma twoDiagonals_comm {p : ℕ} (s₁ s₂ : ZMod p) : twoDiagonals s₁ s₂ = twoDiagonals s₂ s₁ :=
  Set.ext fun _ => or_comm

/-- A cell off the diagonal `b = a + s` is no cell `(a, a + s)`. -/
lemma mk_add_ne_of_ne {p : ℕ} {s : ZMod p} {c : ZMod p × ZMod p} (hc : c.2 ≠ c.1 + s) (a : ZMod p) :
    (a, a + s) ≠ c := fun h => hc (by subst h; rfl)

example := mk_add_ne_of_ne (p := 2) (s := 0) (c := (0, 1)) (by decide) 0

variable {p : ℕ} [NeZero p]

/-- **Two diagonals without a cell off the first one connect the rows with the columns**
(added): for `p ≥ 2` and `s₂ − s₁` a unit, the cells of `Ω` connect the rows with the columns
when `Ω` holds the diagonals `b = a + s₁` and `b = a + s₂` but the cell `c`, and `c` is off
the first diagonal. -/
theorem connected_of_diagonals_diff (hp : 1 < p) {s₁ s₂ : ZMod p} (hs : IsUnit (s₂ - s₁))
    {c : ZMod p × ZMod p} (hc : c.2 ≠ c.1 + s₁) {Ω : Set (ZMod p × ZMod p)}
    (hΩ : twoDiagonals s₁ s₂ \ {c} ⊆ Ω) : (patternGraph Ω).Connected := by
  have h₁ (a : ZMod p) : (a, a + s₁) ∈ Ω := hΩ ⟨Or.inl rfl, mk_add_ne_of_ne hc a⟩
  have h₂ (a : ZMod p) (ha : a ≠ c.1) : (a, a + s₂) ∈ Ω :=
    hΩ ⟨Or.inr rfl, fun h => ha (by subst h; rfl)⟩
  -- Row `a ≠ c.1` reaches row `a + d` through the column `a + s₂`.
  have step (a : ZMod p) (ha : a ≠ c.1) :
      (patternGraph Ω).Reachable (.inl a) (.inl (a + (s₂ - s₁))) := by
    have h := h₁ (a + (s₂ - s₁))
    rw [add_assoc, sub_add_cancel] at h
    exact (patternGraph_adj_inl_inr.mpr (h₂ a ha)).reachable.trans
      (patternGraph_adj_inr_inl.mpr h).reachable
  -- The rows `c.1 + k d` with `0 < k < p` are not `c.1`.
  have hne (k : ℕ) (hk : k + 1 < p) : c.1 + ((k + 1 : ℕ) : ZMod p) * (s₂ - s₁) ≠ c.1 :=
    fun h => absurd (Nat.le_of_dvd k.succ_pos ((ZMod.natCast_eq_zero_iff _ _).mp
      (hs.mul_left_eq_zero.mp (add_eq_left.mp h)))) (not_le.mpr hk)
  have walk (k : ℕ) (hk : k < p) : (patternGraph Ω).Reachable (.inl (c.1 + (s₂ - s₁)))
      (.inl (c.1 + ((k + 1 : ℕ) : ZMod p) * (s₂ - s₁))) := by
    induction k with
    | zero => rw [zero_add, Nat.cast_one, one_mul]
    | succ k ih =>
      have e : c.1 + ((k + 1 + 1 : ℕ) : ZMod p) * (s₂ - s₁) =
          c.1 + ((k + 1 : ℕ) : ZMod p) * (s₂ - s₁) + (s₂ - s₁) := by
        push_cast
        ring
      rw [e]
      exact (ih (by omega)).trans (step _ (hne k hk))
  -- Every row is `c.1 + k d` for some `0 < k ≤ p`.
  have hrow (a : ZMod p) : (patternGraph Ω).Reachable (.inl (c.1 + (s₂ - s₁))) (.inl a) := by
    obtain ⟨u, hu⟩ := hs
    have hm : c.1 + (((a - c.1) * ↑u⁻¹ : ZMod p).val : ZMod p) * (s₂ - s₁) = a := by
      rw [ZMod.natCast_zmod_val, ← hu, mul_assoc, Units.inv_mul, mul_one, add_sub_cancel]
    have hlt := ZMod.val_lt ((a - c.1) * ↑u⁻¹ : ZMod p)
    rcases Nat.eq_zero_or_pos ((a - c.1) * ↑u⁻¹ : ZMod p).val with h0 | hpos
    · have h := walk (p - 1) (by omega)
      rw [Nat.sub_add_cancel (by omega : 1 ≤ p), ZMod.natCast_self, zero_mul, add_zero] at h
      rw [h0, Nat.cast_zero, zero_mul, add_zero] at hm
      rwa [← hm]
    · have h := walk (((a - c.1) * ↑u⁻¹ : ZMod p).val - 1) (by omega)
      rwa [Nat.sub_add_cancel hpos, hm] at h
  have hall : ∀ x, (patternGraph Ω).Reachable (.inl (c.1 + (s₂ - s₁))) x
    | .inl a => hrow a
    | .inr b => (hrow (b - s₁)).trans
        (Adj.reachable (patternGraph_adj_inl_inr.mpr (by simpa using h₁ (b - s₁))))
  exact (connected_iff _).mpr ⟨fun u v => (hall u).symm.trans (hall v), ⟨.inl c.1⟩⟩

/-- `{0, 1}` minus `(2, 0)` in `ℤ/3`, the staircase of `FewestCells` up to the labels. -/
example := connected_of_diagonals_diff (p := 3) (by norm_num) (s₁ := 0) (s₂ := 1) (by simp)
  (c := (2, 0)) (by decide) subset_rfl

/-- **Two diagonals minus any cell connect the rows with the columns and keep a permutation**
(added): for `p ≥ 2` and `s₂ − s₁` a unit, if `Ω` holds the diagonals `b = a + s₁` and
`b = a + s₂` but at most the cell `c`, its cells connect the rows with the columns and contain
the graph of a permutation. -/
theorem connected_and_perm_of_diagonals_diff (hp : 1 < p) {s₁ s₂ : ZMod p}
    (hs : IsUnit (s₂ - s₁)) (c : ZMod p × ZMod p) {Ω : Set (ZMod p × ZMod p)}
    (hΩ : twoDiagonals s₁ s₂ \ {c} ⊆ Ω) :
    (patternGraph Ω).Connected ∧ ∃ σ : Equiv.Perm (ZMod p), ∀ i, (i, σ i) ∈ Ω := by
  by_cases hc : c.2 = c.1 + s₁
  · have : Fact (1 < p) := ⟨hp⟩
    have hc₂ : c.2 ≠ c.1 + s₂ := fun h =>
      hs.ne_zero (sub_eq_zero.mpr (add_left_cancel (h.symm.trans hc)))
    rw [twoDiagonals_comm] at hΩ
    refine ⟨connected_of_diagonals_diff hp (by rw [← neg_sub]; exact hs.neg) hc₂ hΩ,
      Equiv.addRight s₂, fun i => hΩ ⟨Or.inl rfl, mk_add_ne_of_ne hc₂ i⟩⟩
  · exact ⟨connected_of_diagonals_diff hp hs hc hΩ, Equiv.addRight s₁,
      fun i => hΩ ⟨Or.inl rfl, mk_add_ne_of_ne hc i⟩⟩

/-- In `ℤ/3`, `{0, 1}` minus `(1, 1)`, a cell of the diagonal `0`. -/
example := connected_and_perm_of_diagonals_diff (p := 3) (by norm_num) (s₁ := 0) (s₂ := 1)
  (by simp) (1, 1) subset_rfl

/-- **M recovers every rank-one matrix with entries of modulus one from two diagonals minus a
cell** (added): for `p ≥ 2` and `s₂ − s₁` a unit, from every `Ω` that holds the diagonals
`b = a + s₁` and `b = a + s₂` but at most one cell, M recovers every `u vᵀ` with
`|u_i v_j| = 1`. -/
theorem forall_isMinOn_vecMulVec_of_diagonals_diff (hp : 1 < p) {s₁ s₂ : ZMod p}
    (hs : IsUnit (s₂ - s₁)) (c : ZMod p × ZMod p) {Ω : Set (ZMod p × ZMod p)}
    (hΩ : twoDiagonals s₁ s₂ \ {c} ⊆ Ω) {u v : ZMod p → ℂ}
    (h : ∀ i j, ‖vecMulVec u v i j‖ = 1) :
    ∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2},
      (IsMinOn nuclearNorm {X' : Matrix (ZMod p) (ZMod p) ℂ |
        ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2} X ↔ X = vecMulVec u v) :=
  (forall_isMinOn_vecMulVec_iff_connected_of_entries h).mpr
    (connected_and_perm_of_diagonals_diff hp hs c hΩ)

/-- The all-ones `2 × 2` matrix from `{0, 1}` minus `(0, 0)`. -/
example := forall_isMinOn_vecMulVec_of_diagonals_diff (p := 2) (by norm_num) (s₁ := 0)
  (s₂ := 1) (by simp) (0, 0) subset_rfl (u := fun _ => 1) (v := fun _ => 1)
  fun i j => by simp [vecMulVec_apply]

/-- **M recovers `ψ(a + b)` from two diagonals minus a cell** (`review/flat_spectrum.md`,
Numerical check, "diagonals `{0, 1}` minus one cell", whose check removes `(0, 1)`; added for
every cell and every unit offset): for `p ≥ 2` and `s₂ − s₁` a unit, from every `Ω` that holds
the diagonals `b = a + s₁` and `b = a + s₂` but at most one cell. -/
theorem forall_isMinOn_addKernel_of_diagonals_diff (hp : 1 < p) {s₁ s₂ : ZMod p}
    (hs : IsUnit (s₂ - s₁)) (c : ZMod p × ZMod p) {Ω : Set (ZMod p × ZMod p)}
    (hΩ : twoDiagonals s₁ s₂ \ {c} ⊆ Ω) :
    ∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2} X ↔
        X = addKernel p) :=
  forall_isMinOn_addKernel_iff_connected.mpr (connected_and_perm_of_diagonals_diff hp hs c hΩ)

/-- The review's pattern, `{0, 1}` minus `(0, 1)`, for `p = 4`. -/
example := forall_isMinOn_addKernel_of_diagonals_diff (p := 4) (by norm_num) (s₁ := 0)
  (s₂ := 1) (by simp) (0, 1) subset_rfl

/-- **For prime `p`, any two distinct diagonals minus a cell recover `ψ(a + b)`** (added). -/
theorem forall_isMinOn_addKernel_of_diagonals_diff_of_ne {p : ℕ} [Fact p.Prime]
    {s₁ s₂ : ZMod p} (hs : s₁ ≠ s₂) (c : ZMod p × ZMod p) {Ω : Set (ZMod p × ZMod p)}
    (hΩ : twoDiagonals s₁ s₂ \ {c} ⊆ Ω) :
    ∀ X ∈ {X' : Matrix (ZMod p) (ZMod p) ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = addKernel p ij.1 ij.2} X ↔
        X = addKernel p) :=
  forall_isMinOn_addKernel_of_diagonals_diff (Fact.out : p.Prime).one_lt
    (sub_ne_zero.mpr hs.symm).isUnit c hΩ

example := forall_isMinOn_addKernel_of_diagonals_diff_of_ne (p := 3) (s₁ := 0) (s₂ := 2)
  (by decide) (1, 1) subset_rfl

/-- **Two distinct diagonals have `2p` cells.** -/
theorem ncard_twoDiagonals {s₁ s₂ : ZMod p} (hs : s₁ ≠ s₂) :
    (twoDiagonals s₁ s₂).ncard = 2 * p := by
  have e : twoDiagonals s₁ s₂ =
      Set.range (fun a => (a, a + s₁)) ∪ Set.range fun a => (a, a + s₂) := by
    ext ⟨a, b⟩
    simp [twoDiagonals, eq_comm]
  have hD : Disjoint (Set.range fun a : ZMod p => (a, a + s₁))
      (Set.range fun a : ZMod p => (a, a + s₂)) := Set.disjoint_left.mpr <| by
    rintro _ ⟨a, rfl⟩ ⟨a', h⟩
    exact hs (add_left_cancel ((Prod.mk.inj h).1 ▸ (Prod.mk.inj h).2)).symm
  rw [e, Set.ncard_union_eq hD, Set.ncard_range_of_injective fun a a' h => congrArg Prod.fst h,
    Set.ncard_range_of_injective fun a a' h => congrArg Prod.fst h, Nat.card_zmod, two_mul]

example : (twoDiagonals (p := 3) 0 1).ncard = 6 := ncard_twoDiagonals (by decide)

/-- **Two distinct diagonals minus one of their cells have `2p − 1` cells.** -/
theorem ncard_twoDiagonals_diff {s₁ s₂ : ZMod p} (hs : s₁ ≠ s₂) {c : ZMod p × ZMod p}
    (hc : c ∈ twoDiagonals s₁ s₂) : (twoDiagonals s₁ s₂ \ {c}).ncard = 2 * p - 1 := by
  rw [Set.ncard_sdiff_singleton_of_mem hc, ncard_twoDiagonals hs]

example : (twoDiagonals (p := 3) 0 1 \ {(0, 1)}).ncard = 5 :=
  ncard_twoDiagonals_diff (by decide) (Or.inr rfl)

end Transformer.Grokking.FlatSpectrum

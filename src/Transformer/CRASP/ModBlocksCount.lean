module

public import Transformer.CRASP.ModBlocks
public import Transformer.CRASP.PositionalReductionCount

/-!
# Blocks of one period turn `TL[◁#, MOD]` into `TL[◁#]`: positions and counts

The facts that `Transformer.CRASP.ModBlocksCorrect` puts together. In `g y₁ ⋯ g yₙ`, blocks of
`M` symbols each, the symbol `j < M` of block `i` is the symbol `j` of `g yᵢ`
(`getElem?_flatMap_of_length_eq`). `MOD_m^r` at the position `ρ` of block `i` is what `modTr`
says (`sat_modTr`). A translated term takes a value when its sums balance (`BlockSum.Val`),
and then a comparison of two of them means what it says (`BlockSum.sat_lt`).

A count of `φ` splits as in Appendix F of arXiv:2506.16055v3 (`TermP.val_countL_blockEnd`,
`TermP.val_countL_spread`, here for any string in place of the spread one): the rest of the
current block (`TermP.val_countL_blockRest`), the full blocks, each counted position by position
(`TermP.val_countL_blocks`), and nothing after a prefix (`TermP.val_countL_append`).
-/

@[expose] public section

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} {g : Bool → List σ} {M : ℕ}

/-- Blocks of `M` symbols fill `M` positions per bit. -/
theorem length_flatMap_of_length_eq (hg : ∀ c, (g c).length = M) (y : List Bool) :
    (y.flatMap g).length = M * y.length := by
  induction y with
  | nil => simp
  | cons c y ih => rw [List.flatMap_cons, List.length_append, ih, hg, List.length_cons,
      Nat.mul_add_one, Nat.add_comm]

/-- The symbol `j < M` of block `i` of `g y₁ ⋯ g yₙ` is the symbol `j` of `g yᵢ`. -/
theorem getElem?_flatMap_of_length_eq (hg : ∀ c, (g c).length = M) (y : List Bool) {i j : ℕ}
    (hi : i < y.length) (hj : j < M) : (y.flatMap g)[M * i + j]? = (g y[i])[j]? := by
  induction y generalizing i with
  | nil => exact absurd hi (Nat.not_lt_zero _)
  | cons c y ih =>
      rcases i with _ | i
      · rw [List.flatMap_cons, Nat.mul_zero, Nat.zero_add,
          List.getElem?_append_left (by rw [hg]; exact hj)]
        rfl
      · rw [List.flatMap_cons, Nat.mul_add_one, List.getElem?_append_right (by rw [hg]; omega), hg,
          show M * i + M + j - M = M * i + j by omega, ih (Nat.lt_of_succ_lt_succ hi)]
        rfl

/-- The hypotheses of `length_flatMap_of_length_eq` and `getElem?_flatMap_of_length_eq` are
satisfiable: blocks `c c`, and the second symbol of the second block. -/
example : ([true, false].flatMap fun c => [c, c]).length = 2 * 2 ∧
    ([true, false].flatMap fun c => [c, c])[2 * 1 + 1]? = [false, false][1]? :=
  ⟨length_flatMap_of_length_eq (g := fun c => [c, c]) (M := 2) (fun _ => rfl) _,
    getElem?_flatMap_of_length_eq (g := fun c => [c, c]) (M := 2) (fun _ => rfl) [true, false]
      (i := 1) (j := 1) (by decide) (by decide)⟩

/-- **`MOD_m^r` on blocks.**  At the position `ρ` of block `i + 1`, that is at `M i + ρ`,
`modTr` holds iff `M i + ρ ≡ r (mod m)`, for `m = 0` and for every `m ∣ M`. -/
theorem sat_modTr {m : ℕ} (hM : 0 < M) (hm : m ≠ 0 → m ∣ M) (y : List Bool) (i ρ r : ℕ) :
    (modTr M ρ m r).sat y (i + 1) = decide ((M * i + ρ) % m = r % m) := by
  unfold modTr
  split_ifs with h0 h
  · subst h0
    obtain ⟨hρr, hdiv⟩ := h
    have hq : M * ((r - ρ) / M) = r - ρ := Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hdiv)
    have hval : (Term.countL (Form.ofBool true : Form Bool)).val y (i + 1) = i + 1 := by
      simp [Term.val]
    rw [Form.sat_eq, hval, Term.val_ofPos, Nat.mod_zero, Nat.mod_zero]
    refine decide_eq_decide.mpr ⟨fun h => ?_, fun h => ?_⟩
    · rw [show i = (r - ρ) / M by omega, hq]
      omega
    · have := Nat.eq_of_mul_eq_mul_left hM (show M * i = M * ((r - ρ) / M) by omega)
      omega
  · subst h0
    rw [Form.sat_ofBool, Nat.mod_zero, Nat.mod_zero]
    refine (decide_eq_false fun h' => h ⟨by omega, ?_⟩).symm
    rw [show r - ρ = M * i by omega, Nat.mul_mod_right]
  · obtain ⟨k, hk⟩ := hm h0
    rw [Form.sat_ofBool, hk, Nat.mul_assoc, Nat.mul_add_mod_self_left]

/-- The hypotheses of `sat_modTr` are satisfiable: blocks of two, `MOD_0^3` at the first position
of the second block, and `MOD_2^1` anywhere. -/
example (y : List Bool) (i : ℕ) : (modTr 2 1 0 3).sat y (1 + 1) = decide ((2 * 1 + 1) % 0 = 3 % 0) ∧
    (modTr 2 1 2 1).sat y (i + 1) = decide ((2 * i + 1) % 2 = 1 % 2) :=
  ⟨sat_modTr (by decide) (fun h => absurd rfl h) y 1 1 3,
    sat_modTr (by decide) (fun _ => dvd_refl 2) y i 1 1⟩

namespace BlockSum

/-- **`b` takes the value `v` at `i`**: `const + Σ terms + #poss = v + #negs`, so that the
indicators subtracted never exceed the rest. -/
def Val (b : BlockSum) (y : List Bool) (i v : ℕ) : Prop :=
  b.const + (b.terms.map (·.val y i)).sum + b.poss.countP (·.sat y i) =
    v + b.negs.countP (·.sat y i)

theorem Val.add {b₁ b₂ : BlockSum} {y : List Bool} {i v₁ v₂ : ℕ} (h₁ : b₁.Val y i v₁)
    (h₂ : b₂.Val y i v₂) : (b₁.add b₂).Val y i (v₁ + v₂) := by
  unfold Val at *
  simp only [BlockSum.add, List.map_append, List.sum_append, List.countP_append]
  omega

/-- A comparison of translated terms compares their values. -/
theorem sat_lt {b₁ b₂ : BlockSum} {y : List Bool} {i v₁ v₂ : ℕ} (h₁ : b₁.Val y i v₁)
    (h₂ : b₂.Val y i v₂) : (b₁.lt b₂).sat y i = decide (v₁ < v₂) := by
  unfold Val at h₁ h₂
  rw [BlockSum.lt, Form.sat_ltSum, List.countP_append, List.countP_append]
  exact decide_eq_decide.mpr (by omega)

/-- The hypotheses of `Val.add` and `sat_lt` are satisfiable: the constants `1` and `2`. -/
example (y : List Bool) (i : ℕ) : ((mk 1 [] [] []).add (mk 1 [] [] [])).Val y i 2 ∧
    ((mk 1 [] [] []).lt (mk 2 [] [] [])).sat y i = true :=
  ⟨Val.add (v₁ := 1) (v₂ := 1) (by simp [Val]) (by simp [Val]),
    sat_lt (v₁ := 1) (v₂ := 2) (by simp [Val]) (by simp [Val])⟩

end BlockSum

variable [DecidableEq σ]

/-- **The rest of the current block.**  The count of `φ` up to `M i + ρ`, plus the `ρ' ∈ (ρ, M]`
at which `T ρ'` holds, is the count up to the end `M i + M` of the block, when `T ρ'` holds at
`i + 1` exactly where `φ` holds at `M i + ρ'` (`TermP.val_countL_blockEnd` for any string). -/
theorem TermP.val_countL_blockRest (W : List σ) (y : List Bool) (φ : FormP σ)
    (T : ℕ → Form Bool) {i ρ : ℕ} (hρ : ρ ≤ M)
    (hT : ∀ ρ', ρ < ρ' → ρ' ≤ M → (T ρ').sat y (i + 1) = φ.sat W (M * i + ρ')) :
    (TermP.countL φ).val W (M * i + ρ) +
        ((List.range' (ρ + 1) (M - ρ)).map T).countP (·.sat y (i + 1)) =
      (TermP.countL φ).val W (M * i + M) := by
  rw [TermP.val, TermP.val, ← List.countP_eq_length_filter, ← List.countP_eq_length_filter,
    show M * i + M = M * i + ρ + (M - ρ) by omega, countP_range'_add _ (M * i + ρ) (M - ρ),
    ← List.map_add_range' (a := ρ) 1 (M - ρ) 1, List.countP_map, List.countP_map]
  congr 1
  refine List.countP_congr fun j hj => ?_
  rw [List.mem_range'_1] at hj
  simp only [Function.comp_apply]
  rw [hT (ρ + j) (by omega) (by omega), Nat.add_assoc]

/-- **The full blocks.**  The count of `φ` up to the end `M n` of block `n` is the sum over
`ρ' ∈ [1, M]` of the counts up to `n` of `T ρ'`, when `T ρ'` holds at every `i + 1 ≤ n` exactly
where `φ` holds at `M i + ρ'` (`TermP.val_countL_spread` for any string). -/
theorem TermP.val_countL_blocks (W : List σ) (y : List Bool) (φ : FormP σ) (T : ℕ → Form Bool)
    (M n : ℕ) (hT : ∀ i ρ', i < n → 1 ≤ ρ' → ρ' ≤ M →
      (T ρ').sat y (i + 1) = φ.sat W (M * i + ρ')) :
    (TermP.countL φ).val W (M * n) =
      (((List.range' 1 M).map fun ρ' => Term.countL (T ρ')).map (·.val y n)).sum := by
  induction n with
  | zero => simp [TermP.val, Term.val, Function.comp_def]
  | succ n ih =>
      have ih := ih fun i ρ' h₁ h₂ h₃ => hT i ρ' (by omega) h₂ h₃
      rw [TermP.val, ← List.countP_eq_length_filter] at ih ⊢
      rw [Nat.mul_add_one, countP_range'_add _ (M * n) M, ih]
      simp only [List.map_map, Function.comp_def, val_countL_succ, List.sum_map_add,
        sum_map_ite_eq_countP]
      congr 1
      refine List.countP_congr fun j hj => ?_
      rw [List.mem_range'_1] at hj
      rw [hT n j (by omega) (by omega) (by omega)]

/-- A count up to a position of a prefix ignores what follows it (`FormP.sat_append`). -/
theorem TermP.val_countL_append (u v : List σ) (φ : FormP σ) {i : ℕ} (hi : i ≤ u.length) :
    (TermP.countL φ).val (u ++ v) i = (TermP.countL φ).val u i := by
  rw [TermP.val, TermP.val]
  congr 1
  refine List.filter_congr fun j hj => ?_
  rw [List.mem_range'_1] at hj
  rw [FormP.sat_append u v φ j (by omega) (by omega)]

/-- The hypotheses of the three counts above are satisfiable, and not only vacuously: `⊤`
simulates `MOD_1^0` everywhere, and the empty string is a prefix of any. -/
example (W : List σ) (y : List Bool) (i n : ℕ) :
    (TermP.countL (FormP.mod 1 0 : FormP σ)).val W (2 * i + 1) +
        ((List.range' 2 1).map fun _ => (Form.ofBool true : Form Bool)).countP
          (·.sat y (i + 1)) = (TermP.countL (FormP.mod 1 0 : FormP σ)).val W (2 * i + 2) ∧
      (TermP.countL (FormP.mod 1 0 : FormP σ)).val W (2 * n) =
        (((List.range' 1 2).map fun _ => Term.countL (Form.ofBool true : Form Bool)).map
          (·.val y n)).sum ∧
      (TermP.countL (FormP.mod 1 0 : FormP σ)).val ([] ++ W) 0 =
        (TermP.countL (FormP.mod 1 0 : FormP σ)).val [] 0 :=
  ⟨TermP.val_countL_blockRest (M := 2) (ρ := 1) W y _ _ (by decide) fun _ _ _ => by
      simp [FormP.sat, Nat.mod_one],
    TermP.val_countL_blocks W y _ _ 2 n fun _ _ _ _ _ => by simp [FormP.sat, Nat.mod_one],
    TermP.val_countL_append [] W _ le_rfl⟩

end CRASP
end Transformer

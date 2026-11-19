module

public import Transformer.CRASP.ModBlocksCount

/-!
# Blocks of one period turn `TL[◁#, MOD]` into `TL[◁#]`: the simulation

In no paper; the reduction of `lem:tlclpos_reduction` of arXiv:2506.16055v3, Appendix F, without
`Y` (`Transformer.CRASP.ModBlocks`). Let `φ` be a formula of `TL[◁#, MOD]`, `M > 0` a multiple of
its positive moduli, and let every block `g c` have `M` symbols. Then `T_ρ⟦φ⟧` holds at the bit
`i + 1` of `y` iff `φ` holds at the position `M i + ρ` of `g y₁ ⋯ g yₙ`, the `ρ`-th of block
`i + 1` (`FormP.sat_blockTr`). With `s` in place of the last block, the translation holds at the
last bit of `x c`, whatever `c`, iff `φ` holds at the position `M |x| + ρ` of `g x₁ ⋯ g xₙ · s`
(`FormP.sat_lastTr`); at `ρ = |s|` the translation accepts `x c` iff `φ` accepts
`g x₁ ⋯ g xₙ · s` (`FormP.models_lastTr`).
-/

@[expose] public section

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ] {g : Bool → List σ} {M : ℕ}

mutual

/-- **`T_ρ⟦φ⟧` simulates `φ`.**  For `φ` free of `Y`, `M > 0` a multiple of its positive moduli
and blocks of `M` symbols, `T_ρ⟦φ⟧` holds at the bit `i + 1` of `y` iff `φ` holds at the
position `M i + ρ` of `g y₁ ⋯ g yₙ`, `1 ≤ ρ ≤ M`. -/
theorem FormP.sat_blockTr (hM : 0 < M) (hg : ∀ c, (g c).length = M) (y : List Bool) :
    ∀ (φ : FormP σ), φ.prevFree = true → φ.modLcm ∣ M → ∀ {i ρ : ℕ}, i < y.length → 1 ≤ ρ →
      ρ ≤ M → (φ.blockTr g M ρ).sat y (i + 1) = φ.sat (y.flatMap g) (M * i + ρ)
  | .sym a, _, _, i, ρ, hi, hρ₁, hρ₂ => by
      rw [FormP.blockTr, Form.sat_ite, Form.sat_ofBool, Form.sat_ofBool, FormP.sat,
        show M * i + ρ - 1 = M * i + (ρ - 1) by omega,
        getElem?_flatMap_of_length_eq hg y hi (by omega)]
      simp only [Form.sat, Nat.add_sub_cancel, List.getElem?_eq_getElem hi]
      cases y[i] <;> simp
  | .mod m r, _, hd, _, _, _, _, _ => by
      have hm : m ≠ 0 → m ∣ M := fun h0 => by simpa [FormP.modLcm, h0] using hd
      rw [FormP.blockTr, sat_modTr hM hm, FormP.sat]
  | .prev _, hp, _, _, _, _, _, _ => absurd hp (by simp [FormP.prevFree])
  | .lt t₁ t₂, hp, hd, _, _, hi, hρ₁, hρ₂ => by
      rw [FormP.prevFree, Bool.and_eq_true] at hp
      rw [FormP.modLcm] at hd
      rw [FormP.blockTr, FormP.sat,
        BlockSum.sat_lt (t₁.val_blockTr hM hg y hp.1 ((Nat.dvd_lcm_left _ _).trans hd) hi hρ₁ hρ₂)
          (t₂.val_blockTr hM hg y hp.2 ((Nat.dvd_lcm_right _ _).trans hd) hi hρ₁ hρ₂)]
  | .neg φ, hp, hd, _, _, hi, hρ₁, hρ₂ => by
      rw [FormP.prevFree] at hp
      rw [FormP.modLcm] at hd
      rw [FormP.blockTr, FormP.sat, Form.sat, φ.sat_blockTr hM hg y hp hd hi hρ₁ hρ₂]
  | .and φ₁ φ₂, hp, hd, _, _, hi, hρ₁, hρ₂ => by
      rw [FormP.prevFree, Bool.and_eq_true] at hp
      rw [FormP.modLcm] at hd
      rw [FormP.blockTr, FormP.sat, Form.sat,
        φ₁.sat_blockTr hM hg y hp.1 ((Nat.dvd_lcm_left _ _).trans hd) hi hρ₁ hρ₂,
        φ₂.sat_blockTr hM hg y hp.2 ((Nat.dvd_lcm_right _ _).trans hd) hi hρ₁ hρ₂]

/-- `T_ρ⟦t⟧` takes the value of `t` at the position `M i + ρ`. -/
theorem TermP.val_blockTr (hM : 0 < M) (hg : ∀ c, (g c).length = M) (y : List Bool) :
    ∀ (t : TermP σ), t.prevFree = true → t.modLcm ∣ M → ∀ {i ρ : ℕ}, i < y.length → 1 ≤ ρ →
      ρ ≤ M → (t.blockTr g M ρ).Val y (i + 1) (t.val (y.flatMap g) (M * i + ρ))
  | .countL φ, hp, hd, i, _, hi, _, hρ₂ => by
      rw [TermP.prevFree] at hp
      rw [TermP.modLcm] at hd
      rw [TermP.blockTr, BlockSum.Val]
      simp only [List.countP_nil, Nat.add_zero, Nat.zero_add]
      rw [TermP.val_countL_blockRest (y.flatMap g) y φ (fun ρ' => φ.blockTr g M ρ') hρ₂
          fun ρ' h₁ h₂ => φ.sat_blockTr hM hg y hp hd hi (by omega) h₂, ← Nat.mul_add_one,
        TermP.val_countL_blocks (y.flatMap g) y φ (fun ρ' => φ.blockTr g M ρ') M (i + 1)
          fun i' ρ' h₁ h₂ h₃ => φ.sat_blockTr hM hg y hp hd (i := i') (by omega) h₂ h₃]
  | .add t₁ t₂, hp, hd, _, _, hi, hρ₁, hρ₂ => by
      rw [TermP.prevFree, Bool.and_eq_true] at hp
      rw [TermP.modLcm] at hd
      rw [TermP.blockTr, TermP.val]
      exact (t₁.val_blockTr hM hg y hp.1 ((Nat.dvd_lcm_left _ _).trans hd) hi hρ₁ hρ₂).add
        (t₂.val_blockTr hM hg y hp.2 ((Nat.dvd_lcm_right _ _).trans hd) hi hρ₁ hρ₂)
  | .one, _, _, _, _, _, _, _ => by
      rw [TermP.blockTr, TermP.val, BlockSum.Val]
      simp

end

mutual

/-- **The last block, `s` in its place.**  For `φ` free of `Y`, `M > 0` a multiple of its
positive moduli and blocks of `M` symbols, the translation of `φ` at `ρ ≥ 1` holds at the last
bit of `x c` iff `φ` holds at the position `M |x| + ρ` of `g x₁ ⋯ g xₙ · s`. -/
theorem FormP.sat_lastTr (hM : 0 < M) (hg : ∀ c, (g c).length = M) (x : List Bool) (c : Bool)
    (s : List σ) :
    ∀ (φ : FormP σ), φ.prevFree = true → φ.modLcm ∣ M → ∀ {ρ : ℕ}, 1 ≤ ρ →
      (φ.lastTr g M s ρ).sat (x ++ [c]) (x.length + 1) =
        φ.sat (x.flatMap g ++ s) (M * x.length + ρ)
  | .sym a, _, _, ρ, hρ => by
      rw [FormP.lastTr, Form.sat_ofBool, FormP.sat,
        List.getElem?_append_right (by rw [length_flatMap_of_length_eq hg]; omega),
        length_flatMap_of_length_eq hg, show M * x.length + ρ - 1 - M * x.length = ρ - 1 by omega]
  | .mod m r, _, hd, _, _ => by
      have hm : m ≠ 0 → m ∣ M := fun h0 => by simpa [FormP.modLcm, h0] using hd
      rw [FormP.lastTr, sat_modTr hM hm, FormP.sat]
  | .prev _, hp, _, _, _ => absurd hp (by simp [FormP.prevFree])
  | .lt t₁ t₂, hp, hd, _, hρ => by
      rw [FormP.prevFree, Bool.and_eq_true] at hp
      rw [FormP.modLcm] at hd
      rw [FormP.lastTr, FormP.sat,
        BlockSum.sat_lt (t₁.val_lastTr hM hg x c s hp.1 ((Nat.dvd_lcm_left _ _).trans hd) hρ)
          (t₂.val_lastTr hM hg x c s hp.2 ((Nat.dvd_lcm_right _ _).trans hd) hρ)]
  | .neg φ, hp, hd, _, hρ => by
      rw [FormP.prevFree] at hp
      rw [FormP.modLcm] at hd
      rw [FormP.lastTr, FormP.sat, Form.sat, φ.sat_lastTr hM hg x c s hp hd hρ]
  | .and φ₁ φ₂, hp, hd, _, hρ => by
      rw [FormP.prevFree, Bool.and_eq_true] at hp
      rw [FormP.modLcm] at hd
      rw [FormP.lastTr, FormP.sat, Form.sat,
        φ₁.sat_lastTr hM hg x c s hp.1 ((Nat.dvd_lcm_left _ _).trans hd) hρ,
        φ₂.sat_lastTr hM hg x c s hp.2 ((Nat.dvd_lcm_right _ _).trans hd) hρ]

/-- A term at the last block takes the value of `t` at the position `M |x| + ρ`. -/
theorem TermP.val_lastTr (hM : 0 < M) (hg : ∀ c, (g c).length = M) (x : List Bool) (c : Bool)
    (s : List σ) :
    ∀ (t : TermP σ), t.prevFree = true → t.modLcm ∣ M → ∀ {ρ : ℕ}, 1 ≤ ρ →
      (t.lastTr g M s ρ).Val (x ++ [c]) (x.length + 1)
        (t.val (x.flatMap g ++ s) (M * x.length + ρ))
  | .countL φ, hp, hd, ρ, _ => by
      rw [TermP.prevFree] at hp
      rw [TermP.modLcm] at hd
      have hx := length_flatMap_of_length_eq hg x
      have h₁ := TermP.val_countL_blocks ((x ++ [c]).flatMap g) (x ++ [c]) φ
        (fun ρ' => φ.blockTr g M ρ') M x.length fun i ρ' hi h₂ h₃ =>
          φ.sat_blockTr hM hg (x ++ [c]) hp hd (i := i)
            (by rw [List.length_append, List.length_singleton]; omega) h₂ h₃
      rw [List.flatMap_append, TermP.val_countL_append _ _ φ hx.ge,
        ← TermP.val_countL_append (x.flatMap g) s φ hx.ge] at h₁
      have h₂ : (TermP.countL φ).val (x.flatMap g ++ s) (M * x.length + ρ) =
          (TermP.countL φ).val (x.flatMap g ++ s) (M * x.length) +
            ((List.range' 1 ρ).map fun ρ' => φ.lastTr g M s ρ').countP
              (·.sat (x ++ [c]) (x.length + 1)) := by
        rw [TermP.val, TermP.val, ← List.countP_eq_length_filter, ← List.countP_eq_length_filter,
          countP_range'_add, List.countP_map]
        congr 1
        refine List.countP_congr fun j hj => ?_
        rw [List.mem_range'_1] at hj
        simp only [Function.comp_apply]
        rw [φ.sat_lastTr hM hg x c s hp hd (ρ := j) (by omega)]
      rw [TermP.lastTr, BlockSum.Val, h₂]
      simp only [Nat.zero_add, List.map_map, Function.comp_def, val_countL_succ,
        List.sum_map_add, sum_map_ite_eq_countP, List.countP_map] at h₁ ⊢
      omega
  | .add t₁ t₂, hp, hd, _, hρ => by
      rw [TermP.prevFree, Bool.and_eq_true] at hp
      rw [TermP.modLcm] at hd
      rw [TermP.lastTr, TermP.val]
      exact (t₁.val_lastTr hM hg x c s hp.1 ((Nat.dvd_lcm_left _ _).trans hd) hρ).add
        (t₂.val_lastTr hM hg x c s hp.2 ((Nat.dvd_lcm_right _ _).trans hd) hρ)
  | .one, _, _, _, _ => by
      rw [TermP.lastTr, TermP.val, BlockSum.Val]
      simp

end

/-- **The translation accepts `x c` iff `φ` accepts `g x₁ ⋯ g xₙ · s`**, for `s` nonempty. -/
theorem FormP.models_lastTr (hM : 0 < M) (hg : ∀ c, (g c).length = M) {φ : FormP σ}
    (hp : φ.prevFree = true) (hd : φ.modLcm ∣ M) (x : List Bool) (c : Bool) {s : List σ}
    (hs : s ≠ []) :
    (φ.lastTr g M s s.length).models (x ++ [c]) ↔ φ.models (x.flatMap g ++ s) := by
  have h := φ.sat_lastTr hM hg x c s hp hd (List.length_pos_of_ne_nil hs)
  rw [Form.models, FormP.models, List.length_append, List.length_append, List.length_singleton,
    length_flatMap_of_length_eq hg, h]

/-- The hypotheses of the five theorems above are satisfiable: `Q_b` and `◁#[Q_b]` over blocks
of one symbol, and `ab` in place of the last. -/
example : ((FormP.sym true : FormP Bool).blockTr (fun c => [c]) 1 1).sat [true] (0 + 1) =
      (FormP.sym true).sat ([true].flatMap fun c => [c]) (1 * 0 + 1) ∧
    ((TermP.countL (.sym true) : TermP Bool).blockTr (fun c => [c]) 1 1).Val [true] (0 + 1)
      ((TermP.countL (.sym true)).val ([true].flatMap fun c => [c]) (1 * 0 + 1)) ∧
    ((FormP.sym true : FormP Bool).lastTr (fun c => [c]) 1 [false, true] 2).sat
        ([true] ++ [false]) ([true].length + 1) =
      (FormP.sym true).sat (([true].flatMap fun c => [c]) ++ [false, true])
        (1 * [true].length + 2) ∧
    ((TermP.countL (.sym true) : TermP Bool).lastTr (fun c => [c]) 1 [false, true] 2).Val
        ([true] ++ [false]) ([true].length + 1)
      ((TermP.countL (.sym true)).val (([true].flatMap fun c => [c]) ++ [false, true])
        (1 * [true].length + 2)) ∧
    (((FormP.sym true : FormP Bool).lastTr (fun c => [c]) 1 [false, true]
        [false, true].length).models ([true] ++ [false]) ↔
      (FormP.sym true).models (([true].flatMap fun c => [c]) ++ [false, true])) :=
  ⟨FormP.sat_blockTr Nat.one_pos (fun _ => rfl) _ _ rfl (one_dvd _) (by decide) le_rfl le_rfl,
    TermP.val_blockTr Nat.one_pos (fun _ => rfl) _ _ rfl (one_dvd _) (by decide) le_rfl le_rfl,
    FormP.sat_lastTr Nat.one_pos (fun _ => rfl) _ _ _ _ rfl (one_dvd _) (by decide),
    TermP.val_lastTr Nat.one_pos (fun _ => rfl) _ _ _ _ rfl (one_dvd _) (by decide),
    FormP.models_lastTr Nat.one_pos (fun _ => rfl) rfl (one_dvd _) _ _ (by decide)⟩

end CRASP
end Transformer

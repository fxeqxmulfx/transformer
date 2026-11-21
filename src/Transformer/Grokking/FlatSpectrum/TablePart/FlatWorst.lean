module

public import Transformer.Grokking.FlatSpectrum.TablePart.FlatBest
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatWeight

/-!
# PT4: the worst part

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT4, *The worst*. Except
at `q = 2`, `k = l = 1`,

  `δ(H) = min(q^k, q^l)` (`minDistance_flatClass`).

The indicator of a row has `q^l` nonzero cells and that of a column `q^k`. A nonzero member with no
zero row has at least `q^k` nonzero cells, and one with no zero column at least `q^l`. A member with
a zero row `u₀` and a zero column `v₀` is `(u − u₀)ᵀ M (v − v₀)` by its mixed difference, with
`M ≠ 0`, and `FlatWeight` counts its cells. This shortens the source's case split, which treats a
zero row alone. `FlatThreshold` reads off what this says about parts.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Finset Matrix Module

variable {F ι κ : Type*} [Field F] [Fintype F] [DecidableEq F]
  [Fintype ι] [Fintype κ] [DecidableEq ι] [DecidableEq κ]

omit [Fintype F] [DecidableEq ι] [DecidableEq κ] in
/-- The indicator of a row is a member of the flat class. -/
lemma row_indicator_mem (u₀ : ι → F) :
    (fun c : (ι → F) × (κ → F) => if c.1 = u₀ then (1 : F) else 0) ∈ flatClass F ι κ :=
  mem_flatClass.2 ⟨fun u => if u = u₀ then 1 else 0, 0, 0, fun u v => by simp⟩

omit [Fintype F] [DecidableEq ι] [DecidableEq κ] in
/-- The indicator of a column is a member of the flat class. -/
lemma col_indicator_mem (v₀ : κ → F) :
    (fun c : (ι → F) × (κ → F) => if c.2 = v₀ then (1 : F) else 0) ∈ flatClass F ι κ :=
  mem_flatClass.2 ⟨0, fun v => if v = v₀ then 1 else 0, 0, fun u v => by simp⟩

/-- The indicator of a row has `q^l` nonzero cells. -/
lemma hammingNorm_row (u₀ : ι → F) :
    hammingNorm (fun c : (ι → F) × (κ → F) => if c.1 = u₀ then (1 : F) else 0) =
      Fintype.card F ^ Fintype.card κ := by
  rw [hammingNorm_eq_card (X := κ → F) (fun v => (u₀, v)) (fun _ _ h => (Prod.ext_iff.1 h).2),
    Fintype.card_fun]
  rintro ⟨u, v⟩
  constructor
  · intro h
    have hu : u = u₀ := by
      by_contra hne
      simp [hne] at h
    exact ⟨v, by simp [hu]⟩
  · rintro ⟨v', h⟩
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
    simp

/-- The indicator of a column has `q^k` nonzero cells. -/
lemma hammingNorm_col (v₀ : κ → F) :
    hammingNorm (fun c : (ι → F) × (κ → F) => if c.2 = v₀ then (1 : F) else 0) =
      Fintype.card F ^ Fintype.card ι := by
  rw [hammingNorm_eq_card (X := ι → F) (fun u => (u, v₀)) (fun _ _ h => (Prod.ext_iff.1 h).1),
    Fintype.card_fun]
  rintro ⟨u, v⟩
  constructor
  · intro h
    have hv : v = v₀ := by
      by_contra hne
      simp [hne] at h
    exact ⟨u, by simp [hv]⟩
  · rintro ⟨u', h⟩
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj h
    simp

/-- **The lower bound of PT4**: outside `q = 2`, `k = l = 1`, a nonzero member of the flat class has
at least `min(q^k, q^l)` nonzero cells. -/
lemma min_le_hammingNorm
    (hne : ¬ (Fintype.card F = 2 ∧ Fintype.card ι = 1 ∧ Fintype.card κ = 1))
    {f : (ι → F) × (κ → F) → F} (hf : f ∈ flatClass F ι κ) (h0 : f ≠ 0) :
    min (Fintype.card F ^ Fintype.card ι) (Fintype.card F ^ Fintype.card κ) ≤ hammingNorm f := by
  by_cases hrow : ∀ u : ι → F, ∃ v, f (u, v) ≠ 0
  · choose s hs using hrow
    have hle := card_le_hammingNorm (X := ι → F) (fun u => (u, s u))
      (fun _ _ h => (Prod.ext_iff.1 h).1) hs
    rw [Fintype.card_fun] at hle
    exact (min_le_left _ _).trans hle
  by_cases hcol : ∀ v : κ → F, ∃ u, f (u, v) ≠ 0
  · choose s hs using hcol
    have hle := card_le_hammingNorm (X := κ → F) (fun v => (s v, v))
      (fun _ _ h => (Prod.ext_iff.1 h).2) hs
    rw [Fintype.card_fun] at hle
    exact (min_le_right _ _).trans hle
  push Not at hrow hcol
  obtain ⟨u₀, hu₀⟩ := hrow
  obtain ⟨v₀, hv₀⟩ := hcol
  rw [← range_flatEval] at hf
  obtain ⟨p, rfl⟩ := hf
  have hform (u : ι → F) (v : κ → F) :
      flatEval F ι κ p (u, v) = (u - u₀) ⬝ᵥ p.2 *ᵥ (v - v₀) := by
    have hmix := flatEval_mixed p u u₀ v v₀
    rw [hu₀ v, hv₀ u, hu₀ v₀] at hmix
    simpa using hmix
  have hM : p.2 ≠ 0 := fun hM => h0 (funext fun ⟨u, v⟩ => by rw [hform, hM]; simp)
  obtain ⟨i₀, j₀, hij⟩ : ∃ i j, p.2 i j ≠ 0 := by
    by_contra hall
    push Not at hall
    exact hM (Matrix.ext hall)
  have : Nonempty ι := ⟨i₀⟩
  have : Nonempty κ := ⟨j₀⟩
  obtain ⟨k, hk⟩ : ∃ k, Fintype.card ι = k + 1 :=
    ⟨Fintype.card ι - 1, by have := Fintype.card_pos (α := ι); omega⟩
  obtain ⟨l, hl⟩ : ∃ l, Fintype.card κ = l + 1 :=
    ⟨Fintype.card κ - 1, by have := Fintype.card_pos (α := κ); omega⟩
  have hq : 2 ≤ Fintype.card F := Fintype.one_lt_card
  have hnum := min_pow_succ_le_mul (q := Fintype.card F) (k := k) (l := l) hq
    fun ⟨h2, h0k, h0l⟩ => hne ⟨h2, by omega, by omega⟩
  have hw := mul_le_hammingNorm_form hij u₀ v₀ hk hl
  have hfun : (fun c : (ι → F) × (κ → F) => (c.1 - u₀) ⬝ᵥ p.2 *ᵥ (c.2 - v₀)) =
      flatEval F ι κ p := funext fun ⟨u, v⟩ => (hform u v).symm
  rw [hfun] at hw
  rw [hk, hl]
  exact hnum.trans hw

example : min (Fintype.card (ZMod 3) ^ Fintype.card (Fin 1))
    (Fintype.card (ZMod 3) ^ Fintype.card (Fin 1)) ≤ hammingNorm
      (fun c : (Fin 1 → ZMod 3) × (Fin 1 → ZMod 3) => if c.1 = 0 then (1 : ZMod 3) else 0) :=
  min_le_hammingNorm (by simp [ZMod.card]) (row_indicator_mem 0) fun h => by
    simpa using congrFun h (0, 0)

/-- **`δ(H) = min(q^k, q^l)`** (`review/flat_spectrum.md`, PT4), except at `q = 2`, `k = l = 1`. -/
theorem minDistance_flatClass
    (hne : ¬ (Fintype.card F = 2 ∧ Fintype.card ι = 1 ∧ Fintype.card κ = 1)) :
    minDistance (flatClass F ι κ : Set ((ι → F) × (κ → F) → F)) =
      min (Fintype.card F ^ Fintype.card ι) (Fintype.card F ^ Fintype.card κ) := by
  rw [minDistance_submodule]
  refine IsLeast.csInf_eq ⟨?_, ?_⟩
  · rcases le_total (Fintype.card F ^ Fintype.card ι) (Fintype.card F ^ Fintype.card κ) with h | h
    · rw [min_eq_left h]
      refine ⟨_, col_indicator_mem (0 : κ → F), fun h0 => ?_, hammingNorm_col 0⟩
      simpa using congrFun h0 (0, 0)
    · rw [min_eq_right h]
      refine ⟨_, row_indicator_mem (0 : ι → F), fun h0 => ?_, hammingNorm_row 0⟩
      simpa using congrFun h0 (0, 0)
  · rintro _ ⟨f, hf, h0, rfl⟩
    exact min_le_hammingNorm hne hf h0

example : minDistance (flatClass (ZMod 3) (Fin 1) (Fin 1) :
    Set ((Fin 1 → ZMod 3) × (Fin 1 → ZMod 3) → ZMod 3)) = 3 := by
  rw [minDistance_flatClass (by simp [ZMod.card])]
  simp [ZMod.card]

end Transformer.Grokking.FlatSpectrum.TablePart

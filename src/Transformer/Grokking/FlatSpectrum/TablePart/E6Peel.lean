module

public import Transformer.Grokking.FlatSpectrum.CertificateGram
public import Transformer.Grokking.FlatSpectrum.CertificateTests
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Lines
public import Transformer.Grokking.FlatSpectrum.TablePart.E6PeelMasks

/-!
# PT6: peeling a symmetric table with a tridiagonal inverse

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. Let `K` be symmetric
and invertible, with `K T = c 1` for a tridiagonal `T`, and let it keep its form under the flip
`x ↦ 96 − x`, up to phases. Its certificate `Y` is unique, so `Y` is symmetric and keeps the same
form, and it vanishes on every image of an unseen cell. And `Y T` is Hermitian up to a phase, so an
equation of this relation with one live entry kills that entry. The kernel runs these two rules on
the training half: the live cells go 4704, 588, 42, 2, 0 (`peel_chain`). So `Y = 0`, which is
impossible, and M does not return `K` (`not_returned_of_peel`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

attribute [local instance] fact_prime_97

/-- The neighbours of `x` on the path `0, …, 96`. -/
def nb (x : ℕ) : List ℕ :=
  (if 0 < x then [x - 1] else []) ++ x :: (if x < 96 then [x + 1] else [])

lemma mem_nb {x y : ℕ} (hx : x < 97) : y ∈ nb x ↔ y < 97 ∧ x ≤ y + 1 ∧ y ≤ x + 1 := by
  unfold nb
  split_ifs <;> simp <;> omega

example : 3 ∈ nb 4 := (mem_nb (by norm_num)).2 (by norm_num)

lemma mem_nb_comm {x y : ℕ} (hx : x < 97) (hy : y < 97) : y ∈ nb x ↔ x ∈ nb y := by
  rw [mem_nb hx, mem_nb hy]
  omega

example : 3 ∈ nb 4 ↔ 4 ∈ nb 3 := mem_nb_comm (by norm_num) (by norm_num)

/-- The cell `(a, b)` has to vanish given the live cells `S`: one of its images under the transpose
and the flip is dead, or a relation of `Y T` has `(a, b)` as its only live entry. -/
def forced (S a b : ℕ) : Bool :=
  !S.testBit (97 * b + a) || !S.testBit (97 * (96 - a) + (96 - b)) ||
    !S.testBit (97 * (96 - b) + (96 - a)) ||
    (nb b).any fun j =>
      (nb j).all (fun l => l == b || !S.testBit (97 * a + l)) &&
        (nb a).all fun l => !S.testBit (97 * j + l)

/-- Every cell dropped from `S` to `S'` was dead or is forced. -/
abbrev PeelStep (S S' : ℕ) : Prop :=
  ∀ a, a < 97 → ∀ b, b < 97 → S'.testBit (97 * a + b) = false →
    S.testBit (97 * a + b) = false ∨ forced S a b = true

/-- **The peeling of the training half reaches the empty mask in four steps.** -/
theorem peel_chain : PeelStep halfMask peel1 ∧ PeelStep peel1 peel2 ∧ PeelStep peel2 peel3 ∧
    PeelStep peel3 0 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel, by decide +kernel⟩

/-- `Y` vanishes on the cells outside the mask `S`. -/
def Vanishes (Y : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ) (S : ℕ) : Prop :=
  ∀ a b, a < 97 → b < 97 → S.testBit (97 * a + b) = false → Y (line 97 a) (line 97 b) = 0

lemma line_val (u : Fin 1 → ZMod 97) : line 97 (u 0).val = u := by
  funext t
  rw [Subsingleton.elim t 0]
  simp [line]

/-- One step of the peeling keeps `Y` vanishing. -/
theorem Vanishes.step {Y T : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ} {c : ℂ} (hc : c ≠ 0)
    (hrel : star c • (Y * T) = c • (Y * T)ᴴ) (hsymm : ∀ u v, Y v u = Y u v) {γ δ : ℕ → ℂ}
    (hflip : ∀ a b, a < 97 → b < 97 → Y (line 97 a) (line 97 b) =
      γ a * Y (line 97 (96 - a)) (line 97 (96 - b)) * δ b)
    (hT : ∀ l j, l < 97 → j < 97 → j ∈ nb l → T (line 97 l) (line 97 j) ≠ 0)
    (hT0 : ∀ l j, l < 97 → j < 97 → j ∉ nb l → T (line 97 l) (line 97 j) = 0)
    {S S' : ℕ} (hV : Vanishes Y S) (hstep : PeelStep S S') : Vanishes Y S' := by
  intro a b ha hb hbit
  rcases hstep a ha b hb hbit with h | h
  · exact hV a b ha hb h
  simp only [forced, Bool.or_eq_true, Bool.not_eq_true', List.any_eq_true, Bool.and_eq_true,
    List.all_eq_true, beq_iff_eq] at h
  rcases h with ((h | h) | h) | ⟨j, hj, h1, h2⟩
  · rw [← hsymm (line 97 a) (line 97 b)]
    exact hV b a hb ha h
  · rw [hflip a b ha hb, hV (96 - a) (96 - b) (by omega) (by omega) h]
    simp
  · rw [← hsymm (line 97 a) (line 97 b), hflip b a hb ha,
      hV (96 - b) (96 - a) (by omega) (by omega) h]
    simp
  have hjlt : j < 97 := ((mem_nb hb).1 hj).1
  have hsupp (x y : ℕ) (hx : x < 97) (hy : y < 97) (hne : T (line 97 x) (line 97 y) ≠ 0) :
      x ∈ nb y := by
    refine (mem_nb_comm hx hy).1 ?_
    by_contra hnot
    exact hne (hT0 x y hx hy hnot)
  refine entry_eq_zero_of_peel (T := T) (c := c) (S := {x | Y x.1 x.2 ≠ 0}) hc hrel
    (fun i j' h => by_contra fun h0 => h h0) (j := line 97 j) (hT b j hb hjlt hj) ?_ ?_
  · intro l' hne hl hmem
    have hx := ZMod.val_lt (l' 0)
    have hl' := line_val l'
    have hTne : T (line 97 (l' 0).val) (line 97 j) ≠ 0 := by rwa [hl']
    rcases h1 _ (hsupp _ _ hx hjlt hTne) with hb' | hbit'
    · exact hne (by rw [← hl', hb'])
    · exact hmem (by rw [← hl']; exact hV a _ ha hx hbit')
  · intro l' hl hmem
    have hx := ZMod.val_lt (l' 0)
    have hl' := line_val l'
    have hTne : T (line 97 (l' 0).val) (line 97 a) ≠ 0 := by rwa [hl']
    exact hmem (by rw [← hl']; exact hV j _ hjlt hx (h2 _ (hsupp _ _ hx ha hTne)))

/-- The flip `x ↦ −1 − x` of the lines. -/
def flipLine : Equiv.Perm (Fin 1 → ZMod 97) := Equiv.subLeft fun _ => -1

lemma flipLine_line {a : ℕ} (ha : a < 97) : flipLine (line 97 a) = line 97 (96 - a) := by
  funext t
  have h : ((96 : ℕ) : ZMod 97) = -1 := by decide +kernel
  simp only [flipLine, Equiv.subLeft_apply, Pi.sub_apply, line]
  rw [Nat.cast_sub (by omega), h]

example : flipLine (line 97 1) = line 97 95 := flipLine_line (by norm_num)

/-- **M does not return a symmetric table with a tridiagonal inverse that keeps its form under the
flip**, from the training half. -/
theorem not_returned_of_peel {f : ZMod 97 → ZMod 97 → ZMod 97}
    {T : Matrix (Fin 1 → ZMod 97) (Fin 1 → ZMod 97) ℂ} {c : ℂ}
    (hKT : tableKernel f * T = c • 1) (hc : c ≠ 0)
    (hT : ∀ l j, l < 97 → j < 97 → j ∈ nb l → T (line 97 l) (line 97 j) ≠ 0)
    (hT0 : ∀ l j, l < 97 → j < 97 → j ∉ nb l → T (line 97 l) (line 97 j) = 0)
    (hs : ∀ a b, f b a = f a b) {α β : ZMod 97 → ℂ} (hα : ∀ x, ‖α x‖ = 1) (hβ : ∀ x, ‖β x‖ = 1)
    (hF : ∀ a b, (ZMod.stdAddChar (f (-1 - a) (-1 - b)) : ℂ) =
      α a * ZMod.stdAddChar (f a b) * β b) : NotReturned f := by
  intro hmin
  obtain ⟨Y, hYΩ, hY, htr⟩ := exists_isContraction_of_isMinOn hmin
  set K := tableKernel f with hKdef
  have hdet : K.det ≠ 0 := by
    intro h0
    have h1 := congrArg Matrix.det hKT
    rw [det_mul, h0, zero_mul, det_smul, det_one, mul_one] at h1
    exact pow_ne_zero _ hc h1.symm
  have hKt : Kᵀ = K := by
    ext u v
    simp [hKdef, tableKernel, hs (u 0) (v 0)]
  have htrT : ((Yᵀ)ᴴ * K).trace = nuclearNorm K := by
    have h := trace_transpose_certificate K Y
    rw [hKt] at h
    rw [h, htr]
  have hYT : Y = Yᵀ := eq_of_certificates hdet hY htr hY.transpose htrT
  have hsymm (u v : Fin 1 → ZMod 97) : Y v u = Y u v := by
    have h : Y v u = Yᵀ v u := by rw [← hYT]
    rwa [transpose_apply] at h
  have hsym' (i j : Fin 1 → ZMod 97) :
      K (flipLine i) (flipLine j) = α (i 0) * K i j * β (j 0) := by
    simpa [hKdef, tableKernel, flipLine] using hF (i 0) (j 0)
  have hZ := hY.phased flipLine flipLine (a := fun i => star (α (i 0)))
    (b := fun j => star (β (j 0))) (fun i => by simp [hα]) (fun j => by simp [hβ])
  have hZtr := trace_phased (Y := Y) hsym'
  rw [htr] at hZtr
  have hYZ := eq_of_certificates hdet hY htr hZ hZtr
  have hflip (a b : ℕ) (ha : a < 97) (hb : b < 97) : Y (line 97 a) (line 97 b) =
      star (α (a : ZMod 97)) * Y (line 97 (96 - a)) (line 97 (96 - b)) *
        star (β (b : ZMod 97)) := by
    have h : Y (line 97 a) (line 97 b) = star (α ((line 97 a) 0)) *
        Y (flipLine (line 97 a)) (flipLine (line 97 b)) * star (β ((line 97 b) 0)) := by
      conv_lhs => rw [hYZ]
      rfl
    rw [flipLine_line ha, flipLine_line hb] at h
    exact h
  have hrel := smul_mul_eq_of_certificate hY htr hKT
  have hstep {S S' : ℕ} (hV : Vanishes Y S) (h : PeelStep S S') : Vanishes Y S' :=
    hV.step hc hrel hsymm (γ := fun a => star (α (a : ZMod 97)))
      (δ := fun b => star (β (b : ZMod 97))) hflip hT hT0 h
  have h0 : Vanishes Y halfMask := fun a b ha hb hbit => hYΩ _ _ fun hmem => by
    have h1 := (mem_half_iff ha hb).1 (mem_coe.1 hmem)
    rw [hbit] at h1
    exact Bool.false_ne_true h1
  obtain ⟨c1, c2, c3, c4⟩ := peel_chain
  have hend := hstep (hstep (hstep (hstep h0 c1) c2) c3) c4
  have hY0 : Y = 0 := by
    ext u v
    have h := hend _ _ (ZMod.val_lt (u 0)) (ZMod.val_lt (v 0)) (Nat.zero_testBit _)
    rwa [line_val, line_val] at h
  have hK := mul_gramSqrt_of_certificate hY htr
  rw [hY0, Matrix.zero_mul] at hK
  have hne := tableKernel_ne_zero f 0 0
  rw [← hKdef, ← hK] at hne
  exact hne rfl

end Transformer.Grokking.FlatSpectrum.TablePart.E6

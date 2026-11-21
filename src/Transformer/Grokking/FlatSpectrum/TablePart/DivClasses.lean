module

public import Transformer.Grokking.FlatSpectrum.CertificateClasses
public import Transformer.Grokking.FlatSpectrum.TablePart.Cyclotomic
public import Transformer.Grokking.FlatSpectrum.TablePart.DivFlat

/-!
# PT6: the minor of `(a / b)^k` on the classes of columns

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. When `k` divides
`p − 1`, the column `b` of the kernel of `(a / b)^k` depends only on `b^k`, which takes
`R = (p − 1) / k` values `h^c`, `h = g^k`. On the rows `g^c` the kernel is the circulant
`ψ(h^(c − c'))` over `ℤ/R`. It is invertible when its image in `ZMod q` has a circulant inverse, a
vector of `R` residues that the kernel of Lean multiplies out (`CircCheck`, `det_circ_ne_zero`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix Polynomial

/-- A sum over `ZMod R` is a sum over the residues. -/
lemma sum_zmod_eq_sum_range {M : Type*} [AddCommMonoid M] (R : ℕ) [NeZero R] (G : ℕ → M) :
    ∑ s : ZMod R, G s.val = ∑ n ∈ range R, G n := by
  refine Finset.sum_nbij' (fun x => x.val) (fun n => (n : ZMod R)) ?_ ?_ ?_ ?_ ?_
  · exact fun x _ => mem_range.2 (ZMod.val_lt x)
  · exact fun n _ => mem_univ _
  · exact fun x _ => ZMod.natCast_zmod_val x
  · exact fun n hn => ZMod.val_natCast_of_lt (mem_range.1 hn)
  · exact fun x _ => rfl

/-- The residue of a difference. -/
lemma val_sub_eq {R : ℕ} [NeZero R] (d s : ZMod R) : (d - s).val = (d.val + R - s.val) % R := by
  have hs : s.val ≤ d.val + R := (ZMod.val_lt s).le.trans (Nat.le_add_left _ _)
  have h : ((d.val + R - s.val : ℕ) : ZMod R) = d - s := by
    rw [Nat.cast_sub hs, Nat.cast_add, ZMod.natCast_zmod_val, ZMod.natCast_zmod_val,
      ZMod.natCast_self, add_zero]
  rw [← h, ZMod.val_natCast]

/-- `∑_s z^(e s) W(d − s) ≡ δ(d, 0)` mod `q` on `ℤ/R`. -/
abbrev CircCheck (R q z : ℕ) (e : ℕ → ℕ) (W : List ℕ) : Prop :=
  ∀ d, d < R →
    (∑ s ∈ range R, z ^ e s * W.getD ((d + R - s) % R) 0) % q = if d = 0 then 1 else 0

/-- The check over `ℕ` is the identity in `ZMod q`. -/
lemma circ_of_check {R q z : ℕ} [NeZero R] {e : ℕ → ℕ} {W : List ℕ} (h : CircCheck R q z e W)
    (d : ZMod R) :
    ∑ s : ZMod R, (z : ZMod q) ^ e s.val * (W.getD (d - s).val 0 : ZMod q) =
      if d = 0 then 1 else 0 := by
  have h1 := congrArg (Nat.cast : ℕ → ZMod q) (h d.val (ZMod.val_lt d))
  rw [ZMod.natCast_mod] at h1
  push_cast at h1
  have hsum : ∑ s : ZMod R, (z : ZMod q) ^ e s.val * (W.getD (d - s).val 0 : ZMod q) =
      ∑ s ∈ range R, (z : ZMod q) ^ e s * (W.getD ((d.val + R - s) % R) 0 : ZMod q) := by
    rw [← sum_zmod_eq_sum_range R fun s => (z : ZMod q) ^ e s *
      (W.getD ((d.val + R - s) % R) 0 : ZMod q)]
    exact Finset.sum_congr rfl fun s _ => by rw [val_sub_eq]
  rw [hsum]
  have h2 : ∑ s ∈ range R, (z : ZMod q) ^ e s * (W.getD ((d.val + R - s) % R) 0 : ZMod q) =
      ((if d.val = 0 then 1 else 0 : ℕ) : ZMod q) := by
    simpa using h1
  rw [h2]
  by_cases hd : d = 0
  · simp [hd]
  · have hv : d.val ≠ 0 := fun h0 => hd ((ZMod.val_eq_zero d).1 h0)
    simp [hd, hv]

example (d : ZMod 1) : ∑ s : ZMod 1, ((1 : ℕ) : ZMod 2) ^ (fun _ => 0) s.val *
    (([1] : List ℕ).getD (d - s).val 0 : ZMod 2) = if d = 0 then 1 else 0 :=
  circ_of_check (R := 1) (q := 2) (z := 1) (e := fun _ => 0) (W := [1]) (by decide) d

/-- **A circulant of roots of unity is invertible** when its image in `ZMod q` has a circulant
right inverse. -/
theorem det_circ_ne_zero {p q R z : ℕ} [Fact p.Prime] [Fact q.Prime] [NeZero R] {e : ℕ → ℕ}
    {W : List ℕ} (hz : (z : ZMod q) ≠ 1) (hzp : (z : ZMod q) ^ p = 1) (h : CircCheck R q z e W) :
    (Matrix.of fun c c' : ZMod R =>
      (ZMod.stdAddChar ((e (c - c').val : ℕ) : ZMod p) : ℂ)).det ≠ 0 := by
  set A : Matrix (ZMod R) (ZMod R) ℤ[X] := Matrix.of fun c c' => X ^ e (c - c').val with hA
  have hAζ : A.map (aeval (ZMod.stdAddChar (1 : ZMod p) : ℂ)) = Matrix.of fun c c' =>
      (ZMod.stdAddChar ((e (c - c').val : ℕ) : ZMod p) : ℂ) := by
    ext c c'
    simp [hA, stdAddChar_natCast]
  rw [← hAζ]
  refine det_aeval_ne_zero (q := q) (z := (z : ZMod q)) A hz hzp
    (Matrix.of fun c' c'' => (W.getD (c' - c'').val 0 : ZMod q)) ?_
  ext c c''
  rw [Matrix.mul_apply]
  simp only [hA, map_apply, of_apply, eval₂_pow, eval₂_X]
  have h1 := circ_of_check (q := q) h (c - c'')
  rw [← Equiv.sum_comp (Equiv.subLeft c)] at h1
  simp only [Equiv.subLeft_apply] at h1
  have h2 (x : ZMod R) : c - c'' - (c - x) = x - c'' := by ring
  simp only [h2] at h1
  rw [h1]
  by_cases hcc : c = c''
  · subst hcc
    simp
  · rw [one_apply_ne hcc, ite_eq_right]
    exact fun h0 => hcc (sub_eq_zero.1 h0)

end Transformer.Grokking.FlatSpectrum.TablePart.E6

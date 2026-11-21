module

public import Transformer.Grokking.FlatSpectrum.TablePart.DivClasses

/-!
# PT6: M against the tables `(a / b)^k` with classes of columns

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. The map
`(a, b) ↦ (t a, t b)` keeps the kernel of `(a / b)^k`, and it permutes the classes of columns. So
the cover of `not_isMinOn_of_classes` reduces to two finite checks: one class is unseen in the row
`a = 0`, and for each class `c` some row `t ≠ 0` has the cells `(t, t b)`, `b` in `c`, all unseen.
`ClassCheck` collects them with the inverse of the minor, and `not_returned_classes` is the result.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Finset Matrix

variable {p : ℕ} [Fact p.Prime]

/-- Multiplication of a line by a unit. -/
def scaleRow (t : (ZMod p)ˣ) : Equiv.Perm (Fin 1 → ZMod p) where
  toFun u x := t * u x
  invFun u x := (t⁻¹ : (ZMod p)ˣ) * u x
  left_inv u := funext fun x => Units.inv_mul_cancel_left t (u x)
  right_inv u := funext fun x => Units.mul_inv_cancel_left t (u x)

/-- Multiplication of a nonzero column by a unit. -/
def scaleCol (t : (ZMod p)ˣ) : Equiv.Perm {v : Fin 1 → ZMod p // v ≠ 0} :=
  (scaleRow t).subtypeEquiv fun v => by
    constructor
    · intro hv h0
      refine hv (funext fun x => ?_)
      have h : (t : ZMod p) * v x = 0 := congrFun h0 x
      exact (mul_eq_zero.1 h).resolve_left t.ne_zero
    · intro hv h0
      refine hv (funext fun x => ?_)
      change (t : ZMod p) * v x = 0
      rw [h0]
      exact mul_zero _

/-- A nonzero column is nonzero at `0`. -/
lemma column_ne_zero (v : {v : Fin 1 → ZMod p // v ≠ 0}) : v.1 0 ≠ 0 := fun h =>
  v.2 (funext fun x => by rw [Subsingleton.elim x 0, h]; rfl)

/-- Equality of powers in `ZMod p` as a check over `ℕ`. -/
lemma pow_val_mod_eq {y : ZMod p} {k c g : ℕ} (hy : y ^ k = ((g : ZMod p) ^ k) ^ c) :
    y.val ^ k % p = (g ^ k) ^ c % p := by
  have h : ((y.val ^ k : ℕ) : ZMod p) = (((g ^ k) ^ c : ℕ) : ZMod p) := by
    push_cast
    rw [ZMod.natCast_zmod_val, hy]
  exact (ZMod.natCast_eq_natCast_iff' _ _ _).1 h

example : (2 : ZMod 3).val ^ 2 % 3 = (2 ^ 2) ^ 1 % 3 :=
  pow_val_mod_eq (g := 2) (by decide)

/-- A clear bit leaves the cell `(x, y)` of the kernel unseen. -/
lemma cell_notMem {T : ℕ} {x y : ZMod p} (hy : y ≠ 0)
    (h : T.testBit ((p - 1) * x.val + (y.val - 1)) = false) {u : Fin 1 → ZMod p}
    {v : {v : Fin 1 → ZMod p // v ≠ 0}} (hu : u 0 = x) (hv : v.1 0 = y) :
    (u, v) ∉ divSeen p T := by
  have hp : 1 < p := (Fact.out : p.Prime).one_lt
  have hyv : 0 < y.val := Nat.pos_of_ne_zero fun h0 => hy ((ZMod.val_eq_zero y).1 h0)
  have hlt := ZMod.val_lt y
  have h1 := notMem_divPart (p := p) (b := y.val - 1) hp (ZMod.val_lt x) (by omega) h
  intro hmem
  apply h1
  have hu' : u = line p x.val := by
    funext t
    rw [Subsingleton.elim t 0, hu]
    simp [line]
  have hv' : v.1 = line p (y.val - 1 + 1) := by
    funext t
    rw [Subsingleton.elim t 0, hv, Nat.sub_add_cancel hyv]
    simp [line]
  rw [← hu', ← hv']
  exact hmem

example : ((fun _ => 0, ⟨fun _ => 1, fun h => one_ne_zero (congrFun h 0)⟩) :
    (Fin 1 → ZMod 3) × {v : Fin 1 → ZMod 3 // v ≠ 0}) ∉ divSeen 3 0 :=
  cell_notMem (x := 0) (y := 1) one_ne_zero (by decide) rfl rfl

/-- What the kernel checks for `(a / b)^k` with `R` classes: `g` is nonzero, `h = g^k` has
`h^R = 1`, every `b^k` is a power of `h`, the minor has an inverse in `ZMod q`, one class is unseen
in the row `0`, and for each class some row `t ≠ 0` has its cells `(t, t b)` unseen. -/
abbrev ClassCheck (p k R g q z T : ℕ) (W : List ℕ) : Prop :=
  g % p ≠ 0 ∧ (g ^ k) ^ R % p = 1 ∧
    (∀ b, b < p → 0 < b → ∃ c, c < R ∧ b ^ k % p = (g ^ k) ^ c % p) ∧
    CircCheck R q z (fun s => (g ^ k) ^ s % p) W ∧
    (∃ c, c < R ∧ ∀ b, b < p → 0 < b → b ^ k % p = (g ^ k) ^ c % p →
      T.testBit ((p - 1) * 0 + (b - 1)) = false) ∧
    ∀ c, c < R → ∃ t, t < p ∧ 0 < t ∧ ∀ b, b < p → 0 < b → b ^ k % p = (g ^ k) ^ c % p →
      T.testBit ((p - 1) * t + (t * b % p - 1)) = false

/-- **M does not return `(a / b)^k` from a part that passes `ClassCheck`.** -/
theorem not_returned_classes {k R g q z T : ℕ} [Fact q.Prime] [NeZero R] {W : List ℕ}
    (hz : (z : ZMod q) ≠ 1) (hzp : (z : ZMod q) ^ p = 1) (hcheck : ClassCheck p k R g q z T W) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ divSeen p T, X ij.1 ij.2 = divKernel p k ij.1 ij.2}
      (divKernel p k) := by
  obtain ⟨hg0, hHR, hcl, hcirc, ⟨c₀, hc₀, h0⟩, hcov⟩ := hcheck
  have hg : (g : ZMod p) ≠ 0 := fun h => hg0 (Nat.mod_eq_zero_of_dvd
    ((ZMod.natCast_eq_zero_iff g p).1 h))
  set H : ZMod p := (g : ZMod p) ^ k with hH
  have hH0 : H ≠ 0 := pow_ne_zero _ hg
  have hHRz : H ^ R = 1 := by
    have h := congrArg (Nat.cast : ℕ → ZMod p) hHR
    rw [ZMod.natCast_mod] at h
    simpa [hH] using h
  have hsub (c c' : ZMod R) : H ^ (c - c').val * H ^ c'.val = H ^ c.val := by
    rw [← pow_add, pow_eq_pow_mod _ hHRz, ← ZMod.val_add, sub_add_cancel]
  have hex (y : ZMod p) (hy : y ≠ 0) : ∃ c : ZMod R, y ^ k = H ^ c.val := by
    obtain ⟨c, hc, hce⟩ := hcl y.val (ZMod.val_lt y)
      (Nat.pos_of_ne_zero fun h => hy ((ZMod.val_eq_zero y).1 h))
    refine ⟨(c : ZMod R), ?_⟩
    rw [ZMod.val_natCast_of_lt hc]
    have h := (ZMod.natCast_eq_natCast_iff' _ _ p).2 hce
    push_cast at h
    rwa [ZMod.natCast_zmod_val] at h
  choose cl hcl' using fun v : {v : Fin 1 → ZMod p // v ≠ 0} => hex (v.1 0) (column_ne_zero v)
  set L : Matrix (Fin 1 → ZMod p) (ZMod R) ℂ :=
    Matrix.of fun u c => (ZMod.stdAddChar (u 0 ^ k / H ^ c.val) : ℂ) with hL
  have hKL (u : Fin 1 → ZMod p) (v : {v : Fin 1 → ZMod p // v ≠ 0}) :
      divKernel p k u v = L u (cl v) := by
    simp only [divKernel, divTable, hL, of_apply, div_pow, hcl']
  have hdet : (L.submatrix (fun c : ZMod R => fun _ => (g : ZMod p) ^ c.val) id).det ≠ 0 := by
    have h := det_circ_ne_zero (p := p) hz hzp hcirc
    have hm : L.submatrix (fun c : ZMod R => fun _ => (g : ZMod p) ^ c.val) id =
        Matrix.of fun c c' : ZMod R =>
          (ZMod.stdAddChar ((((g ^ k) ^ (c - c').val % p : ℕ)) : ZMod p) : ℂ) := by
      ext c c'
      simp only [submatrix_apply, id, hL, of_apply]
      congr 1
      have hgc : ((g : ZMod p) ^ c.val) ^ k = H ^ c.val := by rw [hH, pow_right_comm]
      rw [hgc, ZMod.natCast_mod]
      push_cast
      exact (eq_div_of_mul_eq (pow_ne_zero _ hH0) (hsub c c')).symm
    rw [hm]
    exact h
  have hK0 : divKernel p k ≠ 0 := by
    intro hzero
    have hn : divKernel p k 0 ⟨fun _ => 1, fun h => one_ne_zero (congrFun h 0)⟩ ≠ 0 :=
      norm_ne_zero_iff.mp (by simp [divKernel])
    exact hn (by rw [hzero]; rfl)
  refine not_isMinOn_of_classes hK0 cl L hKL _ hdet scaleRow scaleCol (fun _ _ => 1)
    (fun _ _ => 1) (by simp) (by simp) (fun t u v => ?_) fun u c => ?_
  · have h1 : scaleRow t u 0 / (scaleCol t v).1 0 = u 0 / v.1 0 := by
      change (t : ZMod p) * u 0 / ((t : ZMod p) * v.1 0) = u 0 / v.1 0
      exact mul_div_mul_left _ _ t.ne_zero
    simp only [divKernel, divTable, of_apply, h1, one_mul, mul_one]
  · by_cases hu : u 0 = 0
    · have hne : (g : ZMod p) ^ ((c₀ + R - c.val) % R) ≠ 0 := pow_ne_zero _ hg
      refine ⟨Units.mk0 _ hne, fun j hj => ?_⟩
      set y : ZMod p := (g : ZMod p) ^ ((c₀ + R - c.val) % R) * j.1 0 with hy
      have hy0 : y ≠ 0 := mul_ne_zero hne (column_ne_zero j)
      have hle : c.val ≤ c₀ + R := (ZMod.val_lt c).le.trans (Nat.le_add_left _ _)
      have hgk : ((g : ZMod p) ^ ((c₀ + R - c.val) % R)) ^ k = H ^ ((c₀ + R - c.val) % R) := by
        rw [hH, pow_right_comm]
      have hyk : y ^ k = H ^ c₀ := by
        rw [hy, mul_pow, hgk, hcl', hj, ← pow_add, pow_eq_pow_mod _ hHRz, Nat.mod_add_mod,
          Nat.sub_add_cancel hle, Nat.add_mod_right, Nat.mod_eq_of_lt hc₀]
      have hbit := h0 y.val (ZMod.val_lt y)
        (Nat.pos_of_ne_zero fun h => hy0 ((ZMod.val_eq_zero y).1 h)) (pow_val_mod_eq hyk)
      refine cell_notMem (x := 0) hy0 (by simpa using hbit) ?_ rfl
      change (g : ZMod p) ^ ((c₀ + R - c.val) % R) * u 0 = 0
      rw [hu, mul_zero]
    · obtain ⟨cu, hcu⟩ := hex (u 0) hu
      obtain ⟨t', ht', ht0, hbits⟩ := hcov (c - cu).val (ZMod.val_lt _)
      have ht'z : (t' : ZMod p) ≠ 0 := fun h => by
        have hv := congrArg ZMod.val h
        rw [ZMod.val_natCast_of_lt ht', ZMod.val_zero] at hv
        omega
      refine ⟨Units.mk0 ((t' : ZMod p) / u 0) (div_ne_zero ht'z hu), fun j hj => ?_⟩
      set b' : ZMod p := j.1 0 / u 0 with hb'
      have hb'0 : b' ≠ 0 := div_ne_zero (column_ne_zero j) hu
      have hb'k : b' ^ k = H ^ (c - cu).val := by
        rw [hb', div_pow, hcl', hj, hcu]
        exact div_eq_of_eq_mul (pow_ne_zero _ hH0) (hsub c cu).symm
      have hbit := hbits b'.val (ZMod.val_lt b')
        (Nat.pos_of_ne_zero fun h => hb'0 ((ZMod.val_eq_zero b').1 h)) (pow_val_mod_eq hb'k)
      have hy : (t' : ZMod p) * b' ≠ 0 := mul_ne_zero ht'z hb'0
      have hval : ((t' : ZMod p) * b').val = t' * b'.val % p := by
        rw [ZMod.val_mul, ZMod.val_natCast_of_lt ht']
      refine cell_notMem (x := (t' : ZMod p)) hy ?_ ?_ ?_
      · rw [ZMod.val_natCast_of_lt ht', hval]
        exact hbit
      · change (t' : ZMod p) / u 0 * u 0 = t'
        exact div_mul_cancel₀ _ hu
      · change (t' : ZMod p) / u 0 * j.1 0 = (t' : ZMod p) * (j.1 0 / u 0)
        ring

end Transformer.Grokking.FlatSpectrum.TablePart.E6

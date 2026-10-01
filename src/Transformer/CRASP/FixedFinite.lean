/-
# Finite values and exact rounding cells

arXiv:2506.16055v3, Appendix B.1, `def:fixed_precision`, and B.2's
finite-state simulation. The two boundary cells include saturation;
interior cells are the usual half-open grid intervals.
-/

import Transformer.CRASP.Fixed

namespace Transformer.CRASP.Fx

variable {p s : ℕ}

/-- The fixed-precision state space is finite (Appendix B.1). -/
instance : Finite (Fx p s) := Finite.of_injective
  (fun x : Fx p s => (⟨x.m, Finset.mem_Ico.mpr ⟨x.lo, x.hi⟩⟩ :
    {m : ℤ // m ∈ Finset.Ico (-2 ^ (p - 1)) (2 ^ (p - 1))}))
  (fun x y h => Fx.ext (x := x) (y := y) (congrArg Subtype.val h))

/-- Enumeration of all representable values (Appendix B.2). -/
noncomputable instance : Fintype (Fx p s) := Fintype.ofFinite _

/-- Equality of fixed-point values is equality of their integer mantissas (B.1). -/
instance : DecidableEq (Fx p s) := fun x y =>
  decidable_of_iff (x.m = y.m) ⟨Fx.ext, congrArg Fx.m⟩

/-- Clamping selects a mantissa exactly in its two endpoint-aware intervals.
Source: arXiv:2506.16055v3, Appendix B.1, saturating extension of rounding. -/
theorem clamp_eq_iff (y : Fx p s) (z : ℤ) :
    clamp p z = y.m ↔
      (-2 ^ (p - 1) < y.m → y.m ≤ z) ∧
      (y.m < 2 ^ (p - 1) - 1 → z < y.m + 1) := by
  have hlo := y.lo
  have hhi := y.hi
  unfold clamp
  omega

/-- Exact cells of the rounded value, including the two saturated endpoints.
Source: arXiv:2506.16055v3, Appendix B.1, `def:fixed_precision`. -/
theorem round_eq_iff (y : Fx p s) (x : ℝ) :
    round p s x = y ↔
      (-2 ^ (p - 1) < y.m → (y.m : ℝ) ≤ x * 2 ^ s) ∧
      (y.m < 2 ^ (p - 1) - 1 → x * 2 ^ s < (y.m + 1 : ℤ)) := by
  rw [show round p s x = y ↔ (round p s x).m = y.m from
    ⟨fun h => congrArg Fx.m h, Fx.ext⟩, m_round, clamp_eq_iff]
  simp only [Int.le_floor, Int.floor_lt]

/-- Rounding a nonnegative real has a nonnegative mantissa (Appendix B.1). -/
theorem m_round_nonneg (x : ℝ) (hx : 0 ≤ x) : 0 ≤ (round p s x).m := by
  have hf : 0 ≤ ⌊x * 2 ^ s⌋ := Int.floor_nonneg.mpr (by positivity)
  have hp : (0 : ℤ) < 2 ^ (p - 1) := by positivity
  simp only [m_round, clamp]
  omega

/-- The nonnegative rounding hypothesis has a witness (Appendix B.1). -/
example : (0 : ℝ) ≤ 1 := zero_le_one

end Transformer.CRASP.Fx

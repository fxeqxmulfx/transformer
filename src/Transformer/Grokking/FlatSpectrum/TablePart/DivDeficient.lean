module

public import Transformer.Grokking.FlatSpectrum.TablePart.DivCover
public import Transformer.Grokking.FlatSpectrum.TablePart.E6Sums

/-!
# PT6: M misses the five tables `(a / b)^k` with parallel columns

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. When `k` divides
`p − 1`, the columns of `(a / b)^k` fall into `R = (p − 1) / k` classes, so the kernel has rank at
most `R`. Each list below is the inverse of its circulant minor in `ZMod q`, with `q ≡ 1 mod p`. The
kernel of Lean checks the inverse and the cover of the classes on the real training mask.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

attribute [local instance] fact_prime_97 fact_prime_17 fact_prime_193

/-- `103 = 6 · 17 + 1` is prime. -/
local instance fact_prime_103 : Fact (Nat.Prime 103) := ⟨by norm_num⟩

/-- `773 = 4 · 193 + 1` is prime. -/
local instance fact_prime_773 : Fact (Nat.Prime 773) := ⟨by norm_num⟩

/-- The inverse of the minor of `(a / b)^2` at `p = 17`, in `ZMod 103`. -/
def wDiv17k2 : List ℕ := [
  39, 33, 67, 56, 29, 94, 42, 100]

/-- The inverse of the minor of `(a / b)^2` at `p = 97`, in `ZMod 389`. -/
def wDiv97k2 : List ℕ := [
  88, 42, 23, 146, 291, 30, 208, 280, 32, 129, 137, 195, 166, 2, 138, 7, 35, 177, 2, 221, 70, 282,
  12, 216, 263, 375, 316, 108, 18, 338, 225, 216, 364, 326, 280, 303, 86, 273, 133, 10, 345, 339,
  289, 82, 366, 274, 36, 121]

/-- The inverse of the minor of `(a / b)^3` at `p = 97`, in `ZMod 389`. -/
def wDiv97k3 : List ℕ := [
  199, 179, 68, 263, 152, 211, 335, 286, 333, 383, 122, 42, 296, 191, 330, 43, 386, 224, 71, 130,
  348, 70, 280, 75, 176, 67, 97, 187, 372, 310, 284, 213]

/-- The inverse of the minor of `(a / b)^2` at `p = 193`, in `ZMod 773`. -/
def wDiv193k2 : List ℕ := [
  233, 246, 20, 606, 123, 276, 304, 118, 555, 89, 525, 747, 127, 475, 159, 211, 9, 648, 396, 67,
  96, 505, 21, 735, 225, 758, 253, 639, 235, 387, 562, 246, 681, 528, 654, 473, 221, 48, 154, 90,
  450, 654, 682, 83, 404, 490, 608, 481, 196, 290, 129, 141, 637, 401, 420, 16, 324, 385, 658, 186,
  621, 4, 8, 298, 176, 620, 61, 402, 47, 125, 263, 221, 228, 655, 87, 444, 559, 132, 113, 655, 606,
  712, 549, 494, 368, 571, 745, 152, 625, 108, 375, 655, 351, 191, 495, 270]

/-- The inverse of the minor of `(a / b)^3` at `p = 193`, in `ZMod 773`. -/
def wDiv193k3 : List ℕ := [
  673, 136, 296, 387, 452, 101, 606, 132, 736, 399, 142, 669, 318, 692, 12, 260, 727, 250, 416,
  236, 308, 414, 239, 677, 743, 442, 316, 422, 159, 751, 306, 563, 360, 382, 669, 264, 544, 682,
  525, 439, 428, 272, 278, 335, 335, 528, 56, 761, 507, 47, 346, 214, 418, 387, 54, 538, 159, 707,
  760, 758, 623, 678, 614, 392]

/-- **M does not return `(a / b)^2` at `p = 17` from its training cells.** -/
theorem not_returned_div17_k2 :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ divSeen 17 mask17, X ij.1 ij.2 = divKernel 17 2 ij.1 ij.2}
      (divKernel 17 2) :=
  not_returned_classes (R := 8) (g := 3) (q := 103) (z := 64) (W := wDiv17k2)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

/-- **M does not return `(a / b)^2` at `p = 97` from its training cells.** -/
theorem not_returned_div97_k2 :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ divSeen 97 mask97, X ij.1 ij.2 = divKernel 97 2 ij.1 ij.2}
      (divKernel 97 2) :=
  not_returned_classes (R := 48) (g := 5) (q := 389) (z := 16) (W := wDiv97k2)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

/-- **M does not return `(a / b)^3` at `p = 97` from its training cells.** -/
theorem not_returned_div97_k3 :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ divSeen 97 mask97, X ij.1 ij.2 = divKernel 97 3 ij.1 ij.2}
      (divKernel 97 3) :=
  not_returned_classes (R := 32) (g := 5) (q := 389) (z := 16) (W := wDiv97k3)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

/-- **M does not return `(a / b)^2` at `p = 193` from its training cells.** -/
theorem not_returned_div193_k2 :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ divSeen 193 mask193, X ij.1 ij.2 = divKernel 193 2 ij.1 ij.2}
      (divKernel 193 2) :=
  not_returned_classes (R := 96) (g := 5) (q := 773) (z := 16) (W := wDiv193k2)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

/-- **M does not return `(a / b)^3` at `p = 193` from its training cells.** -/
theorem not_returned_div193_k3 :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ divSeen 193 mask193, X ij.1 ij.2 = divKernel 193 3 ij.1 ij.2}
      (divKernel 193 3) :=
  not_returned_classes (R := 64) (g := 5) (q := 773) (z := 16) (W := wDiv193k3)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

example : (Matrix.of fun c c' : ZMod 8 =>
    (ZMod.stdAddChar (((3 ^ 2) ^ (c - c').val % 17 : ℕ) : ZMod 17) : ℂ)).det ≠ 0 :=
  det_circ_ne_zero (p := 17) (q := 103) (z := 64) (e := fun s => (3 ^ 2) ^ s % 17) (W := wDiv17k2)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

example : ¬ IsMinOn nuclearNorm
    {X | ∀ ij ∈ divSeen 17 mask17, X ij.1 ij.2 = divKernel 17 2 ij.1 ij.2} (divKernel 17 2) :=
  not_returned_classes (R := 8) (g := 3) (q := 103) (z := 64) (W := wDiv17k2)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

end Transformer.Grokking.FlatSpectrum.TablePart.E6

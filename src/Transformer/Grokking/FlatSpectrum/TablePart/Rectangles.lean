module

public import Transformer.Grokking.FlatSpectrum.TablePart.FlatClass

/-!
# PT6: two rectangles that no function of the class passes through

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. On the rows `a, a'`
and the columns `b, b'`, a function `g(a) + h(b) + m a b` has the alternating sum
`m (a − a')(b − b')` (`exists_altSum_eq`). So two rectangles whose labels ask for two different
couplings leave no function of the class (`not_fits`). Such a pair is a short certificate that a
task lies outside the class on its training cells.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart.E6

open Matrix

/-- A rectangle with labels: `(a, a', b, b', y(a, b), y(a', b), y(a', b'), y(a, b'))`. -/
abbrev Rect := ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ

/-- The corners of a rectangle as `(row, column, label)`, in the order of the cycle. -/
def corners (r : Rect) : List (ℕ × ℕ × ℕ) :=
  [(r.1, r.2.2.1, r.2.2.2.2.1), (r.2.1, r.2.2.1, r.2.2.2.2.2.1),
    (r.2.1, r.2.2.2.1, r.2.2.2.2.2.2.1), (r.1, r.2.2.2.1, r.2.2.2.2.2.2.2)]

/-- The alternating sum of the labels around the rectangle. -/
def altSum (r : Rect) : ℤ :=
  (r.2.2.2.2.1 : ℤ) - r.2.2.2.2.2.1 + r.2.2.2.2.2.2.1 - r.2.2.2.2.2.2.2

/-- The product `(a − a')(b − b')`. -/
def gap (r : Rect) : ℤ := ((r.1 : ℤ) - r.2.1) * ((r.2.2.1 : ℤ) - r.2.2.2.1)

/-- Two rectangles whose labels ask for two different couplings modulo `p`. -/
def refutes (p : ℕ) (r s : Rect) : Bool :=
  (altSum r * gap s - altSum s * gap r) % (p : ℤ) != 0

/-- The line with the number `n`. -/
def line (p n : ℕ) : Fin 1 → ZMod p := fun _ => (n : ZMod p)

variable {p : ℕ} [Fact p.Prime]

/-- A member of the class has one coupling `m`, and on every rectangle whose labels it takes, the
alternating sum is `m (a − a')(b − b')`. -/
lemma exists_altSum_eq {f : (Fin 1 → ZMod p) × (Fin 1 → ZMod p) → ZMod p}
    (hf : f ∈ flatClass (ZMod p) (Fin 1) (Fin 1)) :
    ∃ m : ZMod p, ∀ r : Rect, (∀ c ∈ corners r, f (line p c.1, line p c.2.1) = c.2.2) →
      ((altSum r : ℤ) : ZMod p) = m * ((gap r : ℤ) : ZMod p) := by
  obtain ⟨g, h, M, hfg⟩ := mem_flatClass.1 hf
  have key (x y : ℕ) : line p x ⬝ᵥ M *ᵥ line p y = x * M 0 0 * y := by
    simp [line, dotProduct, mulVec]
    ring
  refine ⟨M 0 0, fun r hr => ?_⟩
  obtain ⟨a, a', b, b', y₁, y₂, y₃, y₄⟩ := r
  have h₁ := hr (a, b, y₁) (by simp [corners])
  have h₂ := hr (a', b, y₂) (by simp [corners])
  have h₃ := hr (a', b', y₃) (by simp [corners])
  have h₄ := hr (a, b', y₄) (by simp [corners])
  simp only [hfg, key] at h₁ h₂ h₃ h₄
  simp only [altSum, gap]
  push_cast
  rw [← h₁, ← h₂, ← h₃, ← h₄]
  ring

example : ∃ m : ZMod 3, ∀ r : Rect, (∀ c ∈ corners r,
    (0 : (Fin 1 → ZMod 3) × (Fin 1 → ZMod 3) → ZMod 3) (line 3 c.1, line 3 c.2.1) = c.2.2) →
      ((altSum r : ℤ) : ZMod 3) = m * ((gap r : ℤ) : ZMod 3) :=
  exists_altSum_eq (Submodule.zero_mem _)

/-- **Two rectangles that ask for two couplings leave no function of the class**
(`review/flat_spectrum.md`, PT6). -/
theorem not_fits {r s : Rect} (h : refutes p r s = true) :
    ¬ ∃ f ∈ flatClass (ZMod p) (Fin 1) (Fin 1),
      (∀ c ∈ corners r, f (line p c.1, line p c.2.1) = c.2.2) ∧
        ∀ c ∈ corners s, f (line p c.1, line p c.2.1) = c.2.2 := by
  rintro ⟨f, hf, hr, hs⟩
  obtain ⟨m, hm⟩ := exists_altSum_eq hf
  have hz : ((altSum r * gap s - altSum s * gap r : ℤ) : ZMod p) = 0 := by
    push_cast
    rw [hm r hr, hm s hs]
    ring
  have h0 := Int.emod_eq_zero_of_dvd ((ZMod.intCast_zmod_eq_zero_iff_dvd _ p).1 hz)
  simp [refutes, h0] at h

example : ¬ ∃ f ∈ flatClass (ZMod 3) (Fin 1) (Fin 1),
    (∀ c ∈ corners (0, 1, 0, 1, 0, 0, 0, 1), f (line 3 c.1, line 3 c.2.1) = c.2.2) ∧
      ∀ c ∈ corners (0, 1, 0, 2, 0, 0, 0, 0), f (line 3 c.1, line 3 c.2.1) = c.2.2 :=
  not_fits (by decide)

end Transformer.Grokking.FlatSpectrum.TablePart.E6

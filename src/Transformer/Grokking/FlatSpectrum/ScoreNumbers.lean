module

public import Transformer.Grokking.FlatSpectrum.ScoreShift
public import Mathlib.Analysis.Convex.Hull

/-!
# Scores that are linear in given numbers: a cell of a hull is not open

Source: `review/flat_spectrum.md`, *The numbers of the tokens* (added), item NT4.
In CS3 to CS5 the score of every row and column is free, and a cell is open unless a chain of
seen cells links its two tokens. Give every row a vector `φ i` and every column a vector `ψ j`
of numbers, and let the scores of the learner be linear in them: `f' i = u (φ i)` and
`h' j = θ − w (ψ j)`, with `u`, `w` and `θ` free. Its answer `h' j ≤ f' i` is then the
half-space `θ ≤ u (φ i) + w (ψ j)` of the pair `(φ i, ψ j)`, and a half-space that holds the
seen cells of one answer holds their convex hull. So scores that fit the seen cells answer as
those cells do at every cell whose pair is in the hull of the pairs of the seen cells with one
answer (`ScoreFit.le_of_mem_convexHull`, `ScoreFit.lt_of_mem_convexHull`). When the scores
themselves are linear in the numbers, such a cell is answered rightly by every linear fit
(`scoreFit_linear_iff`): it is not open, whether or not a chain links its tokens.

The scores of CS1 are of this kind: `2 d₁` in the number `d₁` of the first token, and
`p − 3 d₂ − 2p [d₂ < 0]` in the number `d₂` of the second with the indicator of its half-line.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

variable {m n E F : Type*} [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]

/-- The pairs of numbers of the seen cells with `h j ≤ f i`. -/
def seenLePairs (Ω : Set (m × n)) (f : m → ℝ) (h : n → ℝ) (φ : m → E) (ψ : n → F) :
    Set (E × F) :=
  (fun c : m × n => (φ c.1, ψ c.2)) '' {c | c ∈ Ω ∧ h c.2 ≤ f c.1}

/-- The pairs of numbers of the seen cells with `f i < h j`. -/
def seenLtPairs (Ω : Set (m × n)) (f : m → ℝ) (h : n → ℝ) (φ : m → E) (ψ : n → F) :
    Set (E × F) :=
  (fun c : m × n => (φ c.1, ψ c.2)) '' {c | c ∈ Ω ∧ f c.1 < h c.2}

/-- **NT4**, `review/flat_spectrum.md`: scores linear in the numbers that fit the seen cells
have the column at most the row at every cell in the hull of the seen cells with that
answer. -/
theorem ScoreFit.le_of_mem_convexHull {Ω : Set (m × n)} {f : m → ℝ} {h : n → ℝ} {φ : m → E}
    {ψ : n → F} {u : E →ₗ[ℝ] ℝ} {w : F →ₗ[ℝ] ℝ} {θ : ℝ}
    (hfit : ScoreFit Ω f h (fun i => u (φ i)) fun j => θ - w (ψ j)) {i : m} {j : n}
    (hmem : (φ i, ψ j) ∈ convexHull ℝ (seenLePairs Ω f h φ ψ)) : θ - w (ψ j) ≤ u (φ i) := by
  have hsub : seenLePairs Ω f h φ ψ ⊆ {x : E × F | θ ≤ u.coprod w x} := by
    rintro _ ⟨c, ⟨hc, hle⟩, rfl⟩
    have := (hfit c hc).mpr hle
    simp only [Set.mem_ofPred_eq, LinearMap.coprod_apply]
    linarith
  have := convexHull_min hsub (convex_halfSpace_ge (u.coprod w).isLinear θ) hmem
  simp only [Set.mem_ofPred_eq, LinearMap.coprod_apply] at this
  linarith

/-- **NT4**, `review/flat_spectrum.md`: and the row below the column at every cell in the hull
of the seen cells with that answer. -/
theorem ScoreFit.lt_of_mem_convexHull {Ω : Set (m × n)} {f : m → ℝ} {h : n → ℝ} {φ : m → E}
    {ψ : n → F} {u : E →ₗ[ℝ] ℝ} {w : F →ₗ[ℝ] ℝ} {θ : ℝ}
    (hfit : ScoreFit Ω f h (fun i => u (φ i)) fun j => θ - w (ψ j)) {i : m} {j : n}
    (hmem : (φ i, ψ j) ∈ convexHull ℝ (seenLtPairs Ω f h φ ψ)) : u (φ i) < θ - w (ψ j) := by
  have hsub : seenLtPairs Ω f h φ ψ ⊆ {x : E × F | u.coprod w x < θ} := by
    rintro _ ⟨c, ⟨hc, hlt⟩, rfl⟩
    have := mt (hfit c hc).mp (not_le.mpr hlt)
    simp only [Set.mem_ofPred_eq, LinearMap.coprod_apply]
    linarith
  have := convexHull_min hsub (convex_halfSpace_lt (u.coprod w).isLinear θ) hmem
  simp only [Set.mem_ofPred_eq, LinearMap.coprod_apply] at this
  linarith

/-- One seen cell with the column score `−1` below the row score `0`: it is in the hull of the
seen cells with its answer. -/
example : (-1 : ℝ) - (0 : ℝ →ₗ[ℝ] ℝ) 0 ≤ (LinearMap.id : ℝ →ₗ[ℝ] ℝ) 0 :=
  ScoreFit.le_of_mem_convexHull (Ω := {((), ())}) (f := fun _ => 0) (h := fun _ => -1)
    (φ := fun _ => (0 : ℝ)) (ψ := fun _ => (0 : ℝ)) (i := ()) (j := ())
    (by rintro c rfl; simp) (subset_convexHull ℝ _ ⟨((), ()), ⟨rfl, by norm_num⟩, rfl⟩)

/-- One seen cell with the row score `0` below the column score `1`. -/
example : (LinearMap.id : ℝ →ₗ[ℝ] ℝ) 0 < (1 : ℝ) - (0 : ℝ →ₗ[ℝ] ℝ) 0 :=
  ScoreFit.lt_of_mem_convexHull (Ω := {((), ())}) (f := fun _ => 0) (h := fun _ => 1)
    (φ := fun _ => (0 : ℝ)) (ψ := fun _ => (0 : ℝ)) (i := ()) (j := ())
    (by rintro c rfl; simp) (subset_convexHull ℝ _ ⟨((), ()), ⟨rfl, by norm_num⟩, rfl⟩)

/-- **NT4**, `review/flat_spectrum.md`: when the scores are linear in the numbers, every fit
that is linear in them answers as they do at a cell in the hull of the seen cells of either
answer. -/
theorem scoreFit_linear_iff {Ω : Set (m × n)} {φ : m → E} {ψ : n → F} {u₀ u : E →ₗ[ℝ] ℝ}
    {w₀ w : F →ₗ[ℝ] ℝ} {θ₀ θ : ℝ}
    (hfit : ScoreFit Ω (fun i => u₀ (φ i)) (fun j => θ₀ - w₀ (ψ j)) (fun i => u (φ i))
      fun j => θ - w (ψ j)) {i : m} {j : n}
    (hmem : (φ i, ψ j) ∈
      convexHull ℝ (seenLePairs Ω (fun i => u₀ (φ i)) (fun j => θ₀ - w₀ (ψ j)) φ ψ) ∪
        convexHull ℝ (seenLtPairs Ω (fun i => u₀ (φ i)) (fun j => θ₀ - w₀ (ψ j)) φ ψ)) :
    θ - w (ψ j) ≤ u (φ i) ↔ θ₀ - w₀ (ψ j) ≤ u₀ (φ i) := by
  rcases hmem with hle | hlt
  · exact iff_of_true (hfit.le_of_mem_convexHull hle)
      ((scoreFit_self Ω _ _).le_of_mem_convexHull hle)
  · exact iff_of_false (not_le.mpr (hfit.lt_of_mem_convexHull hlt))
      (not_le.mpr ((scoreFit_self Ω _ _).lt_of_mem_convexHull hlt))

/-- Three rows with the numbers `0`, `1`, `2` and one column with the number `0`. The scores
are the number of the row against `−1`; the rows `0` and `2` are seen. The fit `3 x` against
`−1` is linear, and at the unseen row `1`, the middle of the two seen, it answers as the
scores do. -/
example : (-1 : ℝ) - (0 : ℝ →ₗ[ℝ] ℝ) 0 ≤ (3 • LinearMap.id : ℝ →ₗ[ℝ] ℝ) ((1 : Fin 3) : ℕ) ↔
    (-1 : ℝ) - (0 : ℝ →ₗ[ℝ] ℝ) 0 ≤ (LinearMap.id : ℝ →ₗ[ℝ] ℝ) ((1 : Fin 3) : ℕ) :=
  scoreFit_linear_iff (Ω := {((0 : Fin 3), ()), (2, ())}) (φ := fun i => ((i : ℕ) : ℝ))
    (ψ := fun _ => (0 : ℝ)) (i := 1) (j := ())
    (by
      rintro c (rfl | rfl) <;> simp
      norm_num)
    (by
      refine Or.inl (segment_subset_convexHull (x := ((0 : ℝ), (0 : ℝ)))
        (y := ((2 : ℝ), (0 : ℝ))) ?_ ?_ ?_)
      · exact ⟨(0, ()), ⟨by simp, by simp⟩, by simp⟩
      · exact ⟨(2, ()), ⟨by simp, by norm_num⟩, by simp⟩
      · exact ⟨1 / 2, 1 / 2, by norm_num, by norm_num, by norm_num, by ext <;> simp⟩)

end Transformer.Grokking.FlatSpectrum

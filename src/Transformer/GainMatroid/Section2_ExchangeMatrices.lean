module

public import Transformer.GainMatroid.Section2_Representable

/-!
# Exchanges of matrices: `I(K₄)` and Proposition 2.11

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. The rows of `I(K₄)` (`iK4`) are `x`, `y`, `z`,
`v`. Its first three columns are the triad `x − v`, `y − v`, `z − v`, and its last three are the
triangle `x − y`, `x − z`, `y − z`.

For a matrix whose columns on a triangle `X` are those of `I(K₄)` (`IsTriangleForm`), `Δ_X A`
(`deltaMatrix`) adds a zero row and replaces these columns by the triad. For a matrix whose columns
on a triad `Y` are those of `I(K₄)` (`IsTriadForm`), `∇_Y A` (`nablaMatrix`) replaces them by the
triangle. `IsTriadForm` also asks that the row `v` vanish off `Y`, so that this row is zero in
`∇_Y A`. The source does not ask this, and `Section2_NablaZeroRow` shows that it is needed.

**Proposition 2.11.** "Let `𝔽` be a field and let `(G, φ)` be a `Γ`-gain graph, where
`Γ ∈ {𝔽ˣ, 𝔽⁺}`. Let `X` be a balanced triangle of `(G, B_φ)` and let `Y` be a `K_{1,3}`-subgraph of
`G`. (i) `Δ_X A_F(G, φ)` is a canonical frame matrix particular to `(Δ_X G, φ)`, (ii)
`∇_Y A_F(G, φ)` is a canonical frame matrix particular to `(∇_Y G, φ)`, (iii) `Δ_X A_L(G, φ)` is a
canonical lift matrix particular to `(Δ_X G, φ)`, (iv) `∇_Y A_L(G, φ)` is a canonical lift matrix
particular to `(∇_Y G, φ)`."

The columns of `A_F(G, φ)` on `X` are those of `I(K₄)` only when `φ` is the identity on `X`, so
these statements take identity gains on the three edges. The rows of a canonical lift matrix start
with the gain row, so part (iii) reorders the rows by `liftSwap`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {V E R F : Type*} [DecidableEq V] [DecidableEq E] [DecidableEq R] [Field F]

/-- The columns `a`, `b`, `c` of `A` are the last three columns of `I(K₄)` on the rows `x`, `y`,
`z`: `a = y − z`, `b = x − z` and `c = x − y`. -/
structure IsTriangleForm (A : Matrix R E F) (a b c : E) (x y z : R) : Prop where
  ne_xy : x ≠ y
  ne_xz : x ≠ z
  ne_yz : y ≠ z
  col_a : ∀ r, A r a = (if r = y then 1 else 0) - if r = z then 1 else 0
  col_b : ∀ r, A r b = (if r = x then 1 else 0) - if r = z then 1 else 0
  col_c : ∀ r, A r c = (if r = x then 1 else 0) - if r = y then 1 else 0

/-- The columns `a`, `b`, `c` of `A` are the first three columns of `I(K₄)` on the rows `x`, `y`,
`z`, `v`, and the row `v` is zero on every other column. -/
structure IsTriadForm (A : Matrix R E F) (v : R) (a b c : E) (x y z : R) : Prop where
  ne_xy : x ≠ y
  ne_xz : x ≠ z
  ne_yz : y ≠ z
  ne_vx : v ≠ x
  ne_vy : v ≠ y
  ne_vz : v ≠ z
  col_a : ∀ r, A r a = (if r = x then 1 else 0) - if r = v then 1 else 0
  col_b : ∀ r, A r b = (if r = y then 1 else 0) - if r = v then 1 else 0
  col_c : ∀ r, A r c = (if r = z then 1 else 0) - if r = v then 1 else 0
  row_v : ∀ e, e ≠ a → e ≠ b → e ≠ c → A v e = 0

/-- The matrix `I(K₄)` of §2.4, with rows `x`, `y`, `z`, `v`. -/
def iK4 : Matrix (Fin 4) (Fin 6) F :=
  !![1, 0, 0, 1, 1, 0;
     0, 1, 0, -1, 0, 1;
     0, 0, 1, 0, -1, -1;
     -1, -1, -1, 0, 0, 0]

/-- The last three columns of `I(K₄)` are the triangle: `a`, `b`, `c` are its sixth, fifth and
fourth columns. -/
theorem iK4_isTriangleForm : IsTriangleForm (iK4 : Matrix (Fin 4) (Fin 6) F) 5 4 3 0 1 2 := by
  refine ⟨by decide, by decide, by decide, fun r => ?_, fun r => ?_, fun r => ?_⟩ <;>
    fin_cases r <;> simp [iK4]

/-- The first three columns of `I(K₄)` are the triad, and the row `v` is zero on the others. -/
theorem iK4_isTriadForm : IsTriadForm (iK4 : Matrix (Fin 4) (Fin 6) F) 3 0 1 2 0 1 2 := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, fun r => ?_,
    fun r => ?_, fun r => ?_, fun e => ?_⟩
  · fin_cases r <;> simp [iK4]
  · fin_cases r <;> simp [iK4]
  · fin_cases r <;> simp [iK4]
  · fin_cases e <;> simp [iK4]

/-- **`Δ_X A`** (§2.4): a new zero row `none`, and the columns of `a`, `b`, `c` replaced by the
first three columns of `I(K₄)`, `x − v`, `y − v` and `z − v`, with `v = none`. -/
def deltaMatrix (A : Matrix R E F) (a b c : E) (x y z : R) : Matrix (Option R) E F :=
  Matrix.of fun r e =>
    if e = a then (if r = some x then 1 else 0) - if r = none then 1 else 0
    else if e = b then (if r = some y then 1 else 0) - if r = none then 1 else 0
    else if e = c then (if r = some z then 1 else 0) - if r = none then 1 else 0
    else r.elim 0 fun w => A w e

/-- **`∇_Y A`** (§2.4): the columns of `a`, `b`, `c` replaced by the last three columns of
`I(K₄)`, `y − z`, `x − z` and `x − y`. -/
def nablaMatrix (A : Matrix R E F) (a b c : E) (x y z : R) : Matrix R E F :=
  Matrix.of fun r e =>
    if e = a then (if r = y then 1 else 0) - if r = z then 1 else 0
    else if e = b then (if r = x then 1 else 0) - if r = z then 1 else 0
    else if e = c then (if r = x then 1 else 0) - if r = y then 1 else 0
    else A r e

/-- The order of rows of `A_L(Δ_X G, ψ)` inside `Δ_X A_L(G, ψ)`: its gain row is the old gain row,
and its row of the new vertex is the new row. -/
def liftSwap : Option (Option V) → Option (Option V)
  | none => some none
  | some none => none
  | some (some w) => some (some w)

/-- **Proposition 2.11 (i).** With identity gains on `X`, `Δ_X A_F(G, φ) = A_F(Δ_X G, φ)`. -/
theorem deltaMatrix_frameMatrix (G : OrientedGraph V E) (a b c : E) (x y z : V) {φ : E → Fˣ}
    (ha : φ a = 1) (hb : φ b = 1) (hc : φ c = 1) :
    deltaMatrix (frameMatrix G φ) a b c x y z = frameMatrix (G.deltaGraph a b c x y z) φ := by
  ext r e
  simp only [deltaMatrix, Matrix.of_apply, frameMatrix_apply, OrientedGraph.deltaGraph]
  by_cases h1 : e = a
  · subst h1
    simp [ha]
  by_cases h2 : e = b
  · subst h2
    simp [h1, hb]
  by_cases h3 : e = c
  · subst h3
    simp [h1, h2, hc]
  · cases r <;> simp [h1, h2, h3]

/-- **Proposition 2.11 (ii).** With identity gains on `Y`, `∇_Y A_F(G, φ) = A_F(∇_Y G, φ)`. -/
theorem nablaMatrix_frameMatrix (G : OrientedGraph V E) (a b c : E) (x y z : V) {φ : E → Fˣ}
    (ha : φ a = 1) (hb : φ b = 1) (hc : φ c = 1) :
    nablaMatrix (frameMatrix G φ) a b c x y z = frameMatrix (G.nablaGraph a b c x y z) φ := by
  ext r e
  simp only [nablaMatrix, Matrix.of_apply, frameMatrix_apply, OrientedGraph.nablaGraph]
  by_cases h1 : e = a
  · subst h1
    simp [ha]
  by_cases h2 : e = b
  · subst h2
    simp [h1, hb]
  by_cases h3 : e = c
  · subst h3
    simp [h1, h2, hc]
  · simp [h1, h2, h3]

/-- **Proposition 2.11 (iii).** With gain `0` on `X`, `Δ_X A_L(G, ψ)` is `A_L(Δ_X G, ψ)` after
reordering its rows. -/
theorem deltaMatrix_liftMatrix (G : OrientedGraph V E) (a b c : E) (x y z : V) {ψ : E → F}
    (ha : ψ a = 0) (hb : ψ b = 0) (hc : ψ c = 0) :
    (deltaMatrix (liftMatrix G ψ) a b c (some x) (some y) (some z)).submatrix liftSwap id =
      liftMatrix (G.deltaGraph a b c x y z) ψ := by
  ext r e
  by_cases h1 : e = a
  · subst h1
    rcases r with _ | _ | w <;> simp [deltaMatrix, liftSwap, liftMatrix, deltaGraph, ha]
  by_cases h2 : e = b
  · subst h2
    rcases r with _ | _ | w <;> simp [deltaMatrix, liftSwap, liftMatrix, deltaGraph, h1, hb]
  by_cases h3 : e = c
  · subst h3
    rcases r with _ | _ | w <;> simp [deltaMatrix, liftSwap, liftMatrix, deltaGraph, h1, h2, hc]
  · rcases r with _ | _ | w <;> simp [deltaMatrix, liftSwap, liftMatrix, deltaGraph, h1, h2, h3]

/-- **Proposition 2.11 (iv).** With gain `0` on `Y`, `∇_Y A_L(G, ψ) = A_L(∇_Y G, ψ)`. -/
theorem nablaMatrix_liftMatrix (G : OrientedGraph V E) (a b c : E) (x y z : V) {ψ : E → F}
    (ha : ψ a = 0) (hb : ψ b = 0) (hc : ψ c = 0) :
    nablaMatrix (liftMatrix G ψ) a b c (some x) (some y) (some z) =
      liftMatrix (G.nablaGraph a b c x y z) ψ := by
  ext r e
  by_cases h1 : e = a
  · subst h1
    rcases r with _ | w <;> simp [nablaMatrix, liftMatrix, nablaGraph, ha]
  by_cases h2 : e = b
  · subst h2
    rcases r with _ | w <;> simp [nablaMatrix, liftMatrix, nablaGraph, h1, hb]
  by_cases h3 : e = c
  · subst h3
    rcases r with _ | w <;> simp [nablaMatrix, liftMatrix, nablaGraph, h1, h2, hc]
  · rcases r with _ | w <;> simp [nablaMatrix, liftMatrix, nablaGraph, h1, h2, h3]

example := deltaMatrix_frameMatrix triangle 1 2 0 0 1 2 (φ := (1 : Fin 3 → ℚˣ)) rfl rfl rfl

example := nablaMatrix_frameMatrix clawGraph 0 1 2 1 2 3 (φ := (1 : Fin 5 → ℚˣ)) rfl rfl rfl

example := deltaMatrix_liftMatrix triangle 1 2 0 0 1 2 (ψ := (0 : Fin 3 → ℚ)) rfl rfl rfl

example := nablaMatrix_liftMatrix clawGraph 0 1 2 1 2 3 (ψ := (0 : Fin 5 → ℚ)) rfl rfl rfl

end Transformer.GainMatroid

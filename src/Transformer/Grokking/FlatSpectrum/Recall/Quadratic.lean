module

public import Transformer.Grokking.FlatSpectrum.Recall.Polynomial

/-!
# Readouts of degree one and two in the numbers that cross the cut

Source: `review/flat_spectrum.md`, *Polynomial readouts and two layers* (added), item PR2, for
the readouts of Arora et al., arXiv:2312.04927v1, §4, read at the query.

Fix `D` numbers `z(x)` of the context. A readout is **affine in `z`** when it is
`g(q) + ∑_i a_i(q) z_i(x)` (`IsAffineIn`): a readout of cut width `D` whose features are `z`.
It is **quadratic in `z`** when it is `g(q) + ∑_i a_i(q) z_i + ∑_{i,j} A_{ij}(q) z_i z_j`
(`IsQuadIn`). Affine readouts are closed under sums and multiples by functions of the query
(`IsAffineIn.add`, `IsAffineIn.smul`, `IsAffineIn.sum`). The product of two affine readouts is
quadratic (`IsAffineIn.mul`), and so is a sum of quadratic readouts (`IsQuadIn.sum`).

A quadratic readout has cut width `D + D²`: the numbers `z_i` and their products `z_i z_j`
cross the cut (`IsQuadIn.hasCutWidth`). It is a polynomial readout of degree two, and an
affine one of degree one (`IsQuadIn.isPolyIn`, `IsAffineIn.isPolyIn`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

variable {Q C : Type*} {D : ℕ}

/-- `F` is affine in the numbers `z` of the context: `F q x = g q + ∑_i a_i(q) z_i(x)`. -/
def IsAffineIn (z : C → Fin D → ℝ) (F : Q → C → ℝ) : Prop :=
  ∃ (g : Q → ℝ) (a : Q → Fin D → ℝ), ∀ q x, F q x = g q + ∑ i, a q i * z x i

/-- `F` is quadratic in the numbers `z` of the context. -/
def IsQuadIn (z : C → Fin D → ℝ) (F : Q → C → ℝ) : Prop :=
  ∃ (g : Q → ℝ) (a : Q → Fin D → ℝ) (A : Q → Fin D → Fin D → ℝ),
    ∀ q x, F q x = g q + ∑ i, a q i * z x i + ∑ i, ∑ j, A q i j * (z x i * z x j)

/-- A function of the query alone is affine. -/
lemma isAffineIn_query (z : C → Fin D → ℝ) (g : Q → ℝ) : IsAffineIn z fun q _ => g q :=
  ⟨g, fun _ _ => 0, fun q x => by simp⟩

/-- A crossing number is affine. -/
lemma isAffineIn_coord (z : C → Fin D → ℝ) (i : Fin D) : IsAffineIn z fun (_ : Q) x => z x i :=
  ⟨fun _ => 0, fun _ j => if j = i then 1 else 0, fun q x => by simp⟩

variable {z : C → Fin D → ℝ} {F G : Q → C → ℝ}

/-- An affine readout has cut width `D`. -/
lemma IsAffineIn.hasCutWidth (h : IsAffineIn z F) : HasCutWidth D F :=
  let ⟨g, a, h⟩ := h
  ⟨g, a, z, h⟩

/-- The sum of two affine readouts is affine. -/
lemma IsAffineIn.add (hF : IsAffineIn z F) (hG : IsAffineIn z G) :
    IsAffineIn z fun q x => F q x + G q x := by
  obtain ⟨g, a, hF⟩ := hF
  obtain ⟨g', a', hG⟩ := hG
  refine ⟨fun q => g q + g' q, fun q i => a q i + a' q i, fun q x => ?_⟩
  simp only [hF, hG, add_mul, Finset.sum_add_distrib]
  ring

/-- A multiple of an affine readout by a function of the query is affine. -/
lemma IsAffineIn.smul (c : Q → ℝ) (hF : IsAffineIn z F) :
    IsAffineIn z fun q x => c q * F q x := by
  obtain ⟨g, a, hF⟩ := hF
  refine ⟨fun q => c q * g q, fun q i => c q * a q i, fun q x => ?_⟩
  simp only [hF, mul_add, Finset.mul_sum, mul_assoc]

/-- A finite sum of affine readouts is affine. -/
lemma IsAffineIn.sum {ι : Type*} (s : Finset ι) {F : ι → Q → C → ℝ}
    (h : ∀ i, IsAffineIn z (F i)) : IsAffineIn z fun q x => ∑ i ∈ s, F i q x := by
  choose g a h using h
  refine ⟨fun q => ∑ i ∈ s, g i q, fun q j => ∑ i ∈ s, a i q j, fun q x => ?_⟩
  simp only [h, Finset.sum_add_distrib, Finset.sum_mul]
  rw [Finset.sum_comm]

/-- The product of two affine forms, expanded. -/
lemma affine_mul_affine (g g' : ℝ) (a a' y : Fin D → ℝ) :
    (g + ∑ i, a i * y i) * (g' + ∑ j, a' j * y j) =
      g * g' + ∑ i, (g * a' i + g' * a i) * y i + ∑ i, ∑ j, a i * a' j * (y i * y j) := by
  have h1 : ∑ i, (g * a' i + g' * a i) * y i = g * ∑ j, a' j * y j + g' * ∑ i, a i * y i := by
    simp only [add_mul, Finset.sum_add_distrib, Finset.mul_sum, mul_assoc]
  have h2 : ∑ i, ∑ j, a i * a' j * (y i * y j) = (∑ i, a i * y i) * ∑ j, a' j * y j := by
    rw [Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  rw [h1, h2]
  ring

example (a a' y : Fin 2 → ℝ) := affine_mul_affine 1 2 a a' y

/-- **The product of two affine readouts is quadratic** (`review/flat_spectrum.md`, PR2). -/
lemma IsAffineIn.mul (hF : IsAffineIn z F) (hG : IsAffineIn z G) :
    IsQuadIn z fun q x => F q x * G q x := by
  obtain ⟨g, a, hF⟩ := hF
  obtain ⟨g', a', hG⟩ := hG
  exact ⟨fun q => g q * g' q, fun q i => g q * a' q i + g' q * a q i,
    fun q i j => a q i * a' q j, fun q x => by simp only [hF, hG, affine_mul_affine]⟩

/-- A finite sum of quadratic readouts is quadratic. -/
lemma IsQuadIn.sum {ι : Type*} (s : Finset ι) {F : ι → Q → C → ℝ}
    (h : ∀ i, IsQuadIn z (F i)) : IsQuadIn z fun q x => ∑ i ∈ s, F i q x := by
  choose g a A h using h
  refine ⟨fun q => ∑ c ∈ s, g c q, fun q i => ∑ c ∈ s, a c q i,
    fun q i j => ∑ c ∈ s, A c q i j, fun q x => ?_⟩
  simp only [h, Finset.sum_add_distrib, Finset.sum_mul]
  rw [Finset.sum_comm (s := s), Finset.sum_comm (s := s)]
  congr 1
  exact Finset.sum_congr rfl fun i _ => Finset.sum_comm

/-- **A quadratic readout has cut width `D + D²`** (`review/flat_spectrum.md`, PR2): the
numbers `z_i` and their products `z_i z_j` cross the cut. -/
theorem IsQuadIn.hasCutWidth (h : IsQuadIn z F) : HasCutWidth (D + D * D) F := by
  obtain ⟨g, a, A, h⟩ := h
  have h1 := hasCutWidth_of_fintype (ι := Fin D ⊕ (Fin D × Fin D)) (F := F) g
    (fun q => Sum.elim (a q) fun ij => A q ij.1 ij.2)
    (fun x => Sum.elim (z x) fun ij => z x ij.1 * z x ij.2) fun q x => by
      rw [h, Fintype.sum_sum_type, Fintype.sum_prod_type]
      simp only [Sum.elim_inl, Sum.elim_inr, add_assoc]
  simpa only [Fintype.card_sum, Fintype.card_prod, Fintype.card_fin] using h1

/-- An affine readout is a polynomial readout of degree one. -/
lemma IsAffineIn.isPolyIn (h : IsAffineIn z F) : IsPolyIn z 1 F := by
  obtain ⟨g, a, h⟩ := h
  obtain ⟨P, hd, hP⟩ := ((isPolyIn_query z g).mono zero_le_one).add
    (IsPolyIn.sum Finset.univ fun i =>
      ((isPolyIn_query z fun q => a q i).mul (isPolyIn_coord z i)).mono (zero_add 1).le)
  exact ⟨P, hd, fun q x => (h q x).trans (hP q x)⟩

/-- **A quadratic readout is a polynomial readout of degree two**. -/
lemma IsQuadIn.isPolyIn (h : IsQuadIn z F) : IsPolyIn z 2 F := by
  obtain ⟨g, a, A, h⟩ := h
  obtain ⟨P, hd, hP⟩ := (((isPolyIn_query z g).mono (Nat.zero_le 2)).add
    (IsPolyIn.sum Finset.univ fun i =>
      ((isPolyIn_query z fun q => a q i).mul (isPolyIn_coord z i)).mono (by norm_num))).add
    (IsPolyIn.sum Finset.univ fun i => IsPolyIn.sum Finset.univ fun j =>
      ((isPolyIn_query z fun q => A q i j).mul
        ((isPolyIn_coord z i).mul (isPolyIn_coord z j))).mono (by norm_num))
  exact ⟨P, hd, fun q x => (h q x).trans (hP q x)⟩

/-- `(1 + q z₀)(z₀ + z₁)`: a product of two affine readouts in two numbers has cut width
`2 + 2 · 2` and degree two. -/
example (z : ℝ → Fin 2 → ℝ) :
    HasCutWidth (2 + 2 * 2) fun (q : ℝ) x => (1 + q * z x 0) * (z x 0 + z x 1) :=
  (((isAffineIn_query z fun _ => (1 : ℝ)).add
    ((isAffineIn_coord z 0).smul fun q => q)).mul
      ((isAffineIn_coord z 0).add (isAffineIn_coord z 1))).hasCutWidth

example (z : ℝ → Fin 2 → ℝ) := (IsQuadIn.sum (Finset.univ : Finset (Fin 2)) fun i =>
  (isAffineIn_coord (Q := ℝ) z i).mul (isAffineIn_coord z i)).isPolyIn

example (z : ℝ → Fin 2 → ℝ) := (IsAffineIn.sum (Finset.univ : Finset (Fin 2)) fun i =>
  isAffineIn_coord (Q := ℝ) z i).isPolyIn

example (z : ℝ → Fin 2 → ℝ) := (isAffineIn_coord (Q := ℝ) z 0).hasCutWidth

end Transformer.Grokking.FlatSpectrum.Recall

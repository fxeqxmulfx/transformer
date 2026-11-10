module

public import Transformer.Grokking.FlatSpectrum.Recall.Defs
public import Transformer.Grokking.FlatSpectrum.Recall.Monomials

/-!
# Polynomial readouts of the numbers that cross the cut

Source: `review/flat_spectrum.md`, *Polynomial readouts and two layers* (added), item PR2, for
the task of Arora et al., arXiv:2312.04927v1, §3, and the second scope item of AR10: an MLP
after the layer, here with a polynomial activation.

`D` numbers `z(x)` cross from the context, and the query combines them by a polynomial of total
degree at most `Δ` whose coefficients depend on the query (`IsPolyIn`, `HasPolyCut`). Such
readouts are closed under sums and products, the degrees adding in a product (`IsPolyIn.add`,
`IsPolyIn.mul`, `IsPolyIn.sum`, `IsPolyIn.pow`).

A polynomial readout is a combination of the `C(D + Δ, D)` monomials of degree at most `Δ` in
`z`, with coefficients that depend on the query. So its cut width is at most `C(D + Δ, D)`
(`IsPolyIn.hasCutWidth`), and the bounds of AR4 and AR5 hold for it with this width.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open MvPolynomial

variable {Q C : Type*} {D : ℕ}

/-- **A polynomial readout** (`review/flat_spectrum.md`, PR2): `F q x` is a polynomial of total
degree at most `Δ` in the numbers `z x` of the context, with coefficients that depend on `q`. -/
def IsPolyIn (z : C → Fin D → ℝ) (Δ : ℕ) (F : Q → C → ℝ) : Prop :=
  ∃ P : Q → MvPolynomial (Fin D) ℝ, (∀ q, (P q).totalDegree ≤ Δ) ∧
    ∀ q x, F q x = eval (z x) (P q)

/-- `F` is a polynomial of degree at most `Δ` in `D` numbers that cross from the context. -/
def HasPolyCut (D Δ : ℕ) (F : Q → C → ℝ) : Prop := ∃ z : C → Fin D → ℝ, IsPolyIn z Δ F

/-- A function of the query alone has degree `0`. -/
lemma isPolyIn_query (z : C → Fin D → ℝ) (g : Q → ℝ) : IsPolyIn z 0 fun q _ => g q :=
  ⟨fun q => MvPolynomial.C (g q), fun _ => (totalDegree_C _).le, fun _ _ => (eval_C _).symm⟩

/-- A crossing number has degree `1`. -/
lemma isPolyIn_coord (z : C → Fin D → ℝ) (i : Fin D) : IsPolyIn z 1 fun (_ : Q) x => z x i :=
  ⟨fun _ => X i, fun _ => (totalDegree_X i).le, fun _ _ => (eval_X i).symm⟩

variable {z : C → Fin D → ℝ} {Δ Δ' : ℕ} {F G : Q → C → ℝ}

/-- A larger degree bound holds too. -/
lemma IsPolyIn.mono (h : IsPolyIn z Δ F) (hΔ : Δ ≤ Δ') : IsPolyIn z Δ' F :=
  let ⟨P, hd, hP⟩ := h
  ⟨P, fun q => (hd q).trans hΔ, hP⟩

/-- A sum has the degree of its terms. -/
lemma IsPolyIn.add (hF : IsPolyIn z Δ F) (hG : IsPolyIn z Δ G) :
    IsPolyIn z Δ fun q x => F q x + G q x := by
  obtain ⟨P, hd, hP⟩ := hF
  obtain ⟨P', hd', hP'⟩ := hG
  exact ⟨fun q => P q + P' q, fun q => (totalDegree_add _ _).trans (max_le (hd q) (hd' q)),
    fun q x => by simp only [map_add, hP, hP']⟩

/-- **The degrees add in a product** (`review/flat_spectrum.md`, PR2). -/
lemma IsPolyIn.mul (hF : IsPolyIn z Δ F) (hG : IsPolyIn z Δ' G) :
    IsPolyIn z (Δ + Δ') fun q x => F q x * G q x := by
  obtain ⟨P, hd, hP⟩ := hF
  obtain ⟨P', hd', hP'⟩ := hG
  exact ⟨fun q => P q * P' q,
    fun q => (totalDegree_mul _ _).trans (Nat.add_le_add (hd q) (hd' q)),
    fun q x => by simp only [map_mul, hP, hP']⟩

/-- A finite sum has the degree of its terms. -/
lemma IsPolyIn.sum {ι : Type*} (s : Finset ι) {F : ι → Q → C → ℝ}
    (h : ∀ i, IsPolyIn z Δ (F i)) : IsPolyIn z Δ fun q x => ∑ i ∈ s, F i q x := by
  choose P hd hP using h
  exact ⟨fun q => ∑ i ∈ s, P i q, fun q => totalDegree_finsetSum_le fun i _ => hd i q,
    fun q x => by rw [map_sum]; exact Finset.sum_congr rfl fun i _ => hP i q x⟩

/-- The `n`-th power has `n` times the degree. -/
lemma IsPolyIn.pow (hF : IsPolyIn z Δ F) (n : ℕ) : IsPolyIn z (n * Δ) fun q x => F q x ^ n := by
  obtain ⟨P, hd, hP⟩ := hF
  exact ⟨fun q => P q ^ n,
    fun q => (totalDegree_pow _ _).trans (Nat.mul_le_mul_left n (hd q)),
    fun q x => by simp only [map_pow, hP]⟩

/-- A polynomial readout stays one when the context is the image of another one. -/
lemma IsPolyIn.comp_context {C' : Type*} (hF : IsPolyIn z Δ F) (ι : C' → C) :
    IsPolyIn (fun x => z (ι x)) Δ fun q x => F q (ι x) :=
  let ⟨P, hd, hP⟩ := hF
  ⟨P, hd, fun q x => hP q (ι x)⟩

/-- `1 + q · z₀ + z₀ z₁` has degree two in two numbers. -/
example (z : ℝ → Fin 2 → ℝ) : IsPolyIn z 2 fun (q : ℝ) x => 1 + q * z x 0 + z x 0 * z x 1 :=
  ((((isPolyIn_query z fun _ => (1 : ℝ)).mono (Nat.zero_le 2)).add
    (((isPolyIn_query z fun q => q).mul (isPolyIn_coord z 0)).mono (by norm_num))).add
      ((isPolyIn_coord z 0).mul (isPolyIn_coord z 1)))

/-- The sum of the two numbers, squared, read after a context `2 x`: degree two. -/
example (z : ℝ → Fin 2 → ℝ) :
    IsPolyIn (fun x : ℝ => z (2 * x)) (2 * 1) fun (_ : ℝ) x => (∑ i, z (2 * x) i) ^ 2 :=
  ((IsPolyIn.sum Finset.univ fun i => isPolyIn_coord (Q := ℝ) z i).pow 2).comp_context
    fun x => 2 * x

/-- A readout with features indexed by a finite type has cut width its number of features. -/
lemma hasCutWidth_of_fintype {ι : Type*} [Fintype ι] (g : Q → ℝ) (a : Q → ι → ℝ)
    (b : C → ι → ℝ) (h : ∀ q x, F q x = g q + ∑ i, a q i * b x i) :
    HasCutWidth (Fintype.card ι) F :=
  ⟨g, fun q r => a q ((Fintype.equivFin ι).symm r), fun x r => b x ((Fintype.equivFin ι).symm r),
    fun q x => by
      rw [h]
      congr 1
      exact Fintype.sum_equiv (Fintype.equivFin ι) _ _ fun i => by simp⟩

example : HasCutWidth (Fintype.card Bool) fun (q x : ℝ) => q * x + q * x :=
  hasCutWidth_of_fintype (fun _ => 0) (fun q _ => q) (fun x _ => x) fun q x => by
    rw [Fintype.sum_bool, zero_add]

/-- **A polynomial readout of degree `Δ` in `D` numbers has cut width at most `C(D + Δ, D)`**
(`review/flat_spectrum.md`, PR2): its monomials cross the cut. -/
theorem IsPolyIn.hasCutWidth (h : IsPolyIn z Δ F) : HasCutWidth ((D + Δ).choose D) F := by
  obtain ⟨P, hd, hP⟩ := h
  have h1 := hasCutWidth_of_fintype (ι := ↥(monomialExponents D Δ)) (F := F) (fun _ => 0)
    (fun q e => (P q).coeff (Finsupp.equivFunOnFinite.symm fun i => e.1 i.castSucc))
    (fun x e => ∏ i, z x i ^ e.1 i.castSucc) fun q x => by
      rw [zero_add, hP, eval_eq_sum_monomialExponents (hd q)]
  rwa [card_monomialExponents] at h1

/-- `1 + q · z₀ + z₀ z₁` has cut width at most `C(4, 2) = 6`. -/
example (z : ℝ → Fin 2 → ℝ) :
    HasCutWidth ((2 + 2).choose 2) fun (q : ℝ) x => 1 + q * z x 0 + z x 0 * z x 1 :=
  ((((isPolyIn_query z fun _ => (1 : ℝ)).mono (Nat.zero_le 2)).add
    (((isPolyIn_query z fun q => q).mul (isPolyIn_coord z 0)).mono (by norm_num))).add
      ((isPolyIn_coord z 0).mul (isPolyIn_coord z 1))).hasCutWidth

end Transformer.Grokking.FlatSpectrum.Recall

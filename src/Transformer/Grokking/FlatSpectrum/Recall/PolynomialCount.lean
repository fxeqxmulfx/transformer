module

public import Transformer.Grokking.FlatSpectrum.FlatCapacity
public import Transformer.Grokking.FlatSpectrum.Recall.Parity
public import Transformer.Grokking.FlatSpectrum.Recall.Polynomial

/-!
# A polynomial readout recalls a number of bindings linear in the numbers that cross

Source: `review/flat_spectrum.md`, *Polynomial readouts and two layers* (added), items PR3 and
PR4, for the task of Arora et al., arXiv:2312.04927v1, §3, and the scope item of AR10 on an MLP
after the layer, here with a polynomial activation.

PR3. Let a polynomial readout of degree at most `Δ` in `D` numbers `z` of the context recall `m`
bindings with values `±1` exactly. The product of its values at the keys of a set `S` is the
parity of `S`, a polynomial of degree at most `Δ m` in `z`, and so a combination of the
`C(D + Δ m, D)` monomials of that degree (`eval_eq_sum_monomialExponents`). So the parity
kernel, flat of rank `2^m`, is a product through that width:

    2^m ≤ C(D + Δ m, D)                            (`IsPolyIn.two_pow_le_choose_of_sign`),

and the same for every readout `φ` that is not constant
(`HasPolyCut.two_pow_le_choose_of_recall`). For `Δ = 1` and `D = 1` it is an equality at
`m = 1`. The count is linear in `m`: `(4/3)^m ≤ (2Δ + 1)^{2D}` for every degree
(`HasPolyCut.four_pow_le_of_recall`), and `m ≤ 5 D` at degree two
(`HasPolyCut.card_le_five_mul_of_recall`).

PR4. A hidden layer with the activation `t ↦ t^p` over readouts of degree one is a polynomial
readout of degree `p` (`IsPolyIn.powerHead`), so its exact recall has
`(4/3)^m ≤ (2p + 1)^{2D}`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open MvPolynomial

variable {κ ν : Type*} [Fintype κ] [DecidableEq κ] {D : ℕ}

/-- **The count for values `±1`** (`review/flat_spectrum.md`, PR3): a polynomial readout of
degree at most `Δ` in `D` numbers of the context that returns `±1` of the bound value for every
query and dictionary has `2^m ≤ C(D + Δ m, D)`. -/
theorem IsPolyIn.two_pow_le_choose_of_sign {Δ : ℕ} {z : (κ → Bool) → Fin D → ℝ}
    {F : κ → (κ → Bool) → ℝ} (hF : IsPolyIn z Δ F)
    (h : ∀ q β, F q β = if β q then 1 else -1) :
    2 ^ Fintype.card κ ≤ (D + Δ * Fintype.card κ).choose D := by
  obtain ⟨P, hdeg, hP⟩ := hF
  have hQ (S : Finset κ) : (∏ q ∈ S, P q).totalDegree ≤ Δ * Fintype.card κ :=
    (totalDegree_finsetProd S P).trans ((Finset.sum_le_card_nsmul S _ Δ fun q _ => hdeg q).trans
      (by rw [smul_eq_mul, mul_comm]; exact Nat.mul_le_mul_left Δ (Finset.card_le_univ S)))
  have hfac : parityKernel κ =
      (Matrix.of fun (S : Finset κ) (e : ↥(monomialExponents D (Δ * Fintype.card κ))) =>
        (((∏ q ∈ S, P q).coeff
          (Finsupp.equivFunOnFinite.symm fun i => e.1 i.castSucc) : ℝ) : ℂ)) *
      Matrix.of fun (e : ↥(monomialExponents D (Δ * Fintype.card κ))) (β : κ → Bool) =>
        ((∏ i, z β i ^ e.1 i.castSucc : ℝ) : ℂ) := by
    ext S β
    simp only [Matrix.mul_apply, parityKernel, Matrix.of_apply, ← Complex.ofReal_mul,
      ← Complex.ofReal_sum]
    rw [← eval_eq_sum_monomialExponents (hQ S) (z β), eval_prod]
    exact congrArg _ (Finset.prod_congr rfl fun q _ => ((hP q β).symm.trans (h q β)).symm)
  have hcard := isFlat_parityKernel.card_le_of_eq_mul (by positivity) hfac
  simpa only [Fintype.card_finset, Fintype.card_fun, Fintype.card_bool, min_self,
    card_monomialExponents] using hcard

/-- One key, one number and degree one: `2 ≤ C(1 + 1, 1)`, an equality. -/
example : 2 ^ Fintype.card (Fin 1) ≤ (1 + 1 * Fintype.card (Fin 1)).choose 1 :=
  IsPolyIn.two_pow_le_choose_of_sign
    (isPolyIn_coord (Q := Fin 1) (fun (β : Fin 1 → Bool) (_ : Fin 1) => if β 0 then 1 else -1) 0)
    fun q β => by rw [Subsingleton.elim q 0]

variable {Δ : ℕ} {F : κ → (κ → ν) → ℝ} {φ : ν → ℝ} {v v' : ν}

/-- **A polynomial readout recalls `m` bindings only if `2^m ≤ C(D + Δ m, D)`**
(`review/flat_spectrum.md`, PR3): `D` numbers cross from the context, the query combines them
by a polynomial of degree at most `Δ`, and the readout returns a non-constant `φ` of the bound
value for every query and dictionary. -/
theorem HasPolyCut.two_pow_le_choose_of_recall (hF : HasPolyCut D Δ F) (hφ : φ v ≠ φ v')
    (h : ∀ q π, F q π = φ (π q)) :
    2 ^ Fintype.card κ ≤ (D + Δ * Fintype.card κ).choose D := by
  obtain ⟨z, hz⟩ := hF
  have hab : φ v - φ v' ≠ 0 := sub_ne_zero.2 hφ
  have h1 := (((isPolyIn_query _ fun _ : κ => 2 / (φ v - φ v')).mul
    (hz.comp_context fun (β : κ → Bool) q => if β q then v else v')).mono (zero_add Δ).le).add
      ((isPolyIn_query _ fun _ : κ => -(φ v + φ v') / (φ v - φ v')).mono (Nat.zero_le Δ))
  refine h1.two_pow_le_choose_of_sign fun q β => ?_
  rw [h]
  split_ifs <;> field_simp <;> ring

/-- **The dimension of a polynomial readout is linear in the bindings it recalls**
(`review/flat_spectrum.md`, PR3): `(4/3)^m ≤ (2Δ + 1)^{2D}`, so
`m ≤ 2 D log(2Δ + 1) / log(4/3)`. -/
theorem HasPolyCut.four_pow_le_of_recall (hF : HasPolyCut D Δ F) (hφ : φ v ≠ φ v')
    (h : ∀ q π, F q π = φ (π q)) :
    4 ^ Fintype.card κ ≤ 3 ^ Fintype.card κ * (2 * Δ + 1) ^ (2 * D) :=
  four_pow_le_of_le_choose (hF.two_pow_le_choose_of_recall hφ h)

/-- **A readout of degree two recalls at most five bindings for each number that crosses**
(`review/flat_spectrum.md`, PR3). -/
theorem HasPolyCut.card_le_five_mul_of_recall {F : κ → (κ → ν) → ℝ} (hF : HasPolyCut D 2 F)
    (hφ : φ v ≠ φ v') (h : ∀ q π, F q π = φ (π q)) : Fintype.card κ ≤ 5 * D :=
  le_five_mul_of_le_choose (hF.two_pow_le_choose_of_recall hφ h)

/-- One key with values `±1` is read from one number by a polynomial of degree one, and so of
degree two. -/
lemma hasPolyCut_one_key (Δ : ℕ) (hΔ : 1 ≤ Δ) :
    HasPolyCut 1 Δ fun (_ : Fin 1) (π : Fin 1 → Bool) => if π 0 then (1 : ℝ) else -1 :=
  ⟨fun π _ => if π 0 then 1 else -1, (isPolyIn_coord (Q := Fin 1)
    (fun (π : Fin 1 → Bool) (_ : Fin 1) => if π 0 then (1 : ℝ) else -1) 0).mono hΔ⟩

example : 2 ^ Fintype.card (Fin 1) ≤ (1 + 1 * Fintype.card (Fin 1)).choose 1 :=
  (hasPolyCut_one_key 1 le_rfl).two_pow_le_choose_of_recall
    (φ := fun b : Bool => if b then (1 : ℝ) else -1) (v := true) (v' := false) (by norm_num)
    fun q π => by rw [Subsingleton.elim q 0]

example : 4 ^ Fintype.card (Fin 1) ≤ 3 ^ Fintype.card (Fin 1) * (2 * 1 + 1) ^ (2 * 1) :=
  (hasPolyCut_one_key 1 le_rfl).four_pow_le_of_recall
    (φ := fun b : Bool => if b then (1 : ℝ) else -1) (v := true) (v' := false) (by norm_num)
    fun q π => by rw [Subsingleton.elim q 0]

example : Fintype.card (Fin 1) ≤ 5 * 1 :=
  (hasPolyCut_one_key 2 one_le_two).card_le_five_mul_of_recall
    (φ := fun b : Bool => if b then (1 : ℝ) else -1) (v := true) (v' := false) (by norm_num)
    fun q π => by rw [Subsingleton.elim q 0]

example := hasPolyCut_one_key 3 (by norm_num)

/-- **A hidden layer with the activation `t ↦ t^p`** (`review/flat_spectrum.md`, PR4): over
readouts `Y_j` of degree one in the numbers `z`, the head `∑_h u_h (∑_j V_{hj} Y_j + b_h)^p` is
a polynomial readout of degree `p`. -/
theorem IsPolyIn.powerHead {Q C : Type*} {z : C → Fin D → ℝ} {H r : ℕ}
    {Y : Fin r → Q → C → ℝ} (hY : ∀ j, IsPolyIn z 1 (Y j)) (p : ℕ) (u b : Fin H → ℝ)
    (V : Fin H → Fin r → ℝ) :
    IsPolyIn z p fun q x => ∑ h, u h * (∑ j, V h j * Y j q x + b h) ^ p := by
  refine IsPolyIn.sum Finset.univ fun h => ?_
  have hin : IsPolyIn z 1 fun q x => ∑ j, V h j * Y j q x + b h :=
    (IsPolyIn.sum Finset.univ fun j =>
      ((isPolyIn_query z fun _ : Q => V h j).mul (hY j)).mono (zero_add 1).le).add
        ((isPolyIn_query z fun _ : Q => b h).mono zero_le_one)
  exact ((isPolyIn_query z fun _ : Q => u h).mul (hin.pow p)).mono (by rw [zero_add, mul_one])

/-- One hidden unit that squares the sum of two crossing numbers. -/
example (z : ℝ → Fin 2 → ℝ) :=
  IsPolyIn.powerHead (Q := ℝ) (fun j => isPolyIn_coord z j) 2 (fun _ : Fin 1 => 1) (fun _ => 0)
    fun _ _ => 1

end Transformer.Grokking.FlatSpectrum.Recall

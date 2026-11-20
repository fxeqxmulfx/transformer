module

public import Transformer.Grokking.FlatSpectrum.ProductOrder

/-!
# The alternating sum over the corners of a box

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV7. Fix two values of every
operand and take the alternating sum of a function over the `2^{2k}` corners of the box they span
(`cornerSum`). A function that does not read some operand cancels in pairs across it
(`cornerSum_eq_zero_of_readsOnly`), so every function of order less than `2k` has corner sum zero
(`cornerSum_eq_zero`). The corner sum of a product over the pairs is the product of the pairs'
`2 × 2` mixed differences (`cornerSum_prod`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {κ α β : Type}

/-- The corner `ε` of the box spanned by `x₀` and `x₁`: `x_i` takes its second value where
`ε (inl i)` holds. -/
def cornerX (x₀ x₁ : κ → α) (ε : κ ⊕ κ → Bool) : κ → α :=
  fun i => if ε (Sum.inl i) then x₁ i else x₀ i

/-- The corner `ε` of the box spanned by `y₀` and `y₁`. -/
def cornerY (y₀ y₁ : κ → β) (ε : κ ⊕ κ → Bool) : κ → β :=
  fun i => if ε (Sum.inr i) then y₁ i else y₀ i

variable [DecidableEq κ]

/-- Flip a corner at one operand. -/
def flipAt (o : κ ⊕ κ) (ε : κ ⊕ κ → Bool) : κ ⊕ κ → Bool := Function.update ε o (!ε o)

lemma flipAt_flipAt (o : κ ⊕ κ) (ε : κ ⊕ κ → Bool) : flipAt o (flipAt o ε) = ε := by
  funext o'
  by_cases h : o' = o
  · subst h
    simp [flipAt]
  · simp [flipAt, Function.update_of_ne h]

lemma flipAt_ne (o : κ ⊕ κ) (ε : κ ⊕ κ → Bool) : flipAt o ε ≠ ε := fun h => by
  have := congrFun h o
  simp [flipAt] at this

variable [Fintype κ]

/-- The sign of a corner: `−1` to the number of operands at their second value. -/
def cornerSign (ε : κ ⊕ κ → Bool) : ℂ := ∏ o, if ε o then -1 else 1

/-- The alternating sum of `φ` over the corners of the box, a linear functional. -/
def cornerSum (x₀ x₁ : κ → α) (y₀ y₁ : κ → β) : ((κ → α) → (κ → β) → ℂ) →ₗ[ℂ] ℂ where
  toFun φ := ∑ ε, cornerSign ε * φ (cornerX x₀ x₁ ε) (cornerY y₀ y₁ ε)
  map_add' φ ψ := by simp [mul_add, sum_add_distrib]
  map_smul' c φ := by simp [mul_sum, mul_left_comm]

lemma cornerSum_apply (x₀ x₁ : κ → α) (y₀ y₁ : κ → β) (φ : (κ → α) → (κ → β) → ℂ) :
    cornerSum x₀ x₁ y₀ y₁ φ = ∑ ε, cornerSign ε * φ (cornerX x₀ x₁ ε) (cornerY y₀ y₁ ε) := rfl

lemma cornerSign_flipAt (o : κ ⊕ κ) (ε : κ ⊕ κ → Bool) :
    cornerSign (flipAt o ε) = -cornerSign ε := by
  simp only [cornerSign]
  rw [Fintype.prod_eq_mul_prod_compl o, Fintype.prod_eq_mul_prod_compl o]
  have h : ∏ o' ∈ {o}ᶜ, (if flipAt o ε o' then (-1 : ℂ) else 1) =
      ∏ o' ∈ {o}ᶜ, if ε o' then (-1 : ℂ) else 1 :=
    prod_congr rfl fun o' ho' => by rw [flipAt, Function.update_of_ne (by simpa using ho')]
  rw [h, flipAt, Function.update_self]
  cases ε o <;> simp

/-- **A function that skips an operand cancels in pairs** across it. -/
lemma cornerSum_eq_zero_of_readsOnly (x₀ x₁ : κ → α) (y₀ y₁ : κ → β) {S : Finset (κ ⊕ κ)}
    {o : κ ⊕ κ} (ho : o ∉ S) {φ : (κ → α) → (κ → β) → ℂ} (hφ : ReadsOnly S φ) :
    cornerSum x₀ x₁ y₀ y₁ φ = 0 := by
  have hval (ε : κ ⊕ κ → Bool) : φ (cornerX x₀ x₁ (flipAt o ε)) (cornerY y₀ y₁ (flipAt o ε)) =
      φ (cornerX x₀ x₁ ε) (cornerY y₀ y₁ ε) := by
    refine hφ _ _ _ _ (fun i hi => ?_) fun i hi => ?_
    · have : Sum.inl i ≠ o := fun h => ho (h ▸ hi)
      simp [cornerX, flipAt, Function.update_of_ne this]
    · have : Sum.inr i ≠ o := fun h => ho (h ▸ hi)
      simp [cornerY, flipAt, Function.update_of_ne this]
  rw [cornerSum_apply]
  exact sum_involution (fun ε _ => flipAt o ε)
    (fun ε _ => by rw [hval, cornerSign_flipAt]; ring) (fun ε _ _ => flipAt_ne o ε)
    (fun _ _ => mem_univ _) fun ε _ => flipAt_flipAt o ε

example : cornerSum (κ := Unit) (α := Bool) (β := Bool) (fun _ => false) (fun _ => true)
    (fun _ => false) (fun _ => true) (fun _ _ => 1) = 0 :=
  cornerSum_eq_zero_of_readsOnly _ _ _ _ (S := ∅) (o := Sum.inl ()) (notMem_empty _)
    fun _ _ _ _ _ _ => rfl

/-- **Every function of order less than `2k` has corner sum zero**, `k = |κ|`. -/
theorem cornerSum_eq_zero (x₀ x₁ : κ → α) (y₀ y₁ : κ → β) {r : ℕ} (hr : r < 2 * Fintype.card κ)
    {φ : (κ → α) → (κ → β) → ℂ} (hφ : φ ∈ orderSpan r) : cornerSum x₀ x₁ y₀ y₁ φ = 0 := by
  refine LinearMap.mem_ker.1 (Submodule.span_le.2 (fun ψ hψ => ?_) hφ)
  obtain ⟨S, hS, hψ⟩ := hψ
  obtain ⟨o, -, ho⟩ := exists_mem_notMem_of_card_lt_card (s := S) (t := univ)
    (by rw [card_univ, Fintype.card_sum]; omega)
  exact SetLike.mem_coe.2 (LinearMap.mem_ker.2 (cornerSum_eq_zero_of_readsOnly x₀ x₁ y₀ y₁ ho hψ))

example : cornerSum (κ := Unit) (α := Bool) (β := Bool) (fun _ => false) (fun _ => true)
    (fun _ => false) (fun _ => true) (fun _ _ => 1) = 0 :=
  cornerSum_eq_zero _ _ _ _ (r := 0) (by simp)
    (mem_orderSpan_of_readsOnly (S := ∅) le_rfl fun _ _ _ _ _ _ => rfl)

/-- **The corner sum of a product over the pairs is the product of the pairs' mixed
differences.** -/
theorem cornerSum_prod (x₀ x₁ : κ → α) (y₀ y₁ : κ → β) (φ : κ → α → β → ℂ) :
    cornerSum x₀ x₁ y₀ y₁ (fun x y => ∏ i, φ i (x i) (y i)) =
      ∏ i, (φ i (x₀ i) (y₀ i) - φ i (x₁ i) (y₀ i) - φ i (x₀ i) (y₁ i) + φ i (x₁ i) (y₁ i)) := by
  have hpair (i : κ) : φ i (x₀ i) (y₀ i) - φ i (x₁ i) (y₀ i) - φ i (x₀ i) (y₁ i) +
      φ i (x₁ i) (y₁ i) = ∑ ab : Bool × Bool, (if ab.1 then (-1 : ℂ) else 1) *
        (if ab.2 then (-1 : ℂ) else 1) *
          φ i (if ab.1 then x₁ i else x₀ i) (if ab.2 then y₁ i else y₀ i) := by
    simp [Fintype.sum_prod_type]
    ring
  simp_rw [hpair]
  rw [Fintype.prod_sum, cornerSum_apply]
  let e : (κ ⊕ κ → Bool) ≃ (κ → Bool × Bool) :=
    { toFun := fun ε i => (ε (Sum.inl i), ε (Sum.inr i))
      invFun := fun g => Sum.elim (fun i => (g i).1) fun i => (g i).2
      left_inv := fun ε => by funext o; rcases o with i | i <;> rfl
      right_inv := fun g => rfl }
  refine Fintype.sum_equiv e _ _ fun ε => ?_
  simp only [cornerSign, Fintype.prod_sum_type, cornerX, cornerY, ← prod_mul_distrib]
  exact prod_congr rfl fun i _ => rfl

end Transformer.Grokking.FlatSpectrum

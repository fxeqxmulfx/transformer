module

public import Transformer.Grokking.FlatSpectrum.ProductTree

/-!
# A product layer at most doubles the interaction order

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV7. A function of the
operands `x_i`, `y_i` *reads only* a set `S` of them if it does not change when the others do
(`ReadsOnly`), and has *order* at most `r` if it is a sum of functions that read at most `r`
operands each (`orderSpan`). Products add orders (`mul_mem_orderSpan`) and linear readings keep
them (`linear_mem_orderSpan`), so a product layer at most doubles the order of every coordinate
(`IsProductLayer.order`) and `d` layers multiply it by `2^d` (`IsProductNet.order`). The one-hot
features read one operand per coordinate (`features_mem_orderSpan`), so every linear readout of
`d` layers on them has order at most `2^d` (`readout_mem_orderSpan`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {κ α β : Type}

/-- `φ` reads only the operands in `S`: the `x_i` with `inl i ∈ S` and the `y_i` with
`inr i ∈ S`. -/
def ReadsOnly (S : Finset (κ ⊕ κ)) (φ : (κ → α) → (κ → β) → ℂ) : Prop :=
  ∀ x x' y y', (∀ i, Sum.inl i ∈ S → x i = x' i) → (∀ i, Sum.inr i ∈ S → y i = y' i) →
    φ x y = φ x' y'

/-- The functions of order at most `r`: sums of functions that read at most `r` operands each. -/
def orderSpan (r : ℕ) : Submodule ℂ ((κ → α) → (κ → β) → ℂ) :=
  Submodule.span ℂ {φ | ∃ S : Finset (κ ⊕ κ), S.card ≤ r ∧ ReadsOnly S φ}

lemma mem_orderSpan_of_readsOnly {r : ℕ} {S : Finset (κ ⊕ κ)} (hS : S.card ≤ r)
    {φ : (κ → α) → (κ → β) → ℂ} (h : ReadsOnly S φ) : φ ∈ orderSpan r :=
  Submodule.subset_span ⟨S, hS, h⟩

lemma orderSpan_mono {r s : ℕ} (h : r ≤ s) :
    (orderSpan r : Submodule ℂ ((κ → α) → (κ → β) → ℂ)) ≤ orderSpan s :=
  Submodule.span_mono fun _ ⟨S, hS, hφ⟩ => ⟨S, hS.trans h, hφ⟩

example : (fun _ _ => 1 : (Unit → Bool) → (Unit → Bool) → ℂ) ∈ orderSpan 1 :=
  orderSpan_mono zero_le_one (mem_orderSpan_of_readsOnly (S := ∅) le_rfl fun _ _ _ _ _ _ => rfl)

/-- **Products add orders.** -/
theorem mul_mem_orderSpan {r s : ℕ} {φ ψ : (κ → α) → (κ → β) → ℂ} (hφ : φ ∈ orderSpan r)
    (hψ : ψ ∈ orderSpan s) : φ * ψ ∈ orderSpan (r + s) := by
  classical
  have h := Submodule.mul_mem_mul hφ hψ
  rw [orderSpan, orderSpan, Submodule.span_mul_span] at h
  refine Submodule.span_le.2 ?_ h
  rintro _ ⟨φ, ⟨S, hS, hφ⟩, ψ, ⟨T, hT, hψ⟩, rfl⟩
  refine Submodule.subset_span ⟨S ∪ T, (card_union_le S T).trans (add_le_add hS hT),
    fun x x' y y' hx hy => ?_⟩
  simp only [Pi.mul_apply]
  rw [hφ x x' y y' (fun i hi => hx i (mem_union_left _ hi)) fun i hi => hy i (mem_union_left _ hi),
    hψ x x' y y' (fun i hi => hx i (mem_union_right _ hi)) fun i hi => hy i (mem_union_right _ hi)]

/-- A linear reading of a state whose coordinates have order at most `r` has order at most
`r`. -/
lemma linear_mem_orderSpan {ι : Type*} [Fintype ι] {r : ℕ} {Z : (κ → α) → (κ → β) → ι → ℂ}
    (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan r) (ℓ : (ι → ℂ) →ₗ[ℂ] ℂ) :
    (fun x y => ℓ (Z x y)) ∈ orderSpan r := by
  classical
  have h : (fun x y => ℓ (Z x y)) =
      ∑ c, ℓ (fun j => if c = j then 1 else 0) • fun x y => Z x y c := by
    funext x y
    rw [LinearMap.pi_apply_eq_sum_univ ℓ (Z x y)]
    simp [Finset.sum_apply, mul_comm]
  rw [h]
  exact Submodule.sum_mem _ fun c _ => Submodule.smul_mem _ _ (hZ c)

/-- **A product layer at most doubles the order** of every coordinate. -/
theorem IsProductLayer.order {ι : Type*} [Fintype ι] {m r : ℕ} {G : (ι → ℂ) → ι → ℂ}
    (hG : IsProductLayer m G) {Z : (κ → α) → (κ → β) → ι → ℂ}
    (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan r) (c : ι) :
    (fun x y => G (Z x y) c) ∈ orderSpan (2 * r) := by
  obtain ⟨η, _, -, g, w, o, hG⟩ := hG
  have h : (fun x y => G (Z x y) c) = (fun x y => Z x y c) +
      ∑ u, o u c • ((fun x y => g u (Z x y)) * fun x y => w u (Z x y)) := by
    funext x y
    simp [hG, Finset.sum_apply, mul_comm]
  rw [h, two_mul]
  refine Submodule.add_mem _ (orderSpan_mono (Nat.le_add_right r r) (hZ c))
    (Submodule.sum_mem _ fun u _ => Submodule.smul_mem _ _ ?_)
  exact mul_mem_orderSpan (linear_mem_orderSpan hZ (g u)) (linear_mem_orderSpan hZ (w u))

/-- **`d` product layers multiply the order by at most `2^d`.** -/
theorem IsProductNet.order {ι : Type*} [Fintype ι] {m d r : ℕ} {F : (ι → ℂ) → ι → ℂ}
    (hF : IsProductNet m d F) {Z : (κ → α) → (κ → β) → ι → ℂ}
    (hZ : ∀ c, (fun x y => Z x y c) ∈ orderSpan r) :
    ∀ c, (fun x y => F (Z x y) c) ∈ orderSpan (2 ^ d * r) := by
  induction hF with
  | zero => simpa using hZ
  | @succ d F G _ hG ih =>
    intro c
    have e : 2 ^ (d + 1) * r = 2 * (2 ^ d * r) := by ring
    rw [e]
    exact hG.order (Z := fun x y => F (Z x y)) ih c

variable [DecidableEq α] [DecidableEq β]

/-- The one-hot features of the pairs, and the zero scratch, read one operand per coordinate. -/
lemma features_mem_orderSpan {T : Type} (c : ((κ × α) ⊕ (κ × β)) ⊕ T) :
    (fun x y => (Sum.elim (pairFeatures x y) 0 : ((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) c) ∈
      orderSpan 1 := by
  rcases c with (⟨i, a⟩ | ⟨i, b⟩) | t
  · refine mem_orderSpan_of_readsOnly (S := {Sum.inl i}) (by simp) fun x x' y y' hx _ => ?_
    simp [pairFeatures, oneHot, hx i (mem_singleton_self _)]
  · refine mem_orderSpan_of_readsOnly (S := {Sum.inr i}) (by simp) fun x x' y y' _ hy => ?_
    simp [pairFeatures, oneHot, hy i (mem_singleton_self _)]
  · exact mem_orderSpan_of_readsOnly (S := ∅) (by simp) fun _ _ _ _ _ _ => rfl

/-- **Every linear readout of `d` product layers on the features has order at most `2^d`.** -/
theorem readout_mem_orderSpan [Fintype κ] [Fintype α] [Fintype β] {T : Type} [Fintype T]
    {m d : ℕ} {F : (((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) → ((κ × α) ⊕ (κ × β)) ⊕ T → ℂ}
    (hF : IsProductNet m d F) (ℓ : (((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) →ₗ[ℂ] ℂ) :
    (fun x y => ℓ (F (Sum.elim (pairFeatures x y) 0))) ∈ orderSpan (2 ^ d) := by
  have h := hF.order (Z := fun x y => Sum.elim (pairFeatures x y) 0) features_mem_orderSpan
  rw [mul_one] at h
  exact linear_mem_orderSpan h ℓ

example := readout_mem_orderSpan (κ := Unit) (α := Bool) (β := Bool) (T := Unit)
  (IsProductNet.id_net 0 2) (LinearMap.proj (Sum.inr ()))

example :
    ((fun _ _ => 1) * (fun _ _ => 1) : (Unit → Bool) → (Unit → Bool) → ℂ) ∈ orderSpan (0 + 0) :=
  mul_mem_orderSpan (mem_orderSpan_of_readsOnly (S := ∅) le_rfl fun _ _ _ _ _ _ => rfl)
    (mem_orderSpan_of_readsOnly (S := ∅) le_rfl fun _ _ _ _ _ _ => rfl)

example := linear_mem_orderSpan (Z := fun x y =>
  (Sum.elim (pairFeatures x y) 0 : ((Unit × Bool) ⊕ (Unit × Bool)) ⊕ Unit → ℂ))
  features_mem_orderSpan (LinearMap.proj (Sum.inr ()))

example := (isProductLayer_id 0).order (Z := fun x y =>
  (Sum.elim (pairFeatures x y) 0 : ((Unit × Bool) ⊕ (Unit × Bool)) ⊕ Unit → ℂ))
  features_mem_orderSpan (Sum.inr ())

example := (IsProductNet.id_net 0 1).order (Z := fun x y =>
  (Sum.elim (pairFeatures x y) 0 : ((Unit × Bool) ⊕ (Unit × Bool)) ⊕ Unit → ℂ))
  features_mem_orderSpan

end Transformer.Grokking.FlatSpectrum

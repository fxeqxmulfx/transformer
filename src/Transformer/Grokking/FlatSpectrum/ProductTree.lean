module

public import Transformer.Grokking.FlatSpectrum.ProductLayer

/-!
# A product tree on the one-hot features of pairs

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV7. On a scratch stream of
fewer than `2^{L+1}` coordinates, `L` product layers of at most `2^L` units multiply `2^L`
leaves: the two subtrees run side by side, and one unit multiplies their roots
(`exists_productTree`). In front of it, one layer writes the leaves from the one-hot features of
the pairs (`pairFeatures`): a leaf `∑_u G_u(x) H_u(y)` of a pair `(x, y)` takes `|U|` units, each
multiplying a reading `G_u(x)` of the features of `x` by a reading `H_u(y)` of those of `y`
(`exists_productNet`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

/-- **A product tree**: on a scratch stream `T` of fewer than `2^{L+1}` coordinates, `L` product
layers of at most `2^L` hidden units write the product of the `2^L` leaves into the root, from
any values at the leaves and zeros elsewhere. -/
theorem exists_productTree (L : ℕ) :
    ∃ (Λ T : Type) (_ : Fintype Λ) (_ : Fintype T) (leaf : Λ → T) (root : T)
      (F : (T → ℂ) → T → ℂ), Fintype.card Λ = 2 ^ L ∧ Fintype.card T < 2 ^ (L + 1) ∧
      Function.Injective leaf ∧ IsProductNet (2 ^ L) L F ∧
      ∀ z : T → ℂ, (∀ n, (∀ l, leaf l ≠ n) → z n = 0) → F z root = ∏ l, z (leaf l) := by
  induction L with
  | zero =>
    exact ⟨Unit, Unit, inferInstance, inferInstance, id, (), id, by simp, by simp,
      Function.injective_id, IsProductNet.zero, fun z _ => by simp⟩
  | succ L ih =>
    obtain ⟨Λ, T, _, _, leaf, root, F, hΛ, hT, hinj, hF, hval⟩ := ih
    let leaf' : Λ ⊕ Λ → (T ⊕ T) ⊕ Unit :=
      Sum.elim (fun l => Sum.inl (Sum.inl (leaf l))) (fun l => Sum.inl (Sum.inr (leaf l)))
    let o : (T ⊕ T) ⊕ Unit → ℂ := Sum.elim (fun _ => 0) (fun _ => 1)
    let G : ((T ⊕ T) ⊕ Unit → ℂ) → (T ⊕ T) ⊕ Unit → ℂ := fun z =>
      z + (z (Sum.inl (Sum.inl root)) * z (Sum.inl (Sum.inr root))) • o
    have hG : IsProductLayer (2 ^ (L + 1)) G :=
      ⟨Unit, inferInstance, by simpa using Nat.one_le_two_pow,
        fun _ => LinearMap.proj (Sum.inl (Sum.inl root)),
        fun _ => LinearMap.proj (Sum.inl (Sum.inr root)), fun _ => o,
        fun z => by simp [G]⟩
    have hnet := ((hF.sideBySide hF).sideBySide (IsProductNet.id_net (ι := Unit) 0 L)).mono
      (show 2 ^ L + 2 ^ L + 0 ≤ 2 ^ (L + 1) by rw [pow_succ]; omega)
    refine ⟨Λ ⊕ Λ, (T ⊕ T) ⊕ Unit, inferInstance, inferInstance, leaf', Sum.inr (),
      G ∘ sideBySide (sideBySide F F) id, by rw [Fintype.card_sum, hΛ, pow_succ]; omega,
      by rw [Fintype.card_sum, Fintype.card_sum, Fintype.card_unit, pow_succ]; omega,
      ?_, hnet.succ hG, fun z hz => ?_⟩
    · rintro (l | l) (l' | l') h <;> simp [leaf', hinj.eq_iff] at h ⊢ <;> exact h
    · have hL (n : T) (hn : ∀ l, leaf l ≠ n) : z (Sum.inl (Sum.inl n)) = 0 :=
        hz _ (by rintro (l | l) h <;> simp [leaf'] at h; exact hn l h)
      have hR (n : T) (hn : ∀ l, leaf l ≠ n) : z (Sum.inl (Sum.inr n)) = 0 :=
        hz _ (by rintro (l | l) h <;> simp [leaf'] at h; exact hn l h)
      have w0 : sideBySide (sideBySide F F) id z (Sum.inr ()) = 0 :=
        hz _ (by rintro (l | l) h <;> simp [leaf'] at h)
      have w1 : sideBySide (sideBySide F F) id z (Sum.inl (Sum.inl root)) =
          F (fun n => z (Sum.inl (Sum.inl n))) root := rfl
      have w2 : sideBySide (sideBySide F F) id z (Sum.inl (Sum.inr root)) =
          F (fun n => z (Sum.inl (Sum.inr n))) root := rfl
      show G (sideBySide (sideBySide F F) id z) (Sum.inr ()) = _
      simp only [G, o, Pi.add_apply, Pi.smul_apply, Sum.elim_inr, smul_eq_mul,
        mul_one, w0, w1, w2, hval _ hL, hval _ hR, zero_add, Fintype.prod_sum_type, leaf',
        Sum.elim_inl]

variable {κ α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]

/-- The one-hot vector of `x : κ → α`: coordinate `(i, a)` is `1` if `x_i = a` and `0` otherwise. -/
def oneHot (x : κ → α) : κ × α → ℂ := fun ia => if x ia.1 = ia.2 then 1 else 0

/-- The features of the pairs `(x_i, y_i)`, `i ∈ κ`: a one-hot vector of each operand. -/
def pairFeatures (x : κ → α) (y : κ → β) : (κ × α) ⊕ (κ × β) → ℂ := Sum.elim (oneHot x) (oneHot y)

/-- **A product network on the features of the pairs**: for any assignment `π` of the `2^L`
leaves to pairs and any values `∑_u G_{j,u}(x) H_{j,u}(y)` of the leaves, `1 + L` product
layers with at most `2^L |U|` hidden units each, on the features and a scratch stream `T` of
fewer than `2^{L+1}` coordinates that starts at zero, write the product of the leaves into
one coordinate. -/
theorem exists_productNet (L : ℕ) (U : Type) [Fintype U] [Nonempty U] (π : Fin (2 ^ L) → κ)
    (G : Fin (2 ^ L) → U → α → ℂ) (H : Fin (2 ^ L) → U → β → ℂ) :
    ∃ (T : Type) (_ : Fintype T) (root : T)
      (F : (((κ × α) ⊕ (κ × β)) ⊕ T → ℂ) → ((κ × α) ⊕ (κ × β)) ⊕ T → ℂ),
      Fintype.card T < 2 ^ (L + 1) ∧ IsProductNet (2 ^ L * Fintype.card U) (1 + L) F ∧
      ∀ x y, F (Sum.elim (pairFeatures x y) 0) (Sum.inr root) =
        ∏ j, ∑ u, G j u (x (π j)) * H j u (y (π j)) := by
  classical
  obtain ⟨Λ, T, _, _, leaf, root, Ft, hΛ, hT, hinj, hFt, hval⟩ := exists_productTree L
  let e : Λ ≃ Fin (2 ^ L) := Fintype.equivFinOfCardEq hΛ
  let S := ((κ × α) ⊕ (κ × β)) ⊕ T
  let gate : Λ × U → (S → ℂ) →ₗ[ℂ] ℂ := fun lu =>
    ∑ a, G (e lu.1) lu.2 a • LinearMap.proj (Sum.inl (Sum.inl (π (e lu.1), a)))
  let up : Λ × U → (S → ℂ) →ₗ[ℂ] ℂ := fun lu =>
    ∑ b, H (e lu.1) lu.2 b • LinearMap.proj (Sum.inl (Sum.inr (π (e lu.1), b)))
  let out : Λ × U → S → ℂ := fun lu => Pi.single (Sum.inr (leaf lu.1)) 1
  let W : (S → ℂ) → S → ℂ := fun z => z + ∑ lu, (gate lu z * up lu z) • out lu
  have hW : IsProductLayer (2 ^ L * Fintype.card U) W :=
    ⟨Λ × U, inferInstance, by rw [Fintype.card_prod, hΛ], gate, up, out, fun z => rfl⟩
  refine ⟨T, inferInstance, root, sideBySide id Ft ∘ W, hT, ?_, fun x y => ?_⟩
  · have h1 : 2 ^ L ≤ 2 ^ L * Fintype.card U := Nat.le_mul_of_pos_right _ Fintype.card_pos
    exact (IsProductNet.of_layer hW).comp
      (((IsProductNet.id_net 0 L).sideBySide hFt).mono (by omega))
  · have hg (lu : Λ × U) : gate lu (Sum.elim (pairFeatures x y) 0) =
        G (e lu.1) lu.2 (x (π (e lu.1))) := by
      simp [gate, pairFeatures, oneHot]
    have hu (lu : Λ × U) : up lu (Sum.elim (pairFeatures x y) 0) =
        H (e lu.1) lu.2 (y (π (e lu.1))) := by
      simp [up, pairFeatures, oneHot]
    have hWz (n : T) : W (Sum.elim (pairFeatures x y) 0) (Sum.inr n) =
        ∑ l, if leaf l = n then
          ∑ u, G (e l) u (x (π (e l))) * H (e l) u (y (π (e l))) else 0 := by
      simp only [W, out, Pi.add_apply, Sum.elim_inr, Pi.zero_apply, zero_add, Finset.sum_apply,
        Pi.smul_apply, smul_eq_mul, hg, hu, Fintype.sum_prod_type]
      refine Finset.sum_congr rfl fun l _ => ?_
      by_cases h : leaf l = n
      · subst h
        simp
      · simp [h, Pi.single_eq_of_ne (fun h' => h (Sum.inr_injective h').symm)]
    have hz (n : T) (hn : ∀ l, leaf l ≠ n) :
        (W (Sum.elim (pairFeatures x y) 0) ∘ Sum.inr) n = 0 := by
      simp [hWz, hn]
    change Ft (W (Sum.elim (pairFeatures x y) 0) ∘ Sum.inr) root = _
    rw [hval _ hz]
    exact Fintype.prod_equiv e _ _ fun l => by simp [hWz, hinj.eq_iff]

example := exists_productNet (κ := Fin 2) (α := Bool) (β := Bool) 1 Unit (fun _ => 0)
  (fun _ _ _ => 1) (fun _ _ _ => 1)

end Transformer.Grokking.FlatSpectrum

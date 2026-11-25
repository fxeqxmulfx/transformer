module

public import Transformer.GainMatroid.Section2_ExchangeIndep

/-!
# The matroids of an exchange pair differ by `Δ_X`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4, toward Proposition 2.8. For biased graphs `Ω`
on `T` and `Ω'` on `S` of an exchange pair on `X = {a, b, c}` whose balanced cycles agree across the
exchange (`BiasedGraph.BalanceAgree`), the cycles of `T \ X` and of `T \ c` are balanced through
the embeddings into `T`, `S` and `S / c` alike (`IsExchange.sEmbedding₀_balanced_iff_tEmbedding`,
`IsExchange.sEmbedding_balanced_iff_tEmbedding`). So frame and lift independence agree off `X`
(`IsExchange.isFrameIndep_iff_of_disjoint`, `IsExchange.isLiftIndep_iff_of_disjoint`), and for `I`
off `c`, `I` with `a`, `b` exchanged and `c` added is independent in `Ω'` exactly when `I` is in `Ω`
(`IsExchange.isFrameIndep_insert_swap_iff`, `IsExchange.isLiftIndep_insert_swap_iff`).

When `X` is balanced in `Ω`, this is the description of `Δ_X` by independent sets
(`isDeltaMatroid_of_indep_iff`), with the labels of `X` permuted for `a` and `b`: the frame and
lift matroids of `Ω'` are `Δ_X` of those of `Ω` (`IsExchange.isDeltaMatroid_frame`,
`IsExchange.isDeltaMatroid_lift`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph.IsExchange

open BiasedGraph

variable {V W E : Type*} {ι : V → W} {v : W} {a b c : E} {x y z : V}
  {Ω : BiasedGraph V E} {Ω' : BiasedGraph W E}

/-- A cycle of `T \ X` is balanced through `S` exactly when it is through `T`. -/
theorem sEmbedding₀_balanced_iff_tEmbedding (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z)
    (hB : Ω.BalanceAgree Ω' {a, b, c}) {C : Set ↥({a, b, c} : Set E)ᶜ}
    (hC : (h.offGraph _).IsCycle C) : h.sEmbedding₀.Balanced C ↔ (h.tEmbedding _).Balanced C := by
  rw [h.sEmbedding₀_balanced_iff, h.tEmbedding_balanced_iff]
  refine (hB.zero _ (h.isCycle_val hC) (Set.eq_empty_of_forall_notMem fun e he => ?_)).symm
  obtain ⟨⟨e', he', rfl⟩, heX⟩ := he
  exact e'.prop heX

/-- A set off `X` is its own image from `T \ X`. -/
theorem image_preimage_of_disjoint {I : Set E} (hI : Disjoint I {a, b, c}) :
    Subtype.val '' (Subtype.val ⁻¹' I : Set ↥({a, b, c} : Set E)ᶜ) = I := by
  rw [Subtype.image_preimage_coe]
  exact Set.inter_eq_right.2 fun e he heX => Set.disjoint_left.1 hI he heX

/-- A set off `c` is its own image from `T \ c`. -/
theorem image_preimage_of_notMem {I : Set E} (hcI : c ∉ I) :
    Subtype.val '' (Subtype.val ⁻¹' I : Set ↥({c} : Set E)ᶜ) = I := by
  rw [Subtype.image_preimage_coe]
  exact Set.inter_eq_right.2 fun e he hc => hcI (Set.mem_singleton_iff.1 hc ▸ he)

variable [DecidableEq E]

/-- A cycle of `T \ c` is balanced through `S / c` exactly when it is through `T`. -/
theorem sEmbedding_balanced_iff_tEmbedding (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z)
    (hB : Ω.BalanceAgree Ω' {a, b, c}) {C : Set ↥({c} : Set E)ᶜ} (hC : (h.offGraph _).IsCycle C) :
    h.sEmbedding.Balanced C ↔ (h.tEmbedding _).Balanced C := by
  rw [h.tEmbedding_balanced_iff, ContractEmbedding.Balanced, h.sEmbedding_image]
  exact h.exists_swap_mem_iff hB (h.isCycle_val hC) fun ⟨e, _, he⟩ =>
    e.prop (Set.mem_singleton_iff.2 he)

variable [Finite V] [Finite W] [Finite E]

omit [DecidableEq E] in
/-- Off `X`, frame independence agrees across the exchange. -/
theorem isFrameIndep_iff_of_disjoint (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z)
    (hB : Ω.BalanceAgree Ω' {a, b, c}) {I : Set E} (hI : Disjoint I {a, b, c}) :
    Ω'.IsFrameIndep I ↔ Ω.IsFrameIndep I := by
  have := h.sEmbedding₀.isFrameIndep_congr (h.tEmbedding _)
    (fun _ hC => h.sEmbedding₀_balanced_iff_tEmbedding hB hC) (Subtype.val ⁻¹' I)
  rwa [h.sEmbedding₀_carry, h.tEmbedding_carry, image_preimage_of_disjoint hI] at this

omit [DecidableEq E] in
/-- Off `X`, lift independence agrees across the exchange. -/
theorem isLiftIndep_iff_of_disjoint (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z)
    (hB : Ω.BalanceAgree Ω' {a, b, c}) {I : Set E} (hI : Disjoint I {a, b, c}) :
    Ω'.IsLiftIndep I ↔ Ω.IsLiftIndep I := by
  have := h.sEmbedding₀.isLiftIndep_congr (h.tEmbedding _)
    (fun _ hC => h.sEmbedding₀_balanced_iff_tEmbedding hB hC) (Subtype.val ⁻¹' I)
  rwa [h.sEmbedding₀_carry, h.tEmbedding_carry, image_preimage_of_disjoint hI] at this

/-- **Frame independence through `c`.** For `I` off `c`, `I` with `a` and `b` exchanged and `c`
added is frame-independent in `Ω'` exactly when `I` is frame-independent in `Ω`. -/
theorem isFrameIndep_insert_swap_iff (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z)
    (hB : Ω.BalanceAgree Ω' {a, b, c}) {I : Set E} (hcI : c ∉ I) :
    Ω'.IsFrameIndep (insert c (Equiv.swap a b '' I)) ↔ Ω.IsFrameIndep I := by
  have := h.sEmbedding.isFrameIndep_congr (h.tEmbedding _)
    (fun _ hC => h.sEmbedding_balanced_iff_tEmbedding hB hC) (Subtype.val ⁻¹' I)
  rwa [h.sEmbedding_carry, h.tEmbedding_carry, image_preimage_of_notMem hcI] at this

/-- **Lift independence through `c`.** For `I` off `c`, `I` with `a` and `b` exchanged and `c`
added is lift-independent in `Ω'` exactly when `I` is lift-independent in `Ω`. -/
theorem isLiftIndep_insert_swap_iff (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z)
    (hB : Ω.BalanceAgree Ω' {a, b, c}) {I : Set E} (hcI : c ∉ I) :
    Ω'.IsLiftIndep (insert c (Equiv.swap a b '' I)) ↔ Ω.IsLiftIndep I := by
  have := h.sEmbedding.isLiftIndep_congr (h.tEmbedding _)
    (fun _ hC => h.sEmbedding_balanced_iff_tEmbedding hB hC) (Subtype.val ⁻¹' I)
  rwa [h.sEmbedding_carry, h.tEmbedding_carry, image_preimage_of_notMem hcI] at this

/-- **The frame matroids of an exchange pair differ by `Δ_X`** when `X` is balanced in `Ω`. -/
theorem isDeltaMatroid_frame (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z)
    (hB : Ω.BalanceAgree Ω' {a, b, c}) (hbal : ({a, b, c} : Set E) ∈ Ω.balanced)
    {M M' : Matroid E} (hM : Ω.IsFrameMatroid M) (hM' : Ω'.IsFrameMatroid M') :
    IsDeltaMatroid M a b c M' := by
  refine isDeltaMatroid_of_indep_iff ((hM.2 _).2 (Or.inl hbal)) h.triangle.ne_ab h.triangle.ne_ac
    h.triangle.ne_bc (by rw [hM.1, hM'.1]) (fun I hI => ?_) (fun I hI => ?_) (fun I hI => ?_)
    (fun I hI => ?_) <;> rw [hM'.indep_iff, hM.indep_iff]
  · exact h.isFrameIndep_iff_of_disjoint hB hI
  · exact h.isFrameIndep_insert_swap_iff hB hI
  · exact h.swap₂₃.isFrameIndep_insert_swap_iff (by rwa [triple_acb_eq]) hI
  · exact h.swap₁₂.swap₂₃.isFrameIndep_insert_swap_iff (by rwa [triple_bca_eq]) hI

/-- **The lift matroids of an exchange pair differ by `Δ_X`** when `X` is balanced in `Ω`. -/
theorem isDeltaMatroid_lift (h : IsExchange Ω.graph Ω'.graph ι v a b c x y z)
    (hB : Ω.BalanceAgree Ω' {a, b, c}) (hbal : ({a, b, c} : Set E) ∈ Ω.balanced)
    {M M' : Matroid E} (hM : Ω.IsLiftMatroid M) (hM' : Ω'.IsLiftMatroid M') :
    IsDeltaMatroid M a b c M' := by
  refine isDeltaMatroid_of_indep_iff ((hM.2 _).2 (Or.inl hbal)) h.triangle.ne_ab h.triangle.ne_ac
    h.triangle.ne_bc (by rw [hM.1, hM'.1]) (fun I hI => ?_) (fun I hI => ?_) (fun I hI => ?_)
    (fun I hI => ?_) <;> rw [hM'.indep_iff, hM.indep_iff]
  · exact h.isLiftIndep_iff_of_disjoint hB hI
  · exact h.isLiftIndep_insert_swap_iff hB hI
  · exact h.swap₂₃.isLiftIndep_insert_swap_iff (by rwa [triple_acb_eq]) hI
  · exact h.swap₁₂.swap₂₃.isLiftIndep_insert_swap_iff (by rwa [triple_bca_eq]) hI

end OrientedGraph.IsExchange

end Transformer.GainMatroid

module

public import Transformer.GainMatroid.Section2_ExchangeEmbeddings

/-!
# `T \ c` carried into `S / c`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.4. In an exchange pair on `X = {a, b, c}`,
contracting `c = vz` in `S` merges `v` with `z` (`IsExchange.class_v_eq`), so `a = vx` comes to
join `x` and `z`, as `b` does in `T`, and `b = vy` comes to join `y` and `z`, as `a` does in `T`;
the other edges keep their ends. So an edge `e ≠ c` of `T` joins, under the exchange of `a` and `b`,
the classes of its ends in `S / c` (`IsExchange.joins_swap`), and `T \ c`, on the vertices of `T`
not sent to `v`, is carried into `S / c` (`IsExchange.sEmbedding`): a set `I` of its edges becomes
`I` with `a` and `b` exchanged, with `c` added (`IsExchange.sEmbedding_carry`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph.IsExchange

open BiasedGraph

variable {V W E : Type*} {T : OrientedGraph V E} {S : OrientedGraph W E} {ι : V → W} {v : W}
  {a b c : E} {x y z : V} [DecidableEq E]

/-- In `S / c`, an edge of `T` other than `c` joins, under the exchange of `a` and `b`, the classes
of its ends. -/
theorem joins_swap {Ω : BiasedGraph W E} (h : IsExchange T Ω.graph ι v a b c x y z)
    (e e' : ↥({c} : Set E)ᶜ) (he' : e'.val = Equiv.swap a b e.val) :
    (Ω.graph.contract {c}).Joins e' (Quotient.mk _ (ι (T.tail e.val)))
      (Quotient.mk _ (ι (T.head e.val))) := by
  have hred : ∀ {s t : V}, T.Joins e.val s t →
      (Ω.graph.contract {c}).Joins e' (Quotient.mk _ (ι s)) (Quotient.mk _ (ι t)) →
      (Ω.graph.contract {c}).Joins e' (Quotient.mk _ (ι (T.tail e.val)))
        (Quotient.mk _ (ι (T.head e.val))) := by
    rintro s t (⟨h₁, h₂⟩ | ⟨h₁, h₂⟩) hj
    · rw [h₁, h₂]
      exact hj
    · rw [h₁, h₂]
      exact hj.symm
  by_cases hea : e.val = a
  · have hj : T.Joins e.val y z := by
      rw [hea]
      exact h.triangle.joins_a
    refine hred hj ?_
    have := Joins.contract_mk h.joins_b e' (by rw [he', hea, Equiv.swap_apply_left])
    rw [h.class_v_eq] at this
    exact this.symm
  by_cases heb : e.val = b
  · have hj : T.Joins e.val x z := by
      rw [heb]
      exact h.triangle.joins_b
    refine hred hj ?_
    have := Joins.contract_mk h.joins_a e' (by rw [he', heb, Equiv.swap_apply_right])
    rw [h.class_v_eq] at this
    exact this.symm
  have heX : e.val ∉ ({a, b, c} : Set E) := by
    rintro (h' | h' | h')
    exacts [hea h', heb h', e.prop h']
  exact Joins.contract_mk (Or.inl (h.agree e.val heX)) e'
    (he'.trans (Equiv.swap_apply_of_ne_of_ne hea heb))

/-- `offGraph` off `c` carried into `S / c`, exchanging `a` and `b`. -/
def sEmbedding {Ω : BiasedGraph W E} (h : IsExchange T Ω.graph ι v a b c x y z) :
    Ω.ContractEmbedding (h.offGraph ({c} : Set E)ᶜ) where
  contractSet := {c}
  vertex u := Quotient.mk _ (ι u.val)
  edge e := ⟨Equiv.swap a b e.val, swap_ne h.triangle.ne_ac h.triangle.ne_bc e.prop⟩
  isForest := h.isForest_c
  vertex_injective u w huw := by
    refine Subtype.ext (h.inj (Classical.byContradiction fun hne => ?_))
    obtain ⟨hu, hw⟩ := ReachableIn.mem_verts_of_ne (Quotient.exact huw) hne
    exact hne ((h.eq_of_mem_verts_c hu u.prop).trans (h.eq_of_mem_verts_c hw w.prop).symm)
  edge_injective _ _ hef := Subtype.ext ((Equiv.swap a b).injective (congrArg Subtype.val hef))
  joins e := h.joins_swap e _ rfl

theorem sEmbedding_image {Ω : BiasedGraph W E} (h : IsExchange T Ω.graph ι v a b c x y z)
    (C : Set ↥({c} : Set E)ᶜ) :
    Subtype.val '' (h.sEmbedding.edge '' C) = Equiv.swap a b '' (Subtype.val '' C) := by
  rw [Set.image_image, Set.image_image]
  rfl

theorem sEmbedding_carry {Ω : BiasedGraph W E} (h : IsExchange T Ω.graph ι v a b c x y z)
    (I : Set ↥({c} : Set E)ᶜ) :
    h.sEmbedding.carry I = insert c (Equiv.swap a b '' (Subtype.val '' I)) := by
  rw [ContractEmbedding.carry, h.sEmbedding_image]
  exact Set.union_singleton

end OrientedGraph.IsExchange

end Transformer.GainMatroid

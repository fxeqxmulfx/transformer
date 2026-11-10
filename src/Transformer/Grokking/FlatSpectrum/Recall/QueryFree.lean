module

public import Transformer.Grokking.FlatSpectrum.Recall.Softmax
public import Transformer.Grokking.FlatSpectrum.Recall.Sign

/-!
# Softmax attention whose scores ignore the query stays within the cut

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA4, for
the attention of arXiv:2506.16055v3, §2.1 and Appendix `app:transformers`, Equation `eq:att`.
The transformer that its `thm:TLCl_to_rtfr` builds from a formula of `TL[◁#]` uses uniform
attention: no score depends on the query (`Transformer.CRASP.RTfr.queryFree_model`).

At the query, such an attention averages the tokens of the context and the query's own token
with weights that are functions of the context alone and of the query alone:

    att_r = (N_r(x) + α_r(q)) / (D(x) + δ(q)),        D(x) + δ(q) > 0.

A readout `g q + ∑_{r < d} w_r(q) att_r` has the sign of
`g(q) (D(x) + δ(q)) + ∑_r w_r(q) (N_r(x) + α_r(q))`, of cut width `d + 1`: the `d` sums `N_r` and
the normalization `D` cross the cut. So read by its sign it recalls at most `d + 1` bindings
(`card_le_of_normalized_sign`, `card_le_of_queryFreeAttention_sign`): the exponential leaves the
cut only through scores that pair the query with the key (`Recall.Softmax`).
`Transformer.CRASP.two_pow_le_of_queryFree` counts bits instead, for every readout of the
rounded transformer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset

variable {κ : Type*} [Fintype κ]

/-- **A normalized readout read by its sign** (`review/flat_spectrum.md`, SA4): if
`g q + (∑_{r < d} a_r(q) N_r(π) + α q) / (D π + δ q)`, with a positive denominator, is positive
exactly when the queried key is bound to `true`, then `m ≤ d + 1`. -/
theorem card_le_of_normalized_sign {d : ℕ} {g α δ : κ → ℝ} {a : κ → Fin d → ℝ}
    {N : (κ → Bool) → Fin d → ℝ} {D : (κ → Bool) → ℝ} (hD : ∀ q π, 0 < D π + δ q)
    (h : ∀ π q, 0 < g q + (∑ r, a q r * N π r + α q) / (D π + δ q) ↔ π q = true) :
    Fintype.card κ ≤ d + 1 := by
  have hF : HasCutWidth (d + 1) fun q π => g q * (D π + δ q) + (∑ r, a q r * N π r + α q) :=
    ⟨fun q => g q * δ q + α q, fun q => Fin.snoc (α := fun _ => ℝ) (a q) (g q),
      fun π => Fin.snoc (α := fun _ => ℝ) (N π) (D π), fun q π => by
        simp only [Fin.sum_univ_castSucc, Fin.snoc_castSucc, Fin.snoc_last]
        ring⟩
  refine hF.card_le_of_sign fun π q => ?_
  rw [← h π q, show g q + (∑ r, a q r * N π r + α q) / (D π + δ q) =
    (g q * (D π + δ q) + (∑ r, a q r * N π r + α q)) / (D π + δ q) by
      field_simp [(hD q π).ne'], div_pos_iff_of_pos_right (hD q π)]

/-- One key and no sums: the normalization alone tells its value. -/
example : Fintype.card (Fin 1) ≤ 0 + 1 :=
  card_le_of_normalized_sign (g := fun _ => 1) (α := fun _ => -3) (δ := fun _ => 1)
    (a := fun _ _ => 0) (N := fun _ _ => 0) (D := fun π => if π 0 then 3 else 1)
    (fun q π => by split_ifs <;> norm_num)
    fun π q => by
      obtain rfl : q = 0 := Subsingleton.elim _ _
      cases hπ : π 0 <;> norm_num [hπ]

/-- Softmax attention over the context and the query's own token `none`. -/
lemma softmaxAttention_option {ι : Type*} [Fintype ι] (σ u : ℝ) (s v : ι → ℝ) :
    softmaxAttention (fun o : Option ι => o.elim σ s) (fun o => o.elim u v) =
      (∑ j, Real.exp (s j) * v j + Real.exp σ * u) / (∑ j, Real.exp (s j) + Real.exp σ) := by
  simp only [softmaxAttention, Fintype.sum_option, Option.elim_none, Option.elim_some]
  rw [add_comm (Real.exp σ * u), add_comm (Real.exp σ)]

/-- **Query-free softmax attention read by its sign recalls at most `d + 1` bindings**
(`review/flat_spectrum.md`, SA4): the tokens `j` of the context have scores `s π j` and values
`v π j r`, the query's own token has the score `σ q` and the values `u q r`, and the readout is
`g q + ∑_{r < d} w_r(q) att_r`. If it is positive exactly when the queried key is bound to
`true`, then `m ≤ d + 1`. -/
theorem card_le_of_queryFreeAttention_sign {ι : Type*} [Fintype ι] {d : ℕ} (g σ : κ → ℝ)
    (u w : κ → Fin d → ℝ) (s : (κ → Bool) → ι → ℝ) (v : (κ → Bool) → ι → Fin d → ℝ)
    (h : ∀ π q, 0 < g q + ∑ r, w q r * softmaxAttention
      (fun o : Option ι => o.elim (σ q) (s π)) (fun o => o.elim (u q r) fun j => v π j r) ↔
        π q = true) : Fintype.card κ ≤ d + 1 := by
  refine card_le_of_normalized_sign (g := g) (α := fun q => ∑ r, w q r * (Real.exp (σ q) * u q r))
    (δ := fun q => Real.exp (σ q)) (a := w)
    (N := fun π r => ∑ j, Real.exp (s π j) * v π j r) (D := fun π => ∑ j, Real.exp (s π j))
    (fun q π => add_pos_of_nonneg_of_pos (sum_nonneg fun _ _ => (Real.exp_pos _).le)
      (Real.exp_pos _)) fun π q => ?_
  rw [← h π q]
  simp only [softmaxAttention_option, mul_div_assoc', mul_add, ← sum_div, sum_add_distrib]

/-- One key, read from uniform attention over its token and the query's own. -/
example : Fintype.card (Fin 1) ≤ 1 + 1 :=
  card_le_of_queryFreeAttention_sign (ι := Fin 1) (fun _ => 0) (fun _ => 0) (fun _ _ => 0)
    (fun _ _ => 1) (fun _ _ => 0) (fun π j _ => if π j then 1 else -1) fun π q => by
      obtain rfl : q = 0 := Subsingleton.elim _ _
      cases hπ : π 0 <;> simp [softmaxAttention_option, hπ]

end Transformer.Grokking.FlatSpectrum.Recall

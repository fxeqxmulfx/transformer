module

public import Transformer.Grokking.FlatSpectrum.Recall.Sign
public import Transformer.Grokking.FlatSpectrum.Recall.Width
public import Transformer.Zoology.Appendix_RetNet

/-!
# The RetNet readout recalls one binding for each dimension

Source: `review/flat_spectrum.md`, *Associative recall has a flat spectrum* (added), item AR7,
for the RetNet of Arora et al., arXiv:2312.04927v1, Appendix `app:retnet-proof`: the state obeys
`z^n = γ z^{n−1} + A[n,:]ᵀ V[n,:]` (equation `eq: retnet-recurrence`) and the output at the query
is `Out[N,:] = C[N,:] z^N = C[N,:] (γ z^{N−1} + A[N,:]ᵀ V[N,:])`. Its Corollary `cor: space-ar`
reads: "The RetNet model requires Ω(N)-bits to solve AR for d ≤ √N", by counting the bits of the
`d × d` state; `exact_index_requires_bits` is the finite-state form.

The state holds `d²` real numbers, but the query reads it through the `d` numbers `C[N,:]`:
`∑_c w_c Out[N,c] = g(q) + ∑_r C_r(q) · γ ∑_c w_c z^{N−1}[r,c]`, where `z^{N−1}` depends on the
context alone. So every linear readout of the output has cut width `d`
(`hasCutWidth_retNet`), at any precision. With item AR4 and item AR5, for `m` bindings:

* the squared error on recall is at least `(m − d) c^{m−1} ∑_v (φ(v) − μ)²`
  (`retNet_recall_error`);
* exact recall of a non-constant `φ` has `d ≥ m` (`card_le_of_retNet_recall`);
* a correct argmax over logits `∑_c R_{c v} Out[N,c]` has `d ≥ m`
  (`card_le_of_retNet_argmax`).

For `√N < d < N` the count of bits does not exclude recall of `N` bindings; this bound does.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Transformer.Zoology

variable {κ ν : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν]

/-- **The RetNet output at the query has cut width its dimension** (`review/flat_spectrum.md`,
AR7): `ctx` lists the projected pairs `(A[n,:], V[n,:])` of the context, `qry` is the pair of
the query token and `out` its projection `C[N,:]`; `w` is a linear readout of `Out[N,:]`. -/
theorem hasCutWidth_retNet {Q C : Type*} {d : ℕ} (γ : ℝ) (ctx : C → List (ProjectedPair d))
    (qry : Q → ProjectedPair d) (out : Q → Fin d → ℝ) (w : Fin d → ℝ) :
    HasCutWidth d fun q x =>
      ∑ c, w c * ∑ r, out q r * retNetRun γ (ctx x ++ [qry q]) r c := by
  refine ⟨fun q => ∑ c, w c * ∑ r, out q r * ((qry q).1 r * (qry q).2 c), fun q r => out q r,
    fun x r => γ * ∑ c, w c * retNetRun γ (ctx x) r c, fun q x => ?_⟩
  have hcomm : ∑ r, out q r * (γ * ∑ c, w c * retNetRun γ (ctx x) r c) =
      ∑ c, w c * ∑ r, out q r * (γ * retNetRun γ (ctx x) r c) := by
    simp only [Finset.mul_sum]
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun r _ => by ring
  rw [hcomm, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [← mul_add, ← Finset.sum_add_distrib]
  refine congrArg _ (Finset.sum_congr rfl fun r _ => ?_)
  rw [retNetRun_append, retNetStep]
  ring

/-- **The error of the RetNet readout on recall** (`review/flat_spectrum.md`, AR7): at least
`(m − d) c^{m−1} ∑_v (φ(v) − μ)²`, with `μ` the mean of `φ`. -/
theorem retNet_recall_error {d : ℕ} (γ : ℝ) (ctx : (κ → ν) → List (ProjectedPair d))
    (qry : κ → ProjectedPair d) (out : κ → Fin d → ℝ) (w : Fin d → ℝ) {φ : ν → ℝ} {μ : ℝ}
    (hμ : ∑ v, (φ v - μ) = 0) :
    ((Fintype.card κ : ℝ) - d) * ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) *
      ∑ v, (φ v - μ) ^ 2) ≤ ∑ q, ∑ π : κ → ν,
        (φ (π q) - ∑ c, w c * ∑ r, out q r * retNetRun γ (ctx π ++ [qry q]) r c) ^ 2 :=
  (hasCutWidth_retNet γ ctx qry out w).mul_le_sum_sq hμ

/-- **The RetNet readout of dimension `d` recalls at most `d` bindings**
(`review/flat_spectrum.md`, AR7; compare Corollary `cor: space-ar` of the paper): exact recall
of a non-constant `φ` for `m` keys has `d ≥ m`, whatever the precision of the state. -/
theorem card_le_of_retNet_recall {d : ℕ} (γ : ℝ) (ctx : (κ → ν) → List (ProjectedPair d))
    (qry : κ → ProjectedPair d) (out : κ → Fin d → ℝ) (w : Fin d → ℝ) {φ : ν → ℝ} {v v' : ν}
    (hφ : φ v ≠ φ v')
    (h : ∀ q π, ∑ c, w c * ∑ r, out q r * retNetRun γ (ctx π ++ [qry q]) r c = φ (π q)) :
    Fintype.card κ ≤ d :=
  (hasCutWidth_retNet γ ctx qry out w).card_le_of_recall hφ h

omit [DecidableEq κ] [Fintype ν] in
/-- **The RetNet readout of dimension `d` classifies at most `d` bindings**
(`review/flat_spectrum.md`, AR7): if the largest of the logits `∑_c R_{c v} Out[N,c]` is always
that of the bound value, then `d ≥ m`. -/
theorem card_le_of_retNet_argmax {d : ℕ} (γ : ℝ) (ctx : (κ → ν) → List (ProjectedPair d))
    (qry : κ → ProjectedPair d) (out : κ → Fin d → ℝ) (R : Fin d → ν → ℝ) {v₀ v₁ : ν}
    (hv : v₀ ≠ v₁)
    (h : ∀ π q v, v ≠ π q →
      ∑ c, R c v * ∑ r, out q r * retNetRun γ (ctx π ++ [qry q]) r c <
        ∑ c, R c (π q) * ∑ r, out q r * retNetRun γ (ctx π ++ [qry q]) r c) :
    Fintype.card κ ≤ d := by
  refine card_le_of_argmax (L := fun v q π =>
    ∑ c, R c v * ∑ r, out q r * retNetRun γ (ctx π ++ [qry q]) r c) hv ?_ h
  simpa only [sub_mul, Finset.sum_sub_distrib] using
    hasCutWidth_retNet γ ctx qry out (fun c => R c v₁ - R c v₀)

/-- One key with values `±1` and a state of dimension one: the pair `(1, φ(π 0))` of the context
is read back by the query. -/
example : Fintype.card (Fin 1) ≤ 1 :=
  card_le_of_retNet_recall (ν := Fin 2) 1 (fun π => [(fun _ => 1, fun _ => ![1, -1] (π 0))])
    (fun _ => (0, 0)) (fun _ _ => 1) (fun _ => 1) (φ := ![1, -1]) (v := 0) (v' := 1)
    (by norm_num) fun q π => by
      fin_cases q
      simp [retNetRun, retNetStep]

/-- Its logits `φ(v) Out` have the largest one at the bound value. -/
example : Fintype.card (Fin 1) ≤ 1 :=
  card_le_of_retNet_argmax (ν := Fin 2) 1 (fun π => [(fun _ => 1, fun _ => ![1, -1] (π 0))])
    (fun _ => (0, 0)) (fun _ _ => 1) (fun _ v => ![1, -1] v) (v₀ := 0) (v₁ := 1) (by decide)
    fun π q v hv => by
      obtain rfl : q = 0 := Subsingleton.elim _ _
      simp only [retNetRun, retNetStep, List.foldl_cons, List.foldl_nil, List.cons_append,
        List.nil_append, Fin.sum_univ_one]
      generalize π 0 = u at hv ⊢
      fin_cases v <;> fin_cases u <;> simp_all

example := retNet_recall_error (κ := Fin 2) (ν := Fin 2) (d := 1) 1
  (fun π => [(fun _ => 1, fun _ => ![1, -1] (π 0))]) (fun _ => (0, 0)) (fun _ _ => 1)
  (fun _ => 1) (φ := ![1, -1]) (μ := 0) (by simp [Fin.sum_univ_two])

end Transformer.Grokking.FlatSpectrum.Recall

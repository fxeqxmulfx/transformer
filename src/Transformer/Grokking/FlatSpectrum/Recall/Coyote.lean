module

public import Transformer.Grokking.FlatSpectrum.Recall.Sign
public import Transformer.Grokking.FlatSpectrum.Recall.Width
public import Transformer.Zoology.Section4_Coyote

/-!
# One gated convolution recalls one binding for each dimension

Source: `review/flat_spectrum.md`, *Associative recall has a flat spectrum* (added), item AR6,
for the Coyote layer (printed as BaseConv) of Arora et al., arXiv:2312.04927v1, §4, equation
`eq: coyote-recursion`, `y = (u W + b₁) ⊙ (h ∗ u + b₂)`, and its Claim 1: "Gated-convolution
models with two layers require model dimension to scale at least linearly in sequence length in
order to solve associative recall". The claim is empirical. Here one layer reads the context.

The prompt is `n` context tokens and then the query token (`prompt`). At the query the first
factor of the layer is a function of the query token alone. The convolution is additive in its
input (`causalConvolution_add`), so the second factor is a function of the query token plus one of
the context tokens, one number for each of the `d` channels. So a linear readout
`∑_c w_c y_c` of the layer at the query has cut width `d` (`hasCutWidth_coyoteLayer`), however
the tokens encode the context and the query. With item AR4 and item AR5, for `m` keys:

* the squared error on recall is at least `(m − d) c^{m−1} ∑_v (φ(v) − μ)²`
  (`coyoteLayer_recall_error`);
* exact recall of a non-constant `φ` has `d ≥ m` (`card_le_of_coyoteLayer_recall`);
* a correct argmax over logits `∑_c R_{c v} y_c` has `d ≥ m` (`card_le_of_coyoteLayer_argmax`).

The bound is attained: with one-hot keys, a layer of dimension `m` recalls `m` bindings
(`exists_coyoteLayer_recall`). The circular convolution that Appendix
`sec:simplified-hyena-append` also allows gives the same layer at the query
(`coyoteLayerCyclic_last`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Transformer.Zoology

variable {κ ν : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν]

/-- The prompt: `n` context tokens and then the query token. -/
abbrev prompt {n d : ℕ} (x : Fin n → Fin d → ℝ) (e : Fin d → ℝ) : RealSequence (n + 1) d :=
  Fin.snoc (α := fun _ => Fin d → ℝ) x e

/-- A prompt is its context part plus its query part. -/
lemma prompt_eq_add {n d : ℕ} (x : Fin n → Fin d → ℝ) (e : Fin d → ℝ) :
    prompt x e = prompt x 0 + prompt 0 e := by
  funext i
  refine Fin.lastCases ?_ (fun j => ?_) i <;> simp [prompt]

/-- The causal convolution is additive in its input, §4, equation `eq: coyote-recursion`. -/
lemma causalConvolution_add {n d : ℕ} (u u' h : RealSequence n d) :
    causalConvolution (u + u') h = causalConvolution u h + causalConvolution u' h := by
  funext i c
  simp only [causalConvolution, Pi.add_apply, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun k _ => ?_
  split_ifs <;> ring

/-- At the last position the circular convolution sums the terms of the causal one, so the
layer of Appendix `sec:simplified-hyena-append` with a circular convolution is `coyoteLayer` at
the query, and the results below hold for it. -/
lemma coyoteLayerCyclic_last {n d : ℕ} (p : CoyoteParameters (n + 1) d)
    (u : RealSequence (n + 1) d) :
    coyoteLayerCyclic p u (Fin.last n) = coyoteLayer p u (Fin.last n) := by
  funext c
  simp [coyoteLayerCyclic, coyoteLayer, cyclicConvolution, causalConvolution, Fin.le_last]

/-- **A Coyote layer read at the query has cut width its dimension**
(`review/flat_spectrum.md`, AR6): for every encoding `ctx` of the context in `n` tokens and
`qry` of the query in one token, and every linear readout `w`. -/
theorem hasCutWidth_coyoteLayer {Q C : Type*} {n d : ℕ} (p : CoyoteParameters (n + 1) d)
    (w : Fin d → ℝ) (ctx : C → Fin n → Fin d → ℝ) (qry : Q → Fin d → ℝ) :
    HasCutWidth d fun q x =>
      ∑ c, w c * coyoteLayer p (prompt (ctx x) (qry q)) (Fin.last n) c := by
  refine ⟨fun q => ∑ c, w c * ((linearProjection (prompt 0 (qry q)) p.weight (Fin.last n) c +
      p.bias₁ (Fin.last n) c) * (causalConvolution (prompt 0 (qry q)) p.filter (Fin.last n) c +
        p.bias₂ (Fin.last n) c)),
    fun q c => w c * (linearProjection (prompt 0 (qry q)) p.weight (Fin.last n) c +
      p.bias₁ (Fin.last n) c),
    fun x c => causalConvolution (prompt (ctx x) 0) p.filter (Fin.last n) c, fun q x => ?_⟩
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun c _ => ?_
  have hlin : linearProjection (prompt (ctx x) (qry q)) p.weight (Fin.last n) c =
      linearProjection (prompt 0 (qry q)) p.weight (Fin.last n) c := by
    simp [linearProjection, prompt]
  have hconv : causalConvolution (prompt (ctx x) (qry q)) p.filter (Fin.last n) c =
      causalConvolution (prompt (ctx x) 0) p.filter (Fin.last n) c +
        causalConvolution (prompt 0 (qry q)) p.filter (Fin.last n) c := by
    rw [prompt_eq_add, causalConvolution_add]
    rfl
  simp only [coyoteLayer, hlin, hconv]
  ring

/-- **The error of a Coyote layer on recall** (`review/flat_spectrum.md`, AR6): a layer of
dimension `d` read linearly at the query misses the recall of `m` keys by at least
`(m − d) c^{m−1} ∑_v (φ(v) − μ)²`, with `μ` the mean of `φ`. -/
theorem coyoteLayer_recall_error {n d : ℕ} (p : CoyoteParameters (n + 1) d) (w : Fin d → ℝ)
    (ctx : (κ → ν) → Fin n → Fin d → ℝ) (qry : κ → Fin d → ℝ) {φ : ν → ℝ} {μ : ℝ}
    (hμ : ∑ v, (φ v - μ) = 0) :
    ((Fintype.card κ : ℝ) - d) * ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) *
      ∑ v, (φ v - μ) ^ 2) ≤ ∑ q, ∑ π : κ → ν,
        (φ (π q) - ∑ c, w c * coyoteLayer p (prompt (ctx π) (qry q)) (Fin.last n) c) ^ 2 :=
  (hasCutWidth_coyoteLayer p w ctx qry).mul_le_sum_sq hμ

/-- **A Coyote layer of dimension `d` recalls at most `d` bindings**
(`review/flat_spectrum.md`, AR6): exact recall of a non-constant `φ` for `m` keys has
`d ≥ m`. -/
theorem card_le_of_coyoteLayer_recall {n d : ℕ} (p : CoyoteParameters (n + 1) d)
    (w : Fin d → ℝ) (ctx : (κ → ν) → Fin n → Fin d → ℝ) (qry : κ → Fin d → ℝ) {φ : ν → ℝ}
    {v v' : ν} (hφ : φ v ≠ φ v')
    (h : ∀ q π, ∑ c, w c * coyoteLayer p (prompt (ctx π) (qry q)) (Fin.last n) c = φ (π q)) :
    Fintype.card κ ≤ d :=
  (hasCutWidth_coyoteLayer p w ctx qry).card_le_of_recall hφ h

omit [DecidableEq κ] [Fintype ν] in
/-- **A Coyote layer of dimension `d` classifies at most `d` bindings**
(`review/flat_spectrum.md`, AR6): if the largest of the logits `∑_c R_{c v} y_c` at the query
is always that of the bound value, then `d ≥ m`. -/
theorem card_le_of_coyoteLayer_argmax {n d : ℕ} (p : CoyoteParameters (n + 1) d)
    (R : Fin d → ν → ℝ) (ctx : (κ → ν) → Fin n → Fin d → ℝ) (qry : κ → Fin d → ℝ) {v₀ v₁ : ν}
    (hv : v₀ ≠ v₁)
    (h : ∀ π q v, v ≠ π q →
      ∑ c, R c v * coyoteLayer p (prompt (ctx π) (qry q)) (Fin.last n) c <
        ∑ c, R c (π q) * coyoteLayer p (prompt (ctx π) (qry q)) (Fin.last n) c) :
    Fintype.card κ ≤ d := by
  refine card_le_of_argmax (L := fun v q π =>
    ∑ c, R c v * coyoteLayer p (prompt (ctx π) (qry q)) (Fin.last n) c) hv ?_ h
  simpa only [sub_mul, Finset.sum_sub_distrib] using
    hasCutWidth_coyoteLayer p (fun c => R c v₁ - R c v₀) ctx qry

omit [Fintype ν] in
/-- **A Coyote layer of dimension `m` recalls `m` bindings** (`review/flat_spectrum.md`, AR6):
one-hot keys, the value of the key `j` in every channel of the token `j`, so that the context
lists the values in the order of the keys, and a filter that shifts the token `c` to the query
in channel `c`. -/
theorem exists_coyoteLayer_recall (m : ℕ) (φ : ν → ℝ) :
    ∃ (p : CoyoteParameters (m + 1) m) (w : Fin m → ℝ)
      (ctx : (Fin m → ν) → Fin m → Fin m → ℝ) (qry : Fin m → Fin m → ℝ), ∀ q π,
        ∑ c, w c * coyoteLayer p (prompt (ctx π) (qry q)) (Fin.last m) c = φ (π q) := by
  refine ⟨⟨fun k c => if k = c then 1 else 0,
    fun k c => if k = Fin.last m - c.castSucc then 1 else 0, fun _ _ => 0, fun _ _ => 0⟩,
    fun _ => 1, fun π j _ => φ (π j), fun q k => if k = q then 1 else 0, fun q π => ?_⟩
  have hlin (c : Fin m) : linearProjection
      (prompt (fun j _ => φ (π j)) fun k => if k = q then 1 else 0)
      (fun k c => if k = c then (1 : ℝ) else 0) (Fin.last m) c = if c = q then 1 else 0 := by
    simp [linearProjection, prompt]
  have hconv (c : Fin m) : causalConvolution
      (prompt (fun j _ => φ (π j)) fun k => if k = q then 1 else 0)
      (fun k c => if k = Fin.last m - c.castSucc then (1 : ℝ) else 0) (Fin.last m) c =
        φ (π c) := by
    simp [causalConvolution, Fin.le_last, prompt]
  simp [coyoteLayer, hlin, hconv]

/-- Two keys with values `±1`: a layer of dimension two recalls them, and none of dimension
one does. -/
example : ∃ (p : CoyoteParameters 3 2) (w : Fin 2 → ℝ) (ctx : (Fin 2 → Fin 2) → Fin 2 → Fin 2 → ℝ)
    (qry : Fin 2 → Fin 2 → ℝ), ∀ q π, ∑ c, w c * coyoteLayer p (prompt (ctx π) (qry q))
      (Fin.last 2) c = (![1, -1] : Fin 2 → ℝ) (π q) :=
  exists_coyoteLayer_recall 2 _

/-- The layer above meets the hypothesis of exact recall. -/
example : Fintype.card (Fin 2) ≤ 2 := by
  obtain ⟨p, w, ctx, qry, h⟩ := exists_coyoteLayer_recall 2 (![1, -1] : Fin 2 → ℝ)
  exact card_le_of_coyoteLayer_recall p w ctx qry (v := 0) (v' := 1) (by norm_num) h

/-- Its logits `φ(v) ∑_c w_c y_c = φ(v) φ(π q)` have the largest one at the bound value. -/
example : Fintype.card (Fin 2) ≤ 2 := by
  obtain ⟨p, w, ctx, qry, h⟩ := exists_coyoteLayer_recall 2 (![1, -1] : Fin 2 → ℝ)
  refine card_le_of_coyoteLayer_argmax p (fun c v => (![1, -1] : Fin 2 → ℝ) v * w c) ctx qry
    (v₀ := 0) (v₁ := 1) (by decide) fun π q v hv => ?_
  simp only [mul_assoc, ← Finset.mul_sum, h]
  generalize π q = u at hv ⊢
  fin_cases v <;> fin_cases u <;> simp_all

example (p : CoyoteParameters 3 2) (w : Fin 2 → ℝ) (ctx : (Fin 3 → Fin 2) → Fin 2 → Fin 2 → ℝ)
    (qry : Fin 3 → Fin 2 → ℝ) :=
  coyoteLayer_recall_error p w ctx qry (φ := ![1, -1]) (μ := 0) (by simp [Fin.sum_univ_two])

end Transformer.Grokking.FlatSpectrum.Recall

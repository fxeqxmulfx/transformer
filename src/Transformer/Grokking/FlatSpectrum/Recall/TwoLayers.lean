module

public import Transformer.Grokking.FlatSpectrum.Recall.Coyote
public import Transformer.Grokking.FlatSpectrum.Recall.Quadratic

/-!
# Two gated convolutions pass `2d` numbers to the query, read by a quadratic

Source: `review/flat_spectrum.md`, *Polynomial readouts and two layers* (added), item PR5, for
the Coyote layer of Arora et al., arXiv:2312.04927v1, §4, equation `eq: coyote-recursion`,
`y = (u W + b₁) ⊙ (h ∗ u + b₂)`, and for its Claim 1: "Gated-convolution models with two layers
require model dimension to scale at least linearly in sequence length in order to solve
associative recall".

Stack two layers and read the second at the query. The prompt is `n` context tokens and the
query token. A layer is causal: before the query it does not see the query token
(`coyoteLayer_prompt_castSucc`), so its output is again a prompt, a context part and a query
part (`coyoteLayer_prompt`). At the query a layer is

    y_c = (∑_k W_{kc} e_k + b₁) (B_c + h[0]_c e_c + b₂),      B_c = ∑_{k ≥ 1} h[k]_c u_{N−k,c},

with `e` the query token (`coyoteLayer_prompt_last`): affine in the `d` numbers `B` of the
context. The second layer convolves the outputs of the first before the query, `d` more
numbers `G` of the context (`firstCross`, `secondCross`), and multiplies two affine functions
of `(B, G)`. So a linear readout of two layers at the query is quadratic in `2d` numbers of the
context, with coefficients that depend on the query (`isQuadIn_twoCoyoteLayers`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Transformer.Zoology

variable {n d : ℕ}

/-- Before the query a prompt is its context. -/
lemma prompt_of_ne_last (x : Fin n → Fin d → ℝ) (e e' : Fin d → ℝ) {j : Fin (n + 1)}
    (hj : j ≠ Fin.last n) : prompt x e j = prompt x e' j := by
  obtain ⟨i, rfl⟩ := Fin.exists_castSucc_eq.2 hj
  simp [prompt]

example (x : Fin 2 → Fin 1 → ℝ) (e : Fin 1 → ℝ) : prompt x e 0 = prompt x 0 0 :=
  prompt_of_ne_last x e 0 (by decide)

/-- The query token alone convolves to its own term `h[0] e`. -/
lemma causalConvolution_prompt_zero_last (e : Fin d → ℝ) (h : RealSequence (n + 1) d)
    (c : Fin d) : causalConvolution (prompt 0 e) h (Fin.last n) c = h 0 c * e c := by
  rw [causalConvolution, Finset.sum_eq_single (0 : Fin (n + 1))]
  · simp [prompt]
  · intro k _ hk
    have hne : Fin.last n - k ≠ Fin.last n := fun h0 => hk (sub_eq_self.1 h0)
    rw [prompt_of_ne_last 0 e 0 hne]
    obtain ⟨i, hi⟩ := Fin.exists_castSucc_eq.2 hne
    simp [prompt, ← hi]
  · exact fun h0 => absurd (Finset.mem_univ _) h0

/-- **A Coyote layer at the query** (`review/flat_spectrum.md`, PR5): affine in the numbers
`B_c` that its convolution takes from the context. -/
lemma coyoteLayer_prompt_last (p : CoyoteParameters (n + 1) d) (x : Fin n → Fin d → ℝ)
    (e : Fin d → ℝ) (c : Fin d) :
    coyoteLayer p (prompt x e) (Fin.last n) c =
      (∑ k, p.weight k c * e k + p.bias₁ (Fin.last n) c) *
        (causalConvolution (prompt x 0) p.filter (Fin.last n) c + p.filter 0 c * e c +
          p.bias₂ (Fin.last n) c) := by
  have hconv : causalConvolution (prompt x e) p.filter (Fin.last n) c =
      causalConvolution (prompt x 0) p.filter (Fin.last n) c + p.filter 0 c * e c := by
    rw [prompt_eq_add, causalConvolution_add, ← causalConvolution_prompt_zero_last e p.filter c]
    rfl
  have hlin : linearProjection (prompt x e) p.weight (Fin.last n) c =
      ∑ k, p.weight k c * e k := by
    simp only [linearProjection, prompt, Fin.snoc_last]
    exact Finset.sum_congr rfl fun k _ => mul_comm _ _
  rw [coyoteLayer, hconv, hlin]

/-- **A Coyote layer is causal** (`review/flat_spectrum.md`, PR5): before the query it does not
see the query token. -/
lemma coyoteLayer_prompt_castSucc (p : CoyoteParameters (n + 1) d) (x : Fin n → Fin d → ℝ)
    (e : Fin d → ℝ) (i : Fin n) :
    coyoteLayer p (prompt x e) (Fin.castSucc i) = coyoteLayer p (prompt x 0) (Fin.castSucc i) := by
  funext c
  have hconv : causalConvolution (prompt x e) p.filter (Fin.castSucc i) c =
      causalConvolution (prompt x 0) p.filter (Fin.castSucc i) c := by
    refine Finset.sum_congr rfl fun k _ => ?_
    split_ifs with hk
    · have hne : Fin.castSucc i - k ≠ Fin.last n := by
        intro h0
        have h1 := Fin.coe_sub_iff_le.2 hk
        rw [h0, Fin.val_last, Fin.val_castSucc] at h1
        have := i.isLt
        omega
      rw [prompt_of_ne_last x e 0 hne]
    · rfl
  have hlin : linearProjection (prompt x e) p.weight (Fin.castSucc i) c =
      linearProjection (prompt x 0) p.weight (Fin.castSucc i) c := by
    simp only [linearProjection, prompt, Fin.snoc_castSucc]
  rw [coyoteLayer, hconv, hlin]
  rfl

/-- **The output of a layer on a prompt is a prompt**: the outputs before the query are those
of the context alone. -/
lemma coyoteLayer_prompt (p : CoyoteParameters (n + 1) d) (x : Fin n → Fin d → ℝ)
    (e : Fin d → ℝ) :
    coyoteLayer p (prompt x e) = prompt (fun i => coyoteLayer p (prompt x 0) (Fin.castSucc i))
      (coyoteLayer p (prompt x e) (Fin.last n)) := by
  funext j
  induction j using Fin.lastCases with
  | last => simp only [prompt, Fin.snoc_last]
  | cast i => simp only [prompt, Fin.snoc_castSucc, coyoteLayer_prompt_castSucc p x e i]

/-- The `d` numbers of the context that the first layer passes to the query:
`B_c = ∑_{k ≥ 1} h[k]_c u_{N−k,c}`. -/
noncomputable def firstCross (p₁ : CoyoteParameters (n + 1) d) (x : Fin n → Fin d → ℝ) :
    Fin d → ℝ :=
  fun c => causalConvolution (prompt x 0) p₁.filter (Fin.last n) c

/-- The `d` numbers of the context that the second layer passes to the query: its convolution
of the outputs of the first layer before the query. -/
noncomputable def secondCross (p₁ p₂ : CoyoteParameters (n + 1) d) (x : Fin n → Fin d → ℝ) :
    Fin d → ℝ :=
  fun c => causalConvolution
    (prompt (fun i => coyoteLayer p₁ (prompt x 0) (Fin.castSucc i)) 0) p₂.filter (Fin.last n) c

/-- The `2d` numbers of the context that two layers pass to the query. -/
noncomputable def twoLayerCross (p₁ p₂ : CoyoteParameters (n + 1) d) (x : Fin n → Fin d → ℝ) :
    Fin (d + d) → ℝ :=
  Fin.append (firstCross p₁ x) (secondCross p₁ p₂ x)

/-- **Two Coyote layers at the query** (`review/flat_spectrum.md`, PR5): the second layer
multiplies two affine functions of the output `e'` of the first at the query and of its own
`d` numbers of the context. -/
lemma coyoteLayer_coyoteLayer_prompt_last (p₁ p₂ : CoyoteParameters (n + 1) d)
    (x : Fin n → Fin d → ℝ) (e : Fin d → ℝ) (c : Fin d) :
    coyoteLayer p₂ (coyoteLayer p₁ (prompt x e)) (Fin.last n) c =
      (∑ k, p₂.weight k c * coyoteLayer p₁ (prompt x e) (Fin.last n) k +
        p₂.bias₁ (Fin.last n) c) *
        (secondCross p₁ p₂ x c + p₂.filter 0 c * coyoteLayer p₁ (prompt x e) (Fin.last n) c +
          p₂.bias₂ (Fin.last n) c) :=
  (congrArg (fun u => coyoteLayer p₂ u (Fin.last n) c) (coyoteLayer_prompt p₁ x e)).trans
    (coyoteLayer_prompt_last p₂ _ _ c)

/-- **Two Coyote layers read at the query are quadratic in `2d` numbers of the context**
(`review/flat_spectrum.md`, PR5), for every encoding of the context and of the query and every
linear readout `w`. -/
theorem isQuadIn_twoCoyoteLayers {Q C : Type*} (p₁ p₂ : CoyoteParameters (n + 1) d)
    (w : Fin d → ℝ) (ctx : C → Fin n → Fin d → ℝ) (qry : Q → Fin d → ℝ) :
    IsQuadIn (fun x => twoLayerCross p₁ p₂ (ctx x)) fun q x =>
      ∑ c, w c * coyoteLayer p₂ (coyoteLayer p₁ (prompt (ctx x) (qry q))) (Fin.last n) c := by
  set z : C → Fin (d + d) → ℝ := fun x => twoLayerCross p₁ p₂ (ctx x) with hz
  have hE (k : Fin d) : IsAffineIn z fun q x =>
      coyoteLayer p₁ (prompt (ctx x) (qry q)) (Fin.last n) k := by
    obtain ⟨g, a, h⟩ := ((isAffineIn_coord (Q := Q) z (Fin.castAdd d k)).add
      (isAffineIn_query z fun q => p₁.filter 0 k * qry q k + p₁.bias₂ (Fin.last n) k)).smul
        fun q => ∑ j, p₁.weight j k * qry q j + p₁.bias₁ (Fin.last n) k
    refine ⟨g, a, fun q x => Eq.trans ?_ (h q x)⟩
    beta_reduce
    rw [coyoteLayer_prompt_last]
    simp only [hz, twoLayerCross, Fin.append_left, firstCross]
    ring
  have hU (c : Fin d) : IsAffineIn z fun q x => w c *
      (∑ k, p₂.weight k c * coyoteLayer p₁ (prompt (ctx x) (qry q)) (Fin.last n) k +
        p₂.bias₁ (Fin.last n) c) :=
    ((IsAffineIn.sum Finset.univ fun k => (hE k).smul fun _ => p₂.weight k c).add
      (isAffineIn_query z fun _ => p₂.bias₁ (Fin.last n) c)).smul fun _ => w c
  have hV (c : Fin d) : IsAffineIn z fun q x => z x (Fin.natAdd d c) +
      p₂.filter 0 c * coyoteLayer p₁ (prompt (ctx x) (qry q)) (Fin.last n) c +
        p₂.bias₂ (Fin.last n) c :=
    ((isAffineIn_coord z (Fin.natAdd d c)).add ((hE c).smul fun _ => p₂.filter 0 c)).add
      (isAffineIn_query z fun _ => p₂.bias₂ (Fin.last n) c)
  obtain ⟨g, a, A, h⟩ := IsQuadIn.sum Finset.univ fun c => (hU c).mul (hV c)
  refine ⟨g, a, A, fun q x => Eq.trans ?_ (h q x)⟩
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [coyoteLayer_coyoteLayer_prompt_last]
  simp only [hz, twoLayerCross, Fin.append_right]
  ring

end Transformer.Grokking.FlatSpectrum.Recall

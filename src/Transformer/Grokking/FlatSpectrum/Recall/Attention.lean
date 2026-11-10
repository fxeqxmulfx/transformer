module

public import Transformer.Grokking.FlatSpectrum.Recall.Sign
public import Transformer.Grokking.FlatSpectrum.Recall.Width

/-!
# Attention without softmax recalls one binding for each dimension

Source: `review/flat_spectrum.md`, *Associative recall has a flat spectrum* (added), item AR8,
for Arora et al., arXiv:2312.04927v1, §4, Proposition `prop: attention-ar`: "Attention (even
without using soft-max) solves MQAR … using O(c²) parameters", proved in Appendix Proposition
`prop: app-attention` with one-hot tokens of dimension `d = 3c` for a vocabulary of size `c`.

Attention without softmax returns `∑_j ⟨Q q, K x_j⟩ V x_j` at the query: the score of the
context token `j` is bilinear in the projection `Q q ∈ ℝ^d` of the query and the projection
`K x_j ∈ ℝ^d` of the token. So every linear readout of its output has cut width `d`, the
dimension of the projections (`hasCutWidth_linearAttention`): the `d` numbers
`∑_j (K x_j)_r v(x_j)` cross from the context to the query. With item AR4 and item AR5, for `m`
keys:

* the squared error on recall is at least `(m − d) c^{m−1} ∑_v (φ(v) − μ)²`
  (`linearAttention_recall_error`);
* exact recall of a non-constant `φ` has `d ≥ m` (`card_le_of_linearAttention_recall`);
* a correct argmax has `d ≥ m` (`card_le_of_linearAttention_argmax`).

The appendix adds linear biases to the scores, `(Q Kᵀ + B) V` (Algorithm
`ALiBi-without-softmax`). The row of `B` at the query adds `∑_j B_j v(x_j)`, a number of the
context alone, so the cut width is `d + 1` (`hasCutWidth_biasedAttention`) and the three bounds
hold with `d + 1` (`biasedAttention_recall_error`, `card_le_of_biasedAttention_recall`,
`card_le_of_biasedAttention_argmax`); two keys are recalled at dimension one.

So the dimension of the paper's construction, linear in the number of distinct keys, is needed
without softmax. Softmax divides the scores by a function of the query and the context
together; that output is not of this form, and nothing here bounds it.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

variable {κ ν : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν]

/-- **Attention without softmax has cut width the dimension of its projections**
(`review/flat_spectrum.md`, AR8): `qf q` is the projection `Q q` of the query, `kf x j` the
projection `K x_j` of the context token `j`, `vf x j` a linear readout of its value `V x_j`, and
`self q` the term of the query attending to itself. -/
theorem hasCutWidth_linearAttention {Q C : Type*} {n d : ℕ} (self : Q → ℝ)
    (qf : Q → Fin d → ℝ) (kf : C → Fin n → Fin d → ℝ) (vf : C → Fin n → ℝ) :
    HasCutWidth d fun q x => self q + ∑ j, (∑ r, qf q r * kf x j r) * vf x j := by
  refine ⟨self, qf, fun x r => ∑ j, kf x j r * vf x j, fun q x => ?_⟩
  simp only [Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact congrArg _ (Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun j _ => by ring)

/-- **The error of attention without softmax on recall** (`review/flat_spectrum.md`, AR8): at
least `(m − d) c^{m−1} ∑_v (φ(v) − μ)²`, with `μ` the mean of `φ`. -/
theorem linearAttention_recall_error {n d : ℕ} (self : κ → ℝ) (qf : κ → Fin d → ℝ)
    (kf : (κ → ν) → Fin n → Fin d → ℝ) (vf : (κ → ν) → Fin n → ℝ) {φ : ν → ℝ} {μ : ℝ}
    (hμ : ∑ v, (φ v - μ) = 0) :
    ((Fintype.card κ : ℝ) - d) * ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) *
      ∑ v, (φ v - μ) ^ 2) ≤ ∑ q, ∑ π : κ → ν,
        (φ (π q) - (self q + ∑ j, (∑ r, qf q r * kf π j r) * vf π j)) ^ 2 :=
  (hasCutWidth_linearAttention self qf kf vf).mul_le_sum_sq hμ

/-- **Attention without softmax of dimension `d` recalls at most `d` bindings**
(`review/flat_spectrum.md`, AR8): exact recall of a non-constant `φ` for `m` keys has
`d ≥ m`. -/
theorem card_le_of_linearAttention_recall {n d : ℕ} (self : κ → ℝ) (qf : κ → Fin d → ℝ)
    (kf : (κ → ν) → Fin n → Fin d → ℝ) (vf : (κ → ν) → Fin n → ℝ) {φ : ν → ℝ} {v v' : ν}
    (hφ : φ v ≠ φ v')
    (h : ∀ q π, self q + ∑ j, (∑ r, qf q r * kf π j r) * vf π j = φ (π q)) :
    Fintype.card κ ≤ d :=
  (hasCutWidth_linearAttention self qf kf vf).card_le_of_recall hφ h

omit [DecidableEq κ] [Fintype ν] in
/-- **Attention without softmax of dimension `d` classifies at most `d` bindings**
(`review/flat_spectrum.md`, AR8): if the largest of the logits, read from the values by
`vf u`, is always that of the bound value, then `d ≥ m`. -/
theorem card_le_of_linearAttention_argmax {n d : ℕ} (self : ν → κ → ℝ) (qf : κ → Fin d → ℝ)
    (kf : (κ → ν) → Fin n → Fin d → ℝ) (vf : ν → (κ → ν) → Fin n → ℝ) {v₀ v₁ : ν}
    (hv : v₀ ≠ v₁)
    (h : ∀ π q u, u ≠ π q →
      self u q + ∑ j, (∑ r, qf q r * kf π j r) * vf u π j <
        self (π q) q + ∑ j, (∑ r, qf q r * kf π j r) * vf (π q) π j) :
    Fintype.card κ ≤ d := by
  refine card_le_of_argmax (L := fun u q π =>
    self u q + ∑ j, (∑ r, qf q r * kf π j r) * vf u π j) hv ?_ h
  have h1 := hasCutWidth_linearAttention (fun q => self v₁ q - self v₀ q) qf kf
    fun π j => vf v₁ π j - vf v₀ π j
  simp only [mul_sub, Finset.sum_sub_distrib] at h1
  convert h1 using 3
  ring

/-- **Attention with linear biases has cut width one more than its dimension**
(`review/flat_spectrum.md`, AR8): in the layer `(Q Kᵀ + B) V` of the paper's Appendix Algorithm
`ALiBi-without-softmax`, `bias j` is the entry of `B` at the query and the context token `j`.
The bias adds `∑_j B_j v(x_j)`, one more number of the context alone: it is the projection of
dimension `d + 1` with `1` at the query and `B_j` at the token `j`. -/
theorem hasCutWidth_biasedAttention {Q C : Type*} {n d : ℕ} (self : Q → ℝ)
    (qf : Q → Fin d → ℝ) (kf : C → Fin n → Fin d → ℝ) (bias : Fin n → ℝ)
    (vf : C → Fin n → ℝ) :
    HasCutWidth (d + 1) fun q x =>
      self q + ∑ j, (∑ r, qf q r * kf x j r + bias j) * vf x j := by
  simpa only [Fin.sum_univ_castSucc, Fin.snoc_castSucc, Fin.snoc_last, one_mul] using
    hasCutWidth_linearAttention self (fun q => Fin.snoc (α := fun _ => ℝ) (qf q) 1)
      (fun x j => Fin.snoc (α := fun _ => ℝ) (kf x j) (bias j)) vf

/-- **The error of attention with linear biases on recall** (`review/flat_spectrum.md`, AR8): at
least `(m − d − 1) c^{m−1} ∑_v (φ(v) − μ)²`, with `μ` the mean of `φ`. -/
theorem biasedAttention_recall_error {n d : ℕ} (self : κ → ℝ) (qf : κ → Fin d → ℝ)
    (kf : (κ → ν) → Fin n → Fin d → ℝ) (bias : Fin n → ℝ) (vf : (κ → ν) → Fin n → ℝ)
    {φ : ν → ℝ} {μ : ℝ} (hμ : ∑ v, (φ v - μ) = 0) :
    ((Fintype.card κ : ℝ) - (d + 1 : ℕ)) * ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) *
      ∑ v, (φ v - μ) ^ 2) ≤ ∑ q, ∑ π : κ → ν,
        (φ (π q) - (self q + ∑ j, (∑ r, qf q r * kf π j r + bias j) * vf π j)) ^ 2 :=
  (hasCutWidth_biasedAttention self qf kf bias vf).mul_le_sum_sq hμ

/-- **Attention with linear biases of dimension `d` recalls at most `d + 1` bindings**
(`review/flat_spectrum.md`, AR8): exact recall of a non-constant `φ` for `m` keys has
`d + 1 ≥ m`. -/
theorem card_le_of_biasedAttention_recall {n d : ℕ} (self : κ → ℝ) (qf : κ → Fin d → ℝ)
    (kf : (κ → ν) → Fin n → Fin d → ℝ) (bias : Fin n → ℝ) (vf : (κ → ν) → Fin n → ℝ)
    {φ : ν → ℝ} {v v' : ν} (hφ : φ v ≠ φ v')
    (h : ∀ q π, self q + ∑ j, (∑ r, qf q r * kf π j r + bias j) * vf π j = φ (π q)) :
    Fintype.card κ ≤ d + 1 :=
  (hasCutWidth_biasedAttention self qf kf bias vf).card_le_of_recall hφ h

omit [DecidableEq κ] [Fintype ν] in
/-- **Attention with linear biases of dimension `d` classifies at most `d + 1` bindings**
(`review/flat_spectrum.md`, AR8): if the largest of the logits is always that of the bound
value, then `d + 1 ≥ m`. -/
theorem card_le_of_biasedAttention_argmax {n d : ℕ} (self : ν → κ → ℝ) (qf : κ → Fin d → ℝ)
    (kf : (κ → ν) → Fin n → Fin d → ℝ) (bias : Fin n → ℝ) (vf : ν → (κ → ν) → Fin n → ℝ)
    {v₀ v₁ : ν} (hv : v₀ ≠ v₁)
    (h : ∀ π q u, u ≠ π q →
      self u q + ∑ j, (∑ r, qf q r * kf π j r + bias j) * vf u π j <
        self (π q) q + ∑ j, (∑ r, qf q r * kf π j r + bias j) * vf (π q) π j) :
    Fintype.card κ ≤ d + 1 := by
  refine card_le_of_linearAttention_argmax self (fun q => Fin.snoc (α := fun _ => ℝ) (qf q) 1)
    (fun π j => Fin.snoc (α := fun _ => ℝ) (kf π j) (bias j)) vf hv ?_
  simpa only [Fin.sum_univ_castSucc, Fin.snoc_castSucc, Fin.snoc_last, one_mul] using h

/-- Two keys with values `±1` and one-hot projections of dimension two: the query `q` scores
the token `j` by `[q = j]` and reads its value. -/
example : Fintype.card (Fin 2) ≤ 2 :=
  card_le_of_linearAttention_recall (ν := Fin 2) (n := 2) (fun _ => 0)
    (fun q r => if r = q then 1 else 0) (fun _ j r => if r = j then 1 else 0)
    (fun π j => (![1, -1] : Fin 2 → ℝ) (π j)) (φ := ![1, -1]) (v := 0) (v' := 1)
    (by norm_num) fun q π => by
      simp only [zero_add, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ,
        ↓reduceIte, mul_ite, mul_one, mul_zero]

/-- The logits `φ(u) φ(π q)` of the same projections have the largest one at the bound value. -/
example : Fintype.card (Fin 2) ≤ 2 :=
  card_le_of_linearAttention_argmax (ν := Fin 2) (n := 2) (fun _ _ => 0)
    (fun q r => if r = q then 1 else 0) (fun _ j r => if r = j then 1 else 0)
    (fun u π j => (![1, -1] : Fin 2 → ℝ) u * ![1, -1] (π j)) (v₀ := 0) (v₁ := 1) (by decide)
    fun π q u hu => by
      simp only [zero_add, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ,
        ↓reduceIte, mul_ite, mul_one, mul_zero]
      generalize π q = w at hu ⊢
      fin_cases u <;> fin_cases w <;> simp_all

example := linearAttention_recall_error (κ := Fin 3) (ν := Fin 2) (n := 2) (d := 2)
  (fun _ => 0) (fun q r => if (r : ℕ) = q then 1 else 0) (fun _ j r => if r = j then 1 else 0)
  (fun π j => (![1, -1] : Fin 2 → ℝ) (π ⟨j, by omega⟩)) (φ := ![1, -1]) (μ := 0)
  (by simp [Fin.sum_univ_two])

/-- The bias buys one binding: two keys with values `±1` are recalled at dimension one. The key
`1` is read by the bias alone, and the key `0` adds the score `±1` of the two tokens. -/
example : Fintype.card (Fin 2) ≤ 1 + 1 :=
  card_le_of_biasedAttention_recall (ν := Fin 2) (n := 2) (fun _ => 0)
    (fun q _ => if q = 0 then 1 else 0) (fun _ j _ => if j = 0 then 1 else -1)
    (fun j => if j = 1 then 1 else 0) (fun π j => (![1, -1] : Fin 2 → ℝ) (π j))
    (φ := ![1, -1]) (v := 0) (v' := 1) (by norm_num) fun q π => by
      fin_cases q <;> simp [Fin.sum_univ_two]

/-- The logits `φ(u) φ(π q)` of the same layer have the largest one at the bound value. -/
example : Fintype.card (Fin 2) ≤ 1 + 1 :=
  card_le_of_biasedAttention_argmax (ν := Fin 2) (n := 2) (fun _ _ => 0)
    (fun q _ => if q = 0 then 1 else 0) (fun _ j _ => if j = 0 then 1 else -1)
    (fun j => if j = 1 then 1 else 0)
    (fun u π j => (![1, -1] : Fin 2 → ℝ) u * ![1, -1] (π j)) (v₀ := 0) (v₁ := 1) (by decide)
    fun π q u hu => by
      have hw : ∀ w : Fin 2, u ≠ w →
          (![1, -1] : Fin 2 → ℝ) u * ![1, -1] w < ![1, -1] w * ![1, -1] w := fun w hw => by
        fin_cases u <;> fin_cases w <;> simp_all
      fin_cases q <;> simpa [Fin.sum_univ_two] using hw _ hu

example := biasedAttention_recall_error (κ := Fin 3) (ν := Fin 2) (n := 2) (d := 1)
  (fun _ => 0) (fun q _ => if q = 0 then 1 else 0) (fun _ j _ => if j = 0 then 1 else -1)
  (fun j => if j = 1 then 1 else 0) (fun π j => (![1, -1] : Fin 2 → ℝ) (π ⟨j, by omega⟩))
  (φ := ![1, -1]) (μ := 0) (by simp [Fin.sum_univ_two])

end Transformer.Grokking.FlatSpectrum.Recall

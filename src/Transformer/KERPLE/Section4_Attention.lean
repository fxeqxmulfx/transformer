module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Transformer.KERPLE.Section3_Embedding

/-!
# §4: Kernelized relative positional embedding

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, §4.

* Eq. 1 (`attention`): `a_{m,n} = exp((q_mᵀk_n + k̃(m, n))/√d) / Σ_i exp((q_mᵀk_i + k̃(m, i))/√d)`
  for queries and keys in `ℝ^d` and a kernel `k̃` on positions. The positions are `0, …, L − 1`
  instead of `1, …, L`, which does not change a shift-invariant `k̃`.
* (*) The Softmax normalization is shift-invariant (`softmax_add_const`).
* Eq. 2 (`attention_eq_composite`): the attention does not change when `k̃` is replaced by the bias
  kernel `c + k̃` of Eq. 3 (`biasKernel`), and it is the Softmax of the composite kernel of Eq. 4
  (`compositeKernel`), `k^comp([q, m], [k, n]) = qᵀk + k(m, n)`, the pair `(q, m)` standing for
  the concatenation `[q, m] ∈ ℝ^{d+1}`.
* The composite kernel is PD exactly when the bias kernel is (`isPDKernel_compositeKernel_iff`),
  inner products being PD (`isPDKernel_inner`). Whether the bias kernel is PD, as §4 claims, is
  `Section4_BiasKernel`.
* Practical Choice: the power variant `k̃(m, n) = −r₁|m − n|^{r₂}`, `r₁ > 0`, `0 < r₂ ≤ 2`, and the
  logarithmic variant `k̃(m, n) = −r₁ log(1 + r₂|m − n|)`, `r₁, r₂ > 0`, are CPD by Corollary 1
  (`isCPDKernel_powerKernel`, `isCPDKernel_logKernel`).
* Connection to Prior Work: at `r₂ = 1` the power variant is ALiBi with slope `r₁/√d`
  (`attention_powerKernel_one`); the logarithmic variant's bias strictly decreases with the
  distance, "the log function is monotonic" (`logKernel_strictAntiOn`).
-/

@[expose] public section

namespace Transformer.KERPLE

open Finset

variable {ι X E : Type*} [Fintype ι] [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The Softmax normalization `exp(z_n) / Σ_i exp(z_i)` of Eq. 1. -/
noncomputable def softmax (z : ι → ℝ) (n : ι) : ℝ :=
  Real.exp (z n) / ∑ i, Real.exp (z i)

/-- **(*)** (§4, below Eq. 2): the Softmax normalization is shift-invariant,
`exp(x_i) / Σ_j exp(x_j) = exp(x_i + c) / Σ_j exp(x_j + c)` for every `c ∈ ℝ`. -/
theorem softmax_add_const (z : ι → ℝ) (c : ℝ) : softmax (fun i => z i + c) = softmax z := by
  funext n
  simp only [softmax, Real.exp_add, ← sum_mul]
  exact mul_div_mul_right _ _ (Real.exp_pos c).ne'

/-- **Eq. 1**, the kernelized relative positional embedding: the weight `a_{m,n}` of the key `k_n`
for the query `q_m`, for queries and keys in `ℝ^d`, a kernel `k̃` on positions and the positions
`0, …, L − 1`. -/
noncomputable def attention {L d : ℕ} (q k : Fin L → EuclideanSpace ℝ (Fin d))
    (kt : ℝ → ℝ → ℝ) (m n : Fin L) : ℝ :=
  softmax (fun i : Fin L => (inner ℝ (q m) (k i) + kt (m : ℕ) (i : ℕ)) / Real.sqrt d) n

/-- **Eq. 3**, the bias kernel `k = c + k̃`. -/
def biasKernel (c : ℝ) (kt : X → X → ℝ) (x y : X) : ℝ := c + kt x y

/-- **Eq. 4**, the composite kernel `k^comp([q, m], [k, n]) = qᵀk + k(m, n)`, the pair `(q, m)`
standing for `[q, m]`. -/
def compositeKernel (kb : X → X → ℝ) (a b : E × X) : ℝ := inner ℝ a.1 b.1 + kb a.2 b.2

/-- **Eq. 2.** Step (*): the attention of Eq. 1 does not change when `k̃` is replaced by the bias
kernel `c + k̃`, for every `c ∈ ℝ`; and it is the Softmax of the composite kernel with that bias
kernel, divided by `√d`. -/
theorem attention_eq_composite {L d : ℕ} (q k : Fin L → EuclideanSpace ℝ (Fin d))
    (kt : ℝ → ℝ → ℝ) (c : ℝ) (m n : Fin L) :
    attention q k kt m n = attention q k (biasKernel c kt) m n ∧
      attention q k kt m n = softmax (fun i : Fin L => compositeKernel (biasKernel c kt)
        (q m, ((m : ℕ) : ℝ)) (k i, ((i : ℕ) : ℝ)) / Real.sqrt d) n := by
  have h : attention q k (biasKernel c kt) m n = attention q k kt m n := by
    unfold attention
    rw [← softmax_add_const (fun i : Fin L =>
      (inner ℝ (q m) (k i) + kt (m : ℕ) (i : ℕ)) / Real.sqrt d) (c / Real.sqrt d)]
    congr 1
    funext i
    simp only [biasKernel]
    ring
  exact ⟨h.symm, h.symm⟩

/-- Inner products are PD kernels: their form is `‖Σ_i c_i x_i‖²`. -/
theorem isPDKernel_inner : IsPDKernel fun a b : E => inner ℝ a b := by
  refine ⟨fun a b => real_inner_comm b a, fun N x c => ?_⟩
  have h : ∑ i, ∑ j, c i * c j * inner ℝ (x i) (x j) =
      inner ℝ (∑ i, c i • x i) (∑ j, c j • x j) := by
    simp only [sum_inner, inner_sum, real_inner_smul_left, real_inner_smul_right]
    exact sum_congr rfl fun i _ => sum_congr rfl fun j _ => by rw [real_inner_comm]; ring
  rw [h]
  exact real_inner_self_nonneg

/-- **The composite kernel of Eq. 4 is PD exactly when the bias kernel is**: the inner product part
is PD, and at the queries `0` the composite kernel is the bias kernel. -/
theorem isPDKernel_compositeKernel_iff (kb : X → X → ℝ) :
    IsPDKernel (compositeKernel (E := E) kb) ↔ IsPDKernel kb := by
  constructor
  · intro h
    exact (h.comp fun x => ((0 : E), x)).congr fun x y => by simp [compositeKernel]
  · intro h
    exact ((isPDKernel_inner.comp Prod.fst).add (h.comp Prod.snd)).congr fun a b => rfl

/-- The power variant's kernel `k̃(m, n) = −r₁ |m − n|^{r₂}` (§4, Practical Choice). -/
noncomputable def powerKernel (r₁ r₂ x y : ℝ) : ℝ := -r₁ * |x - y| ^ r₂

/-- The logarithmic variant's kernel `k̃(m, n) = −r₁ log(1 + r₂ |m − n|)` (§4, Practical Choice). -/
noncomputable def logKernel (r₁ r₂ x y : ℝ) : ℝ := -r₁ * Real.log (1 + r₂ * |x - y|)

/-- **The power variant is CPD** for `r₁ > 0` and `0 < r₂ ≤ 2`: Corollary 1(a) on `ℝ`. -/
theorem isCPDKernel_powerKernel {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (hr₂2 : r₂ ≤ 2) :
    IsCPDKernel (powerKernel r₁ r₂) :=
  (isCPDKernel_neg_mul_norm_rpow (E := ℝ) hr₁.le hr₂ hr₂2).congr fun x y => by
    rw [Real.norm_eq_abs, powerKernel]

/-- The hypotheses of `isCPDKernel_powerKernel` hold for `r₁ = r₂ = 1`. -/
example : IsCPDKernel (powerKernel 1 1) := isCPDKernel_powerKernel one_pos one_pos one_le_two

/-- **The logarithmic variant is CPD** for `r₁, r₂ > 0`: Corollary 1(b) on `ℝ` with `p = 1`. -/
theorem isCPDKernel_logKernel {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) :
    IsCPDKernel (logKernel r₁ r₂) :=
  (isCPDKernel_neg_mul_log_one_add (E := ℝ) hr₂.le hr₁.le one_pos one_le_two).congr fun x y => by
    rw [Real.norm_eq_abs, Real.rpow_one, logKernel]

/-- The hypotheses of `isCPDKernel_logKernel` hold for `r₁ = r₂ = 1`. -/
example : IsCPDKernel (logKernel 1 1) := isCPDKernel_logKernel one_pos one_pos

/-- **Connection to ALiBi** (§4): with the bias kernel `c − r₁|m − n|`, the power variant at
`r₂ = 1`, KERPLE's attention is ALiBi's, the Softmax of `q_mᵀk_i/√d − s|m − i|` with the slope
`s = r₁/√d`; the constant `c` drops out. ALiBi's bias `−s|j − i|` is `Transformer.BAM.alibi`, and
the paper's triangle kernel `c − |m − n|` is `r₁ = 1`. -/
theorem attention_powerKernel_one {L d : ℕ} (q k : Fin L → EuclideanSpace ℝ (Fin d)) (c r₁ : ℝ)
    (m n : Fin L) :
    attention q k (biasKernel c (powerKernel r₁ 1)) m n = softmax (fun i : Fin L =>
      inner ℝ (q m) (k i) / Real.sqrt d - r₁ / Real.sqrt d * |((m : ℕ) : ℝ) - (i : ℕ)|) n := by
  rw [← (attention_eq_composite q k (powerKernel r₁ 1) c m n).1]
  unfold attention
  congr 1
  funext i
  simp only [powerKernel, Real.rpow_one]
  ring

/-- **The logarithmic variant is monotonic** (§4, "the log function is monotonic"): its kernel
`−r₁ log(1 + r₂ t)` strictly decreases in the distance `t = |m − n| ≥ 0`, so it cannot learn
relations that are not monotonic in the distance. -/
theorem logKernel_strictAntiOn {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) :
    StrictAntiOn (fun t => logKernel r₁ r₂ t 0) (Set.Ici 0) := by
  intro s hs t ht hst
  simp only [logKernel, sub_zero, abs_of_nonneg (Set.mem_Ici.mp hs),
    abs_of_nonneg (Set.mem_Ici.mp ht), neg_mul, neg_lt_neg_iff]
  have hs' : 0 < 1 + r₂ * s := by nlinarith [Set.mem_Ici.mp hs]
  exact mul_lt_mul_of_pos_left (Real.log_lt_log hs' (by nlinarith)) hr₁

/-- The hypotheses of `logKernel_strictAntiOn` hold for `r₁ = r₂ = 1`. -/
example : StrictAntiOn (fun t => logKernel 1 1 t 0) (Set.Ici 0) :=
  logKernel_strictAntiOn one_pos one_pos

end Transformer.KERPLE

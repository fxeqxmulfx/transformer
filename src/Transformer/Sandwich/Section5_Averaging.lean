module

public import Transformer.Sandwich.Section3_Biases
public import Transformer.Sandwich.Section5_Locality

/-!
# §5 The averaging model

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §5: the TRF is "the maximum number of tokens that
contribute to the prediction of the next token". The maximum is attained: in the averaging model
every token of the window `R(w − 1) + 1` contributes (`avgModel_contributes` in
`Section5_ReceptiveField`).

The averaging model has one head with zero queries and keys and the identity as values, so the
weights `a_{m,n}` depend on the bias alone. Started from `e_n = (1, −1)` and `e_j = 0` elsewhere,
its hidden states stay of the form `(c_j, −c_j)` with `c_j ≥ 0`: the head averages them with
nonnegative weights, and LayerNorm maps `(c, −c)` to `(c, −c)/√(c² + ε)`. Each layer spreads the
positions where `c_j > 0` by `w − 1` (`Spread.avgLayer_apply`), and at the last one `c_m > 0`
makes the first next-token probability exceed the second, while the input `0` gives them
equal.
-/

@[expose] public section

namespace Transformer.Sandwich

open Finset Matrix Transformer.BAM

variable {w R n K : ℕ}

/-- The averaging head: zero queries and keys, the identity as values. Every key in the window
gets the same weight. -/
noncomputable def avgHead : Head 2 2 := ⟨0, 0, 1⟩

/-- The averaging attention: one averaging head, `W_o` the identity on its output. -/
noncomputable def avgAttn : Attn 2 2 1 :=
  ⟨fun _ => avgHead, Matrix.of fun i p => if p.2 = i then 1 else 0⟩

/-- The averaging layer: the averaging attention, LayerNorm with gain `1` and bias `0`. -/
noncomputable def avgLayer : Layer 2 2 1 := ⟨avgAttn, 1, 0⟩

/-- The averaging model with `R ≥ 1` layers, `ε = 1` and `V` the identity. -/
noncomputable def avgModel (R : ℕ) : Model 2 2 1 2 :=
  ⟨List.replicate (R - 1) avgLayer, avgAttn, 1, 1⟩

/-- The averaging model has `R` layers. -/
theorem avgModel_depth (hR : 1 ≤ R) : (avgModel R).depth = R := by
  simp only [Model.depth, avgModel, List.length_replicate]
  omega

/-- The hypothesis of `avgModel_depth` holds for one layer. -/
example : (avgModel 1).depth = 1 := avgModel_depth le_rfl

/-- The averaging attention's output: the average of the inputs with the weights
`exp(b(m − i)) / Σ_j exp(b(m − j))`. -/
theorem avgAttn_out_apply (b : ℕ → EReal) (x : Embs 2) (m : ℕ) (c : Fin 2) :
    avgAttn.out (fun _ => b) x m c = ∑ i ∈ range (m + 1),
      weight (b (m - i)) / (∑ j ∈ range (m + 1), weight (b (m - j))) * x i c := by
  have hs : ∀ i, score avgHead b x m i = b (m - i) := fun i => by simp [score, avgHead]
  have hhead : headOut avgHead b x m c = ∑ i ∈ range (m + 1),
      weight (b (m - i)) / (∑ j ∈ range (m + 1), weight (b (m - j))) * x i c := by
    have hv : avgHead.Wv = 1 := rfl
    simp only [headOut, attnWeight, hs, hv, one_mulVec, Finset.sum_apply, Pi.smul_apply,
      smul_eq_mul]
  rw [← hhead]
  fin_cases c <;> simp [Attn.out, avgAttn, mulVec, dotProduct, Fintype.sum_prod_type]

/-- LayerNorm with gain `1`, bias `0` and `ε = 1` maps `(c, −c)` to `(c, −c)/√(c² + 1)`. -/
theorem layerNorm_antisymm {o : Fin 2 → ℝ} (ho : o 1 = -o 0) (i : Fin 2) :
    layerNorm 1 1 0 o i = o i / Real.sqrt (o 0 ^ 2 + 1) := by
  have hmean : lnMean o = 0 := by simp [lnMean, Fin.sum_univ_two, ho]
  have hvar : lnVar o = o 0 ^ 2 := by
    simp only [lnVar, hmean, sub_zero, Fin.sum_univ_two, ho, neg_sq]
    push_cast
    ring
  simp [layerNorm, hmean, hvar]

/-- The hypothesis of `layerNorm_antisymm` holds for `(1, −1)`. -/
example : layerNorm 1 1 0 ![1, -1] 0 = ![1, -1] 0 / Real.sqrt (![(1 : ℝ), -1] 0 ^ 2 + 1) :=
  layerNorm_antisymm (by simp) 0

/-- The sequence is `(c_j, −c_j)` with `c_j ≥ 0`, and `c_j > 0` at the positions
`n, …, n + K`. -/
def Spread (n K : ℕ) (z : Embs 2) : Prop :=
  (∀ j, z j 1 = -z j 0 ∧ 0 ≤ z j 0) ∧ ∀ j, n ≤ j → j ≤ n + K → 0 < z j 0

/-- **The averaging attention spreads the positive positions by `w − 1`** when the bias is
finite at the distances `< w`. -/
theorem Spread.avgAttn_out {b : ℕ → EReal} (hb : ∀ t, t < w → b t ≠ ⊥ ∧ b t ≠ ⊤) (hw : 1 ≤ w)
    {z : Embs 2} (hz : Spread n K z) : Spread n (K + (w - 1)) (avgAttn.out (fun _ => b) z) := by
  have hpos : ∀ t, t < w → 0 < weight (b t) := fun t ht => by
    rw [← EReal.coe_toReal (hb t ht).2 (hb t ht).1, weight_coe]
    exact Real.exp_pos _
  have hZ : ∀ m, 0 < ∑ j ∈ range (m + 1), weight (b (m - j)) := fun m => by
    have h := single_le_sum (f := fun j => weight (b (m - j))) (fun j _ => weight_nonneg _)
      (mem_range.2 (Nat.lt_succ_self m))
    simp only [Nat.sub_self] at h
    exact lt_of_lt_of_le (hpos 0 (by omega)) h
  have hterm : ∀ m i, 0 ≤ weight (b (m - i)) / (∑ j ∈ range (m + 1), weight (b (m - j))) * z i 0 :=
    fun m i => mul_nonneg (div_nonneg (weight_nonneg _) (hZ m).le) (hz.1 i).2
  refine ⟨fun m => ⟨?_, ?_⟩, fun m hnm hm => ?_⟩
  · rw [avgAttn_out_apply, avgAttn_out_apply, ← sum_neg_distrib]
    exact sum_congr rfl fun i _ => by rw [(hz.1 i).1, mul_neg]
  · rw [avgAttn_out_apply]
    exact sum_nonneg fun i _ => hterm m i
  · rw [avgAttn_out_apply]
    exact sum_pos' (fun i _ => hterm m i) ⟨min m (n + K), mem_range.2 (by omega),
      mul_pos (div_pos (hpos _ (by omega)) (hZ m)) (hz.2 _ (by omega) (by omega))⟩

/-- **The averaging layer spreads the positive positions by `w − 1`**: LayerNorm keeps them. -/
theorem Spread.avgLayer_apply {b : ℕ → EReal} (hb : ∀ t, t < w → b t ≠ ⊥ ∧ b t ≠ ⊤) (hw : 1 ≤ w)
    {z : Embs 2} (hz : Spread n K z) :
    Spread n (K + (w - 1)) (avgLayer.apply 1 (fun _ => b) z) := by
  have h := hz.avgAttn_out hb hw
  have hln : ∀ m i, avgLayer.apply 1 (fun _ => b) z m i = avgAttn.out (fun _ => b) z m i /
      Real.sqrt (avgAttn.out (fun _ => b) z m 0 ^ 2 + 1) := fun m i =>
    layerNorm_antisymm (h.1 m).1 i
  have hs : ∀ m, 0 < Real.sqrt (avgAttn.out (fun _ => b) z m 0 ^ 2 + 1) := fun m =>
    Real.sqrt_pos.2 (by positivity)
  refine ⟨fun m => ⟨?_, ?_⟩, fun m hnm hm => ?_⟩
  · rw [hln, hln, (h.1 m).1, neg_div]
  · rw [hln]
    exact div_nonneg (h.1 m).2 (hs m).le
  · rw [hln]
    exact div_pos (h.2 m hnm hm) (hs m)

/-- `ℓ` averaging layers spread the positive positions by `ℓ(w − 1)`. -/
theorem Spread.hidden_avgLayer {b : ℕ → EReal} (hb : ∀ t, t < w → b t ≠ ⊥ ∧ b t ≠ ⊤) (hw : 1 ≤ w)
    (ℓ : ℕ) {z : Embs 2} (hz : Spread n K z) :
    Spread n (K + ℓ * (w - 1)) (hidden 1 (fun _ => b) (List.replicate ℓ avgLayer) z) := by
  induction ℓ generalizing K z with
  | zero => simpa [hidden] using hz
  | succ ℓ ih =>
    rw [show K + (ℓ + 1) * (w - 1) = K + (w - 1) + ℓ * (w - 1) by ring]
    exact ih (hz.avgLayer_apply hb hw)

/-- The averaging layers map the input `0` to `0`. -/
theorem hidden_avgLayer_zero (b : ℕ → EReal) (ℓ : ℕ) :
    hidden 1 (fun _ => b) (List.replicate ℓ avgLayer) 0 = 0 := by
  have h0 : ∀ m c, avgAttn.out (fun _ => b) 0 m c = 0 := fun m c => by simp [avgAttn_out_apply]
  have hl : avgLayer.apply 1 (fun _ => b) 0 = 0 := by
    funext m i
    rw [show avgLayer.apply 1 (fun _ => b) 0 m i = layerNorm 1 1 0 (avgAttn.out (fun _ => b) 0 m) i
      from rfl, layerNorm_antisymm (by simp [h0]) i]
    simp [h0]
  induction ℓ with
  | zero => rfl
  | succ ℓ ih =>
    show hidden 1 (fun _ => b) (List.replicate ℓ avgLayer) (avgLayer.apply 1 (fun _ => b) 0) = 0
    rw [hl, ih]

/-- The input `e_n = (1, −1)` and `e_j = 0` elsewhere is spread at `n`. -/
theorem spread_single (n : ℕ) : Spread n 0 fun j => if j = n then ![1, -1] else 0 := by
  refine ⟨fun j => ?_, fun j h1 h2 => ?_⟩
  · by_cases hj : j = n <;> simp [hj]
  · simp [show j = n by omega]

/-- The hypotheses of `Spread.avgAttn_out` hold for the bias `0` and `w = 1`. -/
example : Spread 0 (0 + (1 - 1))
    (avgAttn.out (fun _ _ => 0) fun j => if j = 0 then ![1, -1] else 0) :=
  (spread_single 0).avgAttn_out (w := 1) (fun _ _ => ⟨by simp, by simp⟩) le_rfl

/-- The hypotheses of `Spread.avgLayer_apply` hold for the bias `0` and `w = 1`. -/
example : Spread 0 (0 + (1 - 1))
    (avgLayer.apply 1 (fun _ _ => 0) fun j => if j = 0 then ![1, -1] else 0) :=
  (spread_single 0).avgLayer_apply (w := 1) (fun _ _ => ⟨by simp, by simp⟩) le_rfl

/-- The hypotheses of `Spread.hidden_avgLayer` hold for two layers, the bias `0` and `w = 1`. -/
example : Spread 0 (0 + 2 * (1 - 1))
    (hidden 1 (fun _ _ => 0) (List.replicate 2 avgLayer) fun j => if j = 0 then ![1, -1] else 0) :=
  (spread_single 0).hidden_avgLayer (w := 1) (fun _ _ => ⟨by simp, by simp⟩) le_rfl 2

end Transformer.Sandwich

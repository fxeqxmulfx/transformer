module

public import Transformer.SequenceModel.Attention.Stability.Averages
public import Transformer.CRASP.Rotary.Model

/-!
# Vector error at the actual rounded attention averages

Source: `review/flat_spectrum.md`, *Full-stack rounding certificates*, B5.
Natural one-based counting prefixes are first identified with the source's
finite causal tensor. Range premises inspect coefficient and quotient
rounding sites themselves; they assume neither hidden-state error nor classification.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open CRASP GPTMini
open scoped BigOperators

/-- B5: the natural prefix is exactly the finite causal mask, with no sentinel. -/
theorem sum_prefix {M : Type*} [AddCommMonoid M] {T : ℕ} (i : Fin T) (f : ℕ → M) :
    ∑ j ∈ Finset.Icc 1 (i.val + 1), f j = ∑ j ∈ Finset.Iic i, f (j.val + 1) := by
  symm
  apply Finset.sum_bij (fun j _ => j.val + 1)
  · intro j hj
    have hj' : j ≤ i := Finset.mem_Iic.mp hj
    exact Finset.mem_Icc.mpr ⟨by omega, by exact Nat.add_le_add_right hj' 1⟩
  · intro a ha b hb hab
    apply Fin.ext
    omega
  · intro j hj
    have hj' := Finset.mem_Icc.mp hj
    let k : Fin T := ⟨j - 1, by have := i.isLt; omega⟩
    refine ⟨k, Finset.mem_Iic.mpr ?_, ?_⟩
    · change j - 1 ≤ i.val
      omega
    · change j - 1 + 1 = j
      omega
  · intro j hj
    rfl

/-- B5: the real ratio used by the rounding analysis is the corresponding
coordinate of the actual source softmax average on the decoded stream. -/
theorem sourceAverage_coordinate (cfg : Config) (B : BlockParams cfg) (eps : ℝ)
    {T p s : ℕ} (a : ℕ → Fin (cfg.head_dim / 2) → ℝ)
    (q : ℕ → Rotary.State cfg p s) (i : Fin T) (c : Normalized.Coordinate cfg) :
    Normalized.sourceAverage cfg B eps (fun j : Fin T => a j.val)
      (fun j => Normalized.vector (q (j.val + 1)).2) i c.1 c.2 =
    (∑ j ∈ Finset.Icc 1 (i.val + 1),
      Rotary.sourceWeight cfg B eps a (q (i.val + 1)) (q j) c.1 (i.val + 1) j *
        Normalized.component cfg B eps (qkvV cfg) (Normalized.vector (q j).2) c.1 c.2) /
    (∑ j ∈ Finset.Icc 1 (i.val + 1),
      Rotary.sourceWeight cfg B eps a (q (i.val + 1)) (q j) c.1 (i.val + 1) j) := by
  rw [sum_prefix, sum_prefix]
  simp only [Rotary.sourceWeight, Nat.add_sub_cancel, Normalized.sourceAverage,
    WithLp.ofLp_sum, Finset.sum_apply, PiLp.smul_apply, smul_eq_mul,
    causalWeight, ite_mul, zero_mul]
  rw [← Finset.sum_filter]
  have hf : Finset.univ.filter (fun j : Fin T => j ≤ i) = Finset.Iic i := by
    ext j
    simp
  rw [hf, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro j hj
  ring

/-- B5: an actual token list supplies exactly the intended rounded embedding. -/
theorem input_ofFn (cfg : Config) (θ : ModelParams cfg) (eps : ℝ) (p s : ℕ)
    {T : ℕ} (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T) :
    Rotary.input (Rotary.embed cfg θ eps p s) (false, fun _ => 0)
      (List.ofFn tokens) (i.val + 1) =
      Rotary.embed cfg θ eps p s (decide (i.val = 0)) (tokens i) := by
  simp only [Rotary.input, Nat.add_sub_cancel, List.getElem?_ofFn, i.isLt,
    dite_true, Fin.eta, Nat.add_eq_right]

example : (∑ j ∈ Finset.Icc (1 : ℕ) 3, (j : ℝ)) =
    ∑ j ∈ Finset.Iic (2 : Fin 4), (j.val + 1 : ℝ) := by
  simpa using sum_prefix (2 : Fin 4) (fun j => (j : ℝ))

open CRASP GPTMini

/-- No clipping at any coefficient or quotient of this actual prefix aggregate. -/
def AggregateInRange (cfg : Config) (B : BlockParams cfg) (eps : ℝ) (p s : ℕ)
    (a : ℕ → Fin (cfg.head_dim / 2) → ℝ) (q : ℕ → Rotary.State cfg p s) (i : ℕ) : Prop :=
  ∀ c : Normalized.Coordinate cfg,
    let weights := fun j => Rotary.sourceWeight cfg B eps a (q i) (q j) c.1 i j
    let values := fun j => Normalized.component cfg B eps (qkvV cfg)
      (Normalized.vector (q j).2) c.1 c.2
    (∀ j ∈ Finset.Icc 1 i, Fx.NoSaturation p s (weights j) ∧
      Fx.NoSaturation p s (weights j * values j)) ∧
    Fx.NoSaturation p s (roundedRatio (Finset.Icc 1 i) p s weights values)

/-- Per-coordinate attention error; pre-RMSNorm supplies the value bound. -/
def aggregateRadius (cfg : Config) (B : BlockParams cfg) (s : ℕ) : ℝ :=
  (1 / (2 : ℝ) ^ s) * (1 + valueBound cfg B) /
    (Real.exp (-Real.sqrt (cfg.head_dim : ℝ)) - 1 / (2 : ℝ) ^ s) + 1 / (2 : ℝ) ^ s

/-- B5: the required precision gives a nonnegative aggregate error budget. -/
theorem aggregateRadius_nonneg (cfg : Config) (B : BlockParams cfg) (s : ℕ)
    (hp : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ))) :
    0 ≤ aggregateRadius cfg B s := by
  have hd := sub_pos.mpr hp
  unfold aggregateRadius valueBound
  positivity

/-- B5: the actual rounded aggregate is close to the source softmax average
on the decoded residual stream, in the head's Euclidean norm. -/
theorem aggregate_vector_error (cfg : Config) (B : BlockParams cfg) {eps : ℝ}
    (heps : 0 < eps) (xeps : ℝ) (p s : ℕ) {T : ℕ}
    (a : ℕ → Fin (cfg.head_dim / 2) → ℝ) (q : ℕ → Rotary.State cfg p s)
    (i : Fin T) (h : Fin cfg.n_heads)
    (hp : 1 / (2 : ℝ) ^ s < Real.exp (-Real.sqrt (cfg.head_dim : ℝ)))
    (hr : AggregateInRange cfg B eps p s a q (i.val + 1)) :
    ‖Normalized.vector (fun c =>
        (Rotary.sourceLayer cfg B eps xeps p s a).aggregate (q (i.val + 1)) q (i.val + 1) (h, c)) -
      Normalized.sourceAverage cfg B eps (fun j : Fin T => a j.val)
        (fun j => Normalized.vector (q (j.val + 1)).2) i h‖ ≤
      Real.sqrt (cfg.head_dim : ℝ) * aggregateRadius cfg B s := by
  apply norm_le_coordinates _ _ (aggregateRadius_nonneg cfg B s hp)
  intro c
  change |((Rotary.sourceLayer cfg B eps xeps p s a).aggregate
      (q (i.val + 1)) q (i.val + 1) (h, c)).val -
      Normalized.sourceAverage cfg B eps (fun j : Fin T => a j.val)
        (fun j => Normalized.vector (q (j.val + 1)).2) i h c| ≤ _
  rw [sourceAverage_coordinate cfg B eps a q i (h, c)]
  exact source_aggregate_error cfg B heps xeps p s a (q (i.val + 1)) q
    (i.val + 1) (by omega) (h, c) (valueBound cfg B) hp
    (fun j hj => (PiLp.norm_apply_le _ c).trans
      (component_norm cfg B heps _ (qkvV_injective cfg) _ h))
    (fun j hj => ((hr (h, c)).1 j hj).1)
    (fun j hj => ((hr (h, c)).1 j hj).2) (hr (h, c)).2

end Transformer.SequenceModel.Attention.Stability

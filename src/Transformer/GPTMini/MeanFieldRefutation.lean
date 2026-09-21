/-
# `mean_field_clustering` refuted

The mean-field form of `layer_clustering` was stated with the Wasserstein
distance `W₂` as a free parameter, since Mathlib has none.  Quantified over
every `W₂` it is false (`not_mean_field_clustering`): with one token the head
only rescales the stream by a factor `≥ 1` (`attentionHead_one`,
`preLNHead_one_ne_zero`), so the trajectory from any `x₀ ≠ 0` is admissible,
and `W₂ ≡ 1` never tends to `0`.

What is refuted is this development's own transfer of the survey's mean-field
clustering to the discrete pre-LN `gpt-mini` layer, not a claim of the survey.
The survey's statement, `thm:mfclust`, is about the continuity equation
`eq:continuity`; it is stated with `Transformer.Wasserstein.W2` in
`Transformer.MeanField.Clustering`.

Source: arXiv:2512.01868v4, `sec:clustering`, `thm:mfclust` (the claim transferred).
-/

import Transformer.GPTMini.ClusteringTheorem

open Real MeasureTheory

namespace Transformer
namespace GPTMini

variable (cfg : Config)

/-- **The head on a single token**: the causal softmax has one weight, `1`, and
XSA leaves `(1 - (‖v‖/max(‖v‖,eps))²) v`.  `CausalMHA.attentionHead`. -/
theorem attentionHead_one (alpha eps : ℝ) (q k v : Fin 1 → EucSpace cfg.head_dim)
    (positions : Fin 1 → ℝ) :
    attentionHead cfg alpha eps q k v positions 0
      = (1 - (‖v 0‖ / max ‖v 0‖ eps) ^ 2) • v 0 := by
  have hw := causalAttnWeights_row_sum cfg alpha eps
    (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (q j))
    (fun j => applyRope cfg.head_dim cfg.rope_theta (positions j) (k j)) 0
  rw [Fin.sum_univ_one] at hw
  unfold attentionHead attnOutput xsaProjection
  simp only [Fin.sum_univ_one, ite_true]
  rw [hw, one_smul, normL2, inner_smul_right, real_inner_self_eq_norm_sq, smul_smul,
    ← one_smul ℝ (v 0), smul_smul, ← sub_smul, one_smul]
  congr 1
  ring

/-- With one token and `eps ≥ 0` the recursion only rescales the stream by a
factor `≥ 1`: it never reaches `0` from `x₀ ≠ 0`. -/
theorem preLNHead_one_ne_zero (alpha eps : ℝ) (heps : 0 ≤ eps) (positions : Fin 1 → ℝ)
    (x : ℕ → Fin 1 → EucSpace cfg.head_dim) (hx : PreLNHead cfg alpha eps positions x)
    (h0 : x 0 0 ≠ 0) (L : ℕ) (i : Fin 1) : x L i ≠ 0 := by
  rw [Fin.fin_one_eq_zero i]
  induction L with
  | zero => exact h0
  | succ L ih =>
    set u := rmsNormEps eps (x L 0)
    have hu : u = (Real.sqrt cfg.head_dim / Real.sqrt (‖x L 0‖ ^ 2 + cfg.head_dim * eps)) •
        x L 0 := rfl
    have ht : (‖u‖ / max ‖u‖ eps) ^ 2 ≤ 1 := by
      rcases (norm_nonneg u).eq_or_lt with h | h
      · rw [← h]; simp
      · exact pow_le_one₀ (by positivity)
          ((div_le_one (lt_max_of_lt_left h)).2 (le_max_left _ _))
    rw [hx, attentionHead_one]
    change x L 0 + (1 - (‖u‖ / max ‖u‖ eps) ^ 2) • u ≠ 0
    generalize (‖u‖ / max ‖u‖ eps) ^ 2 = t at ht ⊢
    rw [hu, smul_smul, ← one_smul ℝ (x L 0), smul_smul, ← add_smul, one_smul]
    refine smul_ne_zero (ne_of_gt ?_) ih
    have : 0 ≤ (1 - t) * (Real.sqrt cfg.head_dim / Real.sqrt (‖x L 0‖ ^ 2 + cfg.head_dim * eps)) :=
      mul_nonneg (by linarith) (by positivity)
    linarith

/-- The hypotheses of `preLNHead_one_ne_zero` are satisfiable: the recursion
started at a non-zero stream. -/
example : ∃ x : ℕ → Fin 1 → EucSpace Config.default.head_dim,
    PreLNHead Config.default 0 1 0 x ∧ x 0 0 ≠ 0 := by
  have : NeZero Config.default.head_dim := ⟨by norm_num [Config.head_dim, Config.default]⟩
  refine ⟨fun L => Nat.rec (fun _ => EuclideanSpace.single 0 1) (fun _ y i => y i +
    attentionHead Config.default 0 1 (fun j => rmsNormEps 1 (y j)) (fun j => rmsNormEps 1 (y j))
      (fun j => rmsNormEps 1 (y j)) 0 i) L, fun _ _ => rfl, ?_⟩
  simp

/-- **`mean_field_clustering` is false as written.**

The statement quantifies over an arbitrary `W₂`, which nothing constrains to be
a Wasserstein distance.  With `W₂ ≡ 1`, one token, `alpha = 0`, `eps = 1` and
the default config, the trajectory from any `x₀ ≠ 0` stays away from the origin
(`preLNHead_one_ne_zero`), so the hypotheses hold on the complement of a
Lebesgue-null set, while `W₂ ≡ 1` never tends to `0`.

The survey's own `thm:mfclust` is not touched: it is about the continuity
equation, and is stated in `Transformer.MeanField.Clustering`.

Source: arXiv:2512.01868v4, `thm:mfclust` (the claim transferred). -/
theorem not_mean_field_clustering :
    ¬ ∀ (cfg : Config)
      (W₂ : Measure (EucSpace cfg.head_dim) → Measure (EucSpace cfg.head_dim) → ℝ)
      {T : ℕ}, 3 ≤ cfg.head_dim → 0 < T →
      ∀ (alpha eps : ℝ) (positions : Fin T → ℝ), 0 < eps →
      ∀ᵐ x₀ : Fin T → EucSpace cfg.head_dim, ∀ x : ℕ → Fin T → EucSpace cfg.head_dim,
        x 0 = x₀ → PreLNHead cfg alpha eps positions x → (∀ (L : ℕ) (i : Fin T), x L i ≠ 0) →
          ∃ xinf : EucSpace cfg.head_dim, ‖xinf‖ = 1 ∧
            Filter.Tendsto
              (fun L : ℕ =>
                W₂ (((T : ℝ)⁻¹).toNNReal •
                    ∑ i : Fin T, Measure.dirac (Bridge.toSphere cfg.head_dim (x L i)))
                  (Measure.dirac xinf))
              Filter.atTop (nhds (0 : ℝ)) := by
  intro h
  set cfg := Config.default
  have hd : 3 ≤ cfg.head_dim := by norm_num [cfg, Config.head_dim, Config.default]
  have : NeZero cfg.head_dim := ⟨by omega⟩
  have hae := h cfg (fun _ _ => 1) (T := 1) hd one_pos 0 1 0 one_pos
  have hne : ∀ᵐ x₀ : Fin 1 → EucSpace cfg.head_dim, x₀ ∉ ({0} : Set _) := by
    have := Measure.IsAddHaarMeasure.nullSingletonClass
      (volume : Measure (Fin 1 → EucSpace cfg.head_dim))
    exact compl_mem_ae_iff.2 (measure_singleton _)
  obtain ⟨x₀, hx₀, hne⟩ := (hae.and hne).exists
  have h0 : x₀ 0 ≠ 0 := fun h0 => hne (funext fun i => by rw [Fin.fin_one_eq_zero i]; exact h0)
  let x : ℕ → Fin 1 → EucSpace cfg.head_dim := fun L => Nat.rec x₀ (fun _ y i => y i +
    attentionHead cfg 0 1 (fun j => rmsNormEps 1 (y j)) (fun j => rmsNormEps 1 (y j))
      (fun j => rmsNormEps 1 (y j)) 0 i) L
  have hx : PreLNHead cfg 0 1 0 x := fun _ _ => rfl
  obtain ⟨_, -, ht⟩ := hx₀ x rfl hx (preLNHead_one_ne_zero cfg 0 1 zero_le_one 0 x hx h0)
  exact one_ne_zero (tendsto_nhds_unique tendsto_const_nhds ht)

end GPTMini
end Transformer

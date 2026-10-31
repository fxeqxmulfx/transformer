module

public import Transformer.SequenceModel.Attention.Transfer

/-!
# A genuine obstruction to unconditional optimizer convergence

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, S2.
At zero weights the tied readout and final hidden state both vanish.
Every logit derivative is therefore zero although CE is positive for
multiple classes. This is an actual smooth point of clipped XSA. More
width, depth or training steps do not remove this stationary initialization.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini Landscape.Neural ScheduleFree
open scoped BigOperators

/-- S2: zero output projections make the real normalized residual block an identity. -/
theorem block_zero (cfg : Config) (d : Fin cfg.n_layers) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x : Fin T → EucSpace cfg.d_model) :
    block cfg ((decode cfg 0).blocks d) eps xeps angle x = x := by
  funext i
  have hproj : ((decode cfg 0).blocks d).projection = 0 := by
    change matrixEquiv _ _ (0 : Matrix _ _ ℝ) = 0
    exact map_zero _
  have hout : ((decode cfg 0).blocks d).output = 0 := by
    change matrixEquiv _ _ (0 : Matrix _ _ ℝ) = 0
    exact map_zero _
  simp [block, attention, feedForward, hproj, hout, rmsNormEps]

/-- S2: the full hidden state is zero at every finite depth and position. -/
theorem hidden_zero (cfg : Config) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (n : ℕ) (j : Fin T) :
    hidden cfg (decode cfg 0) eps xeps angle tokens n j = 0 := by
  induction n with
  | zero =>
    change rmsNormEps eps (0 : EucSpace cfg.d_model) = 0
    simp [rmsNormEps]
  | succ n ih =>
    simp only [hidden]
    split_ifs with hn
    · rw [block_zero]
      exact ih
    · exact ih

/-- S2: all actual logits tie at zero, independently of angles and input. -/
theorem score_zero (cfg : Config) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (j : Fin T) (tokens : Fin T → Fin cfg.vocab_size) (v : Fin cfg.vocab_size) :
    score cfg eps xeps angle j 0 tokens v = 0 := by
  change inner ℝ _ (0 : EucSpace cfg.d_model) = 0
  exact inner_zero_right _

/-- S2: the logit derivative is actually zero, not an arbitrary total-gradient
value assigned at a nondifferentiable point. Both tied factors vanish. -/
theorem hasFDerivAt_score_zero (cfg : Config) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (j : Fin T) (v : Fin cfg.vocab_size) :
    HasFDerivAt (𝕜 := ℝ) (fun w => score cfg eps xeps angle j w tokens v) 0 0 := by
  let H := fun w => rmsNormEps eps (hidden cfg (decode cfg w) eps xeps angle tokens cfg.n_layers j)
  let E := fun w => (decode cfg w).embedding v
  have hH : ∀ i, AnalyticAt ℝ (fun w => H w i) 0 :=
    analyticAt_rmsNormEps cfg.d_model_pos heps
      (analyticAt_hidden cfg heps hxeps angle tokens 0
        (regular_zero cfg eps hxeps angle tokens) cfg.n_layers j)
  have hE : ∀ i, AnalyticAt ℝ (fun w => E w i) 0 :=
    fun i => analyticAt_parameter (I := Index cfg) (.inl (v, i)) 0
  have hHz : H 0 = 0 := by simp [H, hidden_zero, rmsNormEps]
  have hEz : E 0 = 0 := rfl
  have hp : ∀ i, HasFDerivAt (𝕜 := ℝ) (fun w => E w i * H w i) 0 0 := by
    intro i
    simpa only [hHz, hEz, PiLp.zero_apply, zero_smul, add_zero] using
      (hE i).differentiableAt.hasFDerivAt.fun_mul
      (hH i).differentiableAt.hasFDerivAt
  have hsum := HasFDerivAt.fun_sum (u := Finset.univ) (fun i _ => hp i)
  simpa [score, EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Pi.star_apply,
    star_trivial, H, E] using hsum

example := hasFDerivAt_score_zero witnessConfig (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (sourceAngles witnessConfig (T := 3) (fun _ => true) (fun j => j.val)) (fun j => j) 2 1

/-- S2: the entire actual logit table has zero Jacobian at zero weights. -/
theorem hasFDerivAt_table_zero (cfg : Config) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ} {A : Type*} [Fintype A]
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T) :
    HasFDerivAt (𝕜 := ℝ) (table cfg eps xeps angle inputs readout) 0 0 := by
  let C := (PiLp.continuousLinearEquiv 2 ℝ (fun _ : A × Fin cfg.vocab_size => ℝ)).symm
  have hp : HasFDerivAt (𝕜 := ℝ) (fun w : Parameters cfg => fun av : A × Fin cfg.vocab_size =>
      score cfg eps xeps angle (readout av.1) w (inputs av.1) av.2) 0 0 := by
    apply hasFDerivAt_pi''
    intro av
    simpa using hasFDerivAt_score_zero cfg heps hxeps angle (inputs av.1) (readout av.1) av.2
  have h := C.toContinuousLinearMap.hasFDerivAt.comp 0 hp
  convert! h using 1

/-- S2: ordinary sampled cross-entropy has zero full parameter gradient at
this smooth tied-model initialization, for any finite input table and labels. -/
theorem loss_gradient_zero (cfg : Config) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ} {A : Type*} [Fintype A]
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (rows : Finset A) (target : A → Fin cfg.vocab_size) :
    gradient (loss (predictor cfg eps xeps angle inputs readout) rows target) 0 = 0 := by
  rw [actual_gradient cfg heps hxeps angle inputs readout rows target 0
    (fun _ => regular_zero cfg eps hxeps angle _)]
  rw [(hasFDerivAt_table_zero cfg heps hxeps angle inputs readout).fderiv]
  simp

example := hasFDerivAt_table_zero witnessConfig (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (sourceAngles witnessConfig (T := 3) (fun _ => true) (fun j => j.val))
  (fun a j : Fin 3 => if a = j then 0 else 1) (fun _ => 2)
example := loss_gradient_zero witnessConfig (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  (sourceAngles witnessConfig (T := 3) (fun _ => true) (fun j => j.val))
  (fun a j : Fin 3 => if a = j then 0 else 1) (fun _ => 2) {0, 1} id

end Transformer.SequenceModel.Attention

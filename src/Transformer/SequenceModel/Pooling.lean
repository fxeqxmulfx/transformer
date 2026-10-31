module

public import Transformer.SequenceModel.Rotary

/-!
# Arbitrary positional mixing and finite multivariable rotary pooling

Source: `review/flat_spectrum.md`, *General positional models*, G2--G3.
Position transforms may be rotations, other linear maps, or nonlinear maps.
The pooled predictor retains the actual normalized SwiGLU residual stack.
Only the repeated quarter-turn special case has the five-position collision
proved here. No such collision is asserted for arbitrary rotary phases, and
attention on the unpooled stream is not subject to this pooling premise.
-/

@[expose] public section

noncomputable section

namespace Transformer.SequenceModel

open Landscape.Neural Rotate90
open scoped BigOperators

variable {P V K H D : Type*} [Fintype P]

/-- Sum arbitrary position-dependent transformed embeddings of a finite tuple. -/
def mix (operators : P → (Channels K → ℝ) → (Channels K → ℝ))
    (E : V → Channels K → ℝ) (tokens : P → V) (i : Channels K) : ℝ :=
  ∑ p, operators p (E (tokens p)) i

/-- G3: if a positional mixer repeats under a permutation, pooled inputs
cannot distinguish that permutation of the operand tuple. -/
theorem mix_permutation (operators : P → (Channels K → ℝ) → (Channels K → ℝ))
    (E : V → Channels K → ℝ) (tokens : P → V) (e : P ≃ P)
    (he : ∀ p, operators (e p) = operators p) :
    mix operators E (tokens ∘ e) = mix operators E tokens := by
  funext i
  change (∑ p, operators p (E (tokens (e p))) i) = _
  calc (∑ p, operators p (E (tokens (e p))) i) =
      ∑ p, operators (e p) (E (tokens (e p))) i := by
        apply Finset.sum_congr rfl
        intro p hp
        rw [he p]
    _ = mix operators E tokens i := Equiv.sum_comp e (fun p => operators p (E (tokens p)) i)

omit [Fintype P] in
/-- G2: the existing two-input raw embedding is the zero/quarter-turn instance. -/
theorem mix_two (E : V → Channels K → ℝ) (a b : V) :
    mix (fun p : Fin 2 => rotate (fun _ : K => if p = 0 then 0 else Real.pi / 2))
      E ![a, b] = tiedInput E a b := by
  funext i
  simp only [mix, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
    ↓reduceIte, Fin.isValue, one_ne_zero]
  rw [rotate_zero, rotate_quarter]
  rfl

variable [Fintype V] [Fintype K] [Fintype H] [Fintype D]

/-- The full tied Peri-RMSNorm SwiGLU stack after a general positional mixer. -/
def pooledScore (operators : P → (Channels K → ℝ) → (Channels K → ℝ))
    (eps : ℝ) (order : List D) (w : Learning.Weights V K H D) (tokens : P → V) (v : V) : ℝ :=
  let E := Learning.embedding w
  let x := (order.map (Learning.matrices w)).foldl (fun z B => Power.block eps B z)
    (rmsNorm eps (mix operators E tokens))
  tiedReadout E (rmsNorm eps x) v

omit [Fintype P] [Fintype V] [Fintype D] in
/-- G2: the general multivariable forward recovers all current two-input
scores, including every normalizer and the shared embedding/readout. -/
theorem pooled_two_eq_current (eps : ℝ) (order : List D)
    (w : Learning.Weights V K H D) (a b q : V) :
    pooledScore (fun p : Fin 2 => rotate (fun _ : K => if p = 0 then 0 else Real.pi / 2))
      eps order w ![a, b] q = Learning.score eps order w a b q := by
  dsimp only [pooledScore, Learning.score, Power.stackLogit, Power.stackState]
  rw [mix_two]

omit [Fintype V] [Fintype D] in
/-- G3: an input collision survives every downstream depth and every choice
of trainable matrices. No optimizer can recover discarded distinctions. -/
theorem pooled_collision (operators : P → (Channels K → ℝ) → (Channels K → ℝ))
    (eps : ℝ) (order : List D) (w : Learning.Weights V K H D) (a b : P → V)
    (h : mix operators (Learning.embedding w) a = mix operators (Learning.embedding w) b)
    (v : V) : pooledScore operators eps order w a v = pooledScore operators eps order w b v := by
  dsimp only [pooledScore]
  rw [h]

/-- The five consecutive fixed quarter-turn phases, for the special-case counterexample. -/
def quarterMixer (p : Fin 5) (x : Channels K → ℝ) : Channels K → ℝ :=
  rotate (fun _ => (![0, Real.pi / 2, Real.pi, 3 * Real.pi / 2, 2 * Real.pi] p)) x

omit [Fintype P] [Fintype V] [Fintype K] [Fintype H] [Fintype D] in
/-- G3: the first and fifth fixed-quarter-turn positions are indistinguishable
after pooling, for every embedding and arbitrary channel width. -/
theorem quarter_five_collision (E : V → Channels K → ℝ) (a b c d e : V) :
    mix quarterMixer E ![a, b, c, d, e] = mix quarterMixer E ![e, b, c, d, a] := by
  funext ⟨k, j⟩
  fin_cases j <;> simp [mix, quarterMixer, rotate, Fin.sum_univ_succ] <;> ring

omit [Fintype P] [Fintype V] [Fintype D] in
/-- G3: no width/depth/weights let this special pooled architecture strictly
predict the first of five binary variables on every input tuple. -/
theorem quarter_five_not_first (eps : ℝ) (order : List D)
    (w : Learning.Weights (Fin 2) K H D) :
    ¬ ∀ x : Fin 5 → Fin 2, Correct (pooledScore quarterMixer eps order) w (fun t => t 0) x := by
  intro hall
  have hs := pooled_collision (P := Fin 5) quarterMixer eps order w
    ![0, 0, 0, 0, 1] ![1, 0, 0, 0, 0]
    (quarter_five_collision (Learning.embedding w) 0 0 0 0 1)
  exact collision_not_correct (pooledScore quarterMixer eps order) w (fun t => t 0)
    ![0, 0, 0, 0, 1] ![1, 0, 0, 0, 0] hs (by decide) ⟨hall _, hall _⟩

/-- G2: arbitrary fixed rotary phases preserve analyticity of the entire
normalized SwiGLU stack jointly in every trainable matrix and tied embedding. -/
theorem analyticAt_rotary_pooled {eps : ℝ} (heps : 0 < eps)
    (phase : P → K → ℝ) (order : List D) (x : Learning.Weights V K H D)
    (tokens : P → V) (q : V) :
    AnalyticAt ℝ (fun w => pooledScore (fun p => rotate (phase p)) eps order w tokens q) x := by
  have hE : ∀ v i, AnalyticAt ℝ (fun w : Learning.Weights V K H D => Learning.embedding w v i) x :=
    fun v i => by simpa only [Learning.embedding] using
      (analyticAt_parameter (I := Learning.Index V K H D) (.inl (v, i)) x)
  have hm : ∀ i, AnalyticAt ℝ (fun w : Learning.Weights V K H D =>
      mix (fun p => rotate (phase p)) (Learning.embedding w) tokens i) x := by
    intro i
    apply Finset.analyticAt_fun_sum
    intro p hp
    dsimp only [rotate]
    split_ifs
    · exact ((hE _ _).fun_mul analyticAt_const).fun_sub ((hE _ _).fun_mul analyticAt_const)
    · exact ((hE _ _).fun_mul analyticAt_const).fun_add ((hE _ _).fun_mul analyticAt_const)
  have hin := fun i => analyticAt_rmsNorm heps hm i
  exact analyticAt_tiedReadout hE
    (fun i => analyticAt_rmsNorm heps (fun j => Learning.analyticAt_stack heps order hin j) i) q

example := mix_permutation (P := Fin 5) (K := Fin 2)
  (fun _ x => x) (fun (v : Fin 2) (i : Channels (Fin 2)) => (v.val + i.1.val + i.2.val : ℝ))
  (fun i => (if i = 0 then 0 else 1 : Fin 2)) (Equiv.swap 0 4) (fun _ => rfl)

example := pooled_collision (P := Fin 5) (K := Fin 2) (H := Fin 4)
  quarterMixer (1 / 8388608) (List.finRange 2) (WithLp.toLp 2 (fun _ => 1))
  ![(0 : Fin 2), 0, 0, 0, 1] ![1, 0, 0, 0, 0]
  (quarter_five_collision _ 0 0 0 0 1) 1

example := analyticAt_rotary_pooled (P := Fin 7) (V := Fin 3) (K := Fin 2) (H := Fin 4)
  (eps := 1 / 8388608) (by norm_num) (fun p k => ((p.val + 1) * (k.val + 1) : ℝ) / 7)
  (List.finRange 2) (WithLp.toLp 2 (fun _ => 1)) (fun _ => 1) 2

end Transformer.SequenceModel

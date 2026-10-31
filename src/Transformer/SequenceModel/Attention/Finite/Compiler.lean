module

public import Transformer.SequenceModel.Attention.Finite.Features

/-!
# Compile the finite interpolation into the actual Transformer FFN

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F3.
The matrices have exactly the source hidden width, with unused channels
zero-padded. Both feed-forward normalizers and the residual remain in place.
The hypotheses describe input geometry and branch reachability, not desired
logits or the existence of a trained model.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Expressivity Landscape.Neural
open scoped BigOperators

variable {R Z : Type*} [Fintype R]

/-- Fixed gate rows, placed in the actual source hidden layout. -/
def ridgeMap (cfg : Config) (e : Fin cfg.d_ff ≃ R ⊕ Z)
    (g : R → EucSpace cfg.d_model →L[ℝ] ℝ) :
    EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_ff :=
  LinearMap.toContinuousLinearMap {
    toFun := fun x => WithLp.toLp 2 (fun h => match e h with | .inl r => g r x | .inr _ => 0)
    map_add' := by intro x y; ext h; cases he : e h <;> simp [he]
    map_smul' := by intro c x; ext h; cases he : e h <;> simp [he] }

/-- Output matrix of the fixed vector-valued interpolation coefficients. -/
def ridgeOut (cfg : Config) (e : Fin cfg.d_ff ≃ R ⊕ Z) (out : R → EucSpace cfg.d_model) :
    EucSpace cfg.d_ff →L[ℝ] EucSpace cfg.d_model :=
  LinearMap.toContinuousLinearMap {
    toFun := fun x => ∑ r, x (e.symm (.inl r)) • out r
    map_add' := by intro x y; simp only [PiLp.add_apply, add_smul, Finset.sum_add_distrib]
    map_smul' := by intro c x; simp only [PiLp.smul_apply, smul_eq_mul, mul_smul,
      RingHom.id_apply, Finset.smul_sum] }

/-- Install fixed actual SwiGLU matrices while retaining the attention matrices. -/
def withRidges (cfg : Config) (B : BlockParams cfg) (e : Fin cfg.d_ff ≃ R ⊕ Z)
    (g u : R → EucSpace cfg.d_model →L[ℝ] ℝ) (out : R → EucSpace cfg.d_model) : BlockParams cfg :=
  { B with gate := ridgeMap cfg e g, value := ridgeMap cfg e u, output := ridgeOut cfg e out }

/-- F3: the compiled source FFN equals the stated finite genuine-gate sum. -/
theorem feedForward_withRidges (cfg : Config) (B : BlockParams cfg) (e : Fin cfg.d_ff ≃ R ⊕ Z)
    (g u : R → EucSpace cfg.d_model →L[ℝ] ℝ) (out : R → EucSpace cfg.d_model)
    (x : EucSpace cfg.d_model) :
    feedForward cfg (withRidges cfg B e g u out) x = ∑ r, swiglu (g r x) (u r x) • out r := by
  change (∑ r, swiglu _ _ • out r) = _
  apply Finset.sum_congr rfl
  intro r _
  change swiglu
    (match e (e.symm (.inl r)) with | .inl q => g q x | .inr _ => 0)
    (match e (e.symm (.inl r)) with | .inl q => u q x | .inr _ => 0) • out r = _
  rw [Equiv.apply_symm_apply]

/-- F3: positive-epsilon RMSNorm preserves the common-channel projective
coordinates exactly, including its nonconstant row-dependent scale. -/
theorem rms_projective (d : ℕ) (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (x : EucSpace d) (u : EucSpace d →L[ℝ] ℝ) (hu : 0 < u x) :
    0 < u (rmsNormEps eps x) ∧
      (u (rmsNormEps eps x))⁻¹ • rmsNormEps eps x = (u x)⁻¹ • x := by
  have hs := normScale_pos d hd eps heps (‖x‖ ^ 2) (sq_nonneg _)
  rw [rms_of_normSq d eps x (‖x‖ ^ 2) rfl, map_smul, smul_eq_mul]
  refine ⟨mul_pos hs hu, ?_⟩
  rw [smul_smul]
  congr 1
  field_simp

variable {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]

/-- Actual hidden layout with exactly three channels per input row and padding. -/
def interpolationLayout (cfg : Config) (h : 3 * Fintype.card A ≤ cfg.d_ff) :
    Fin cfg.d_ff ≃ (A × Fin 3) ⊕ Fin (cfg.d_ff - 3 * Fintype.card A) :=
  Fintype.equivOfCardEq (by
    simp only [Fintype.card_fin, Fintype.card_sum, Fintype.card_prod]
    omega)

/-- F3: any projectively distinct finite table within the genuine branch
radius can be mapped exactly to requested residual states by one actual FFN.
The attention matrices are retained; arbitrary source padding is allowed. -/
theorem exists_normalized_ffn (cfg : Config) (hwidth : 3 * Fintype.card A ≤ cfg.d_ff)
    (B : BlockParams cfg) {eps : ℝ} (heps : 0 < eps)
    (x target : A → EucSpace cfg.d_model) (u : EucSpace cfg.d_model →L[ℝ] ℝ)
    (hu : ∀ a, 0 < u (x a)) (hx : Function.Injective (fun a => (u (x a))⁻¹ • x a))
    (hreach : ∀ a, ‖target a - x a‖ ^ 2 < (cfg.d_model : ℝ)) :
    ∃ C : BlockParams cfg, C.qkv = B.qkv ∧ C.projection = B.projection ∧
      ∀ a, x a + rmsNormEps eps (feedForward cfg C (rmsNormEps eps (x a))) = target a := by
  let xn := fun a => rmsNormEps eps (x a)
  have hp := fun a => rms_projective cfg.d_model cfg.d_model_pos heps (x a) u (hu a)
  have hi : Function.Injective (fun a => (u (xn a))⁻¹ • xn a) := by
    intro a b hab
    apply hx
    simpa only [xn, (hp a).2, (hp b).2] using hab
  obtain ⟨g, out, ht⟩ := exists_swiglu_interpolation xn u (fun a => (hp a).1) hi
    (fun a => unnormalize cfg.d_model eps (target a - x a))
  let C := withRidges cfg B (interpolationLayout cfg hwidth) g (fun _ => u) out
  refine ⟨C, rfl, rfl, ?_⟩
  intro a
  rw [show feedForward cfg C (xn a) = _ from feedForward_withRidges cfg B
    (interpolationLayout cfg hwidth) g (fun _ => u) out (xn a), ht]
  exact residual_target cfg.d_model cfg.d_model_pos eps heps (x a) (target a) (hreach a)

example := rms_projective 2 (by decide) (eps := 1 / 8388608) (by norm_num)
  (WithLp.toLp 2 ![2, 1]) (innerSL ℝ (WithLp.toLp 2 ![0, 1]))
  (by norm_num [PiLp.inner_apply, Fin.sum_univ_two])

end Transformer.SequenceModel.Attention.Finite

module

public import Transformer.SequenceModel.Attention.Expressivity.Normalization
public import Transformer.Landscape.Neural.Products

/-!
# Exact quadratic compilation into the actual attention block's SwiGLU

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E2.
Each product uses two opposed genuine SiLU gates. The matrices are fixed
linear maps, with arbitrary zero-padded hidden channels. No activation is
replaced by multiplication in the model definition.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini Landscape.Neural
open scoped BigOperators

variable {R Z : Type*} [Fintype R] [Fintype Z]

/-- Fixed linear gate/value maps with the source's finite hidden layout. -/
def pairedMap (cfg : Config) (e : Fin cfg.d_ff ≃ (R × Fin 2) ⊕ Z)
    (g : R → EucSpace cfg.d_model →L[ℝ] ℝ) (opposed : Bool) :
    EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_ff :=
  LinearMap.toContinuousLinearMap {
    toFun := fun x => WithLp.toLp 2 fun h => match e h with
      | .inl (r, side) => (if opposed && side == 1 then -1 else 1) * g r x
      | .inr _ => 0
    map_add' := by
      intro x y
      ext h
      cases he : e h with
      | inl r => simp [he, mul_add]
      | inr z => simp [he]
    map_smul' := by
      intro c x
      ext h
      cases he : e h with
      | inl r => simp [he]
      | inr z => simp [he] }

/-- The output subtracts the opposed gate and adds all requested directions. -/
def pairedOut (cfg : Config) (e : Fin cfg.d_ff ≃ (R × Fin 2) ⊕ Z)
    (d : R → EucSpace cfg.d_model) : EucSpace cfg.d_ff →L[ℝ] EucSpace cfg.d_model :=
  LinearMap.toContinuousLinearMap {
    toFun := fun x => ∑ r, (x (e.symm (.inl (r, 0))) - x (e.symm (.inl (r, 1)))) • d r
    map_add' := by intro x y; simp only [PiLp.add_apply]; simp_rw [add_sub_add_comm,
                    add_smul]; exact Finset.sum_add_distrib
    map_smul' := by
      intro c x
      simp only [PiLp.smul_apply, smul_eq_mul, RingHom.id_apply]
      simp_rw [← mul_sub, mul_smul]
      exact Finset.smul_sum.symm }

/-- Replace only the three feed-forward matrices, keeping actual attention. -/
def withProducts (cfg : Config) (B : BlockParams cfg)
    (e : Fin cfg.d_ff ≃ (R × Fin 2) ⊕ Z)
    (g u : R → EucSpace cfg.d_model →L[ℝ] ℝ) (d : R → EucSpace cfg.d_model) :
    BlockParams cfg :=
  { B with
    gate := pairedMap cfg e g true
    value := pairedMap cfg e u false
    output := pairedOut cfg e d }

omit [Fintype Z] in
/-- E2: the actual finite SwiGLU forward computes every requested quadratic
form exactly, for every input vector, including all normalization scales. -/
theorem feedForward_withProducts (cfg : Config) (B : BlockParams cfg)
    (e : Fin cfg.d_ff ≃ (R × Fin 2) ⊕ Z)
    (g u : R → EucSpace cfg.d_model →L[ℝ] ℝ) (d : R → EucSpace cfg.d_model)
    (x : EucSpace cfg.d_model) :
    feedForward cfg (withProducts cfg B e g u d) x = ∑ r, (g r x * u r x) • d r := by
  change (∑ r, (swiglu _ _ - swiglu _ _) • d r) = _
  apply Finset.sum_congr rfl
  intro r hr
  change (swiglu
      (match e (e.symm (.inl (r, 0))) with
        | .inl (q, side) => (if true && side == 1 then -1 else 1) * g q x | .inr _ => 0)
      (match e (e.symm (.inl (r, 0))) with
        | .inl (q, side) => (if false && side == 1 then -1 else 1) * u q x | .inr _ => 0) -
    swiglu
      (match e (e.symm (.inl (r, 1))) with
        | .inl (q, side) => (if true && side == 1 then -1 else 1) * g q x | .inr _ => 0)
      (match e (e.symm (.inl (r, 1))) with
        | .inl (q, side) => (if false && side == 1 then -1 else 1) * u q x | .inr _ => 0)) • d r = _
  simp only [Equiv.apply_symm_apply]
  norm_num only [Bool.true_and, Bool.false_and, beq_iff_eq, Fin.zero_ne_one, ↓reduceIte,
    Bool.false_eq_true, one_mul, neg_one_mul]
  rw [swiglu_sub_neg_gate]

end Transformer.SequenceModel.Attention.Expressivity

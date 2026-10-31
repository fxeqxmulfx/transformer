module

public import Transformer.SequenceModel.Attention.Finite.Separation
public import Transformer.SequenceModel.Attention.Finite.Evaluation

/-!
# Exact finite interpolation with bias-free SwiGLU

Source: `review/flat_spectrum.md`, *Finite sequence capacity and information limits*, F2.
Three hidden channels per input row suffice whenever the input directions
are distinct and a shared linear channel is positive. All gate, value and
output maps are fixed before an input is queried. RMSNorm may rescale each
row differently; projective separation accounts for those scales exactly.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Finite

open GPTMini Landscape.Neural
open scoped BigOperators RealInnerProductSpace

variable {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A] {d m : ℕ}

/-- The three linear gates centered at one separated projective coordinate. -/
def tripleGates (l u : EucSpace d →L[ℝ] ℝ) (α : A → ℝ) (δ gain : ℝ)
    (r : A × Fin 3) : EucSpace d →L[ℝ] ℝ :=
  ![gain • (l - (α r.1 - δ) • u), gain • (l - α r.1 • u),
    gain • (l - (α r.1 + δ) • u)] r.2

/-- The three signed output coefficients share one interpolating direction. -/
def tripleOut (gain : ℝ) (c : A → EucSpace m) (r : A × Fin 3) : EucSpace m :=
  (![1, -2, 1] r.2 / gain) • c r.1

/-- F2: fixed actual gates interpolate any vector-valued table when the
ratios of two linear measurements separate its input rows. -/
theorem exists_interpolation_of_ratios (x : A → EucSpace d)
    (l u : EucSpace d →L[ℝ] ℝ) (hu : ∀ a, 0 < u (x a))
    (hα : Function.Injective (fun a => l (x a) / u (x a))) (target : A → EucSpace m) :
    ∃ g : A × Fin 3 → EucSpace d →L[ℝ] ℝ,
      ∃ out : A × Fin 3 → EucSpace m,
        ∀ a, (∑ r, swiglu (g r (x a)) (u (x a)) • out r) = target a := by
  let α := fun a => l (x a) / u (x a)
  let v := fun a => u (x a)
  obtain ⟨δ, hδ, gain, hg, hm⟩ := exists_invertible_evaluation α v hα hu
  let M := evaluation α v δ gain
  let c := interpolationCoefficients M target
  refine ⟨tripleGates l u α δ gain, tripleOut gain c, ?_⟩
  intro a
  rw [Fintype.sum_prod_type]
  calc
    _ = ∑ b, M a b • c b := by
      apply Finset.sum_congr rfl
      intro b _
      simp only [Fin.sum_univ_three, tripleGates, tripleOut,
        Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
        Matrix.head_cons, Matrix.tail_cons, smul_apply, sub_apply, smul_eq_mul, smul_smul]
      rw [← add_smul, ← add_smul]
      congr 1
      have ht := triangle_swiglu gain δ (l (x a)) (u (x a)) (α b) hg.ne' (hu a).ne'
      change _ = (u (x a)) ^ 2 * softTriangle (gain * u (x a)) δ (l (x a) / u (x a) - α b)
      convert ht using 1; ring
    _ = target a := interpolate_matrix M hm target a

/-- F2: arbitrary finite projectively distinct inputs with a positive
common channel admit exact interpolation by genuine, finite SwiGLU weights. -/
theorem exists_swiglu_interpolation (x : A → EucSpace d) (u : EucSpace d →L[ℝ] ℝ)
    (hu : ∀ a, 0 < u (x a))
    (hx : Function.Injective (fun a => (u (x a))⁻¹ • x a)) (target : A → EucSpace m) :
    ∃ g : A × Fin 3 → EucSpace d →L[ℝ] ℝ,
      ∃ out : A × Fin 3 → EucSpace m,
        ∀ a, (∑ r, swiglu (g r (x a)) (u (x a)) • out r) = target a := by
  obtain ⟨w, _, hw⟩ := exists_separating_inner (fun a => (u (x a))⁻¹ • x a) hx
    0 (r := 1) (by norm_num)
  let l := innerSL ℝ w
  have hα : Function.Injective (fun a => l (x a) / u (x a)) := by
    have he : (fun a => ⟪w, (u (x a))⁻¹ • x a⟫) = (fun a => l (x a) / u (x a)) := by
      funext a
      change ⟪w, (u (x a))⁻¹ • x a⟫ = ⟪w, x a⟫ / u (x a)
      rw [real_inner_smul_right, div_eq_mul_inv, mul_comm]
    rw [he] at hw
    exact hw
  exact exists_interpolation_of_ratios x l u hu hα target

private def rows (a : Fin 3) : EucSpace 2 := WithLp.toLp 2 ![(a.val : ℝ), 1]
private def common : EucSpace 2 →L[ℝ] ℝ := innerSL ℝ (WithLp.toLp 2 ![0, 1])
private def measure : EucSpace 2 →L[ℝ] ℝ := innerSL ℝ (WithLp.toLp 2 ![1, 0])
private theorem common_rows (a : Fin 3) : common (rows a) = 1 := by
  norm_num [common, rows, PiLp.inner_apply, Fin.sum_univ_two]
private theorem measure_rows (a : Fin 3) : measure (rows a) = a.val := by
  norm_num [measure, rows, PiLp.inner_apply, Fin.sum_univ_two]

example := exists_interpolation_of_ratios rows measure common
  (by intro a; rw [common_rows]; norm_num)
  (by
    intro a b h
    simp only [common_rows, measure_rows, div_one] at h
    apply Fin.ext
    exact_mod_cast h)
  (fun a => WithLp.toLp 2 ![(a.val : ℝ) ^ 3, 2 - a.val])
example := exists_swiglu_interpolation rows common
  (by intro a; rw [common_rows]; norm_num)
  (by
    intro a b h
    simp only [common_rows, inv_one, one_smul] at h
    have hc : (a.val : ℝ) = b.val := congrArg (fun x : EucSpace 2 => x 0) h
    apply Fin.ext
    exact_mod_cast hc)
  (fun a => WithLp.toLp 2 ![(a.val : ℝ) ^ 3, 2 - a.val])

end Transformer.SequenceModel.Attention.Finite

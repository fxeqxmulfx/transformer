module

public import Transformer.Landscape.Neural.SwiGLU

/-!
# Exact products from paired SwiGLU gates

Source: M9.4b of `landscape_grokking_plan.md`; the explicit division witness
in `experiments/rotate90_analytic/expressivity.py`. The actual gated model is
defined in `Neural.SwiGLU`, from the browser architecture with its branch
replaced by `(SiLU(r G) * (r U)) D`.

Opposite gates and opposite output coefficients cancel SiLU's nonlinear
even part, realizing each product exactly in real arithmetic. The compiler
below places all entries in the actual Euclidean GatedWeights parameter
space. The resulting state retains the skip and the first RMSNorm. Its
final RMSNorm and tied readout remain those of gatedLogit. This is a
constructive representation result, not an invariant of gradient training.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Landscape.Neural

/-- SiLU's exact odd part (M9.4b, paired-gate product construction). -/
theorem silu_sub_neg (t : ℝ) : silu t - silu (-t) = t := by
  have he : Real.exp t ≠ 0 := Real.exp_ne_zero t
  have hp : 1 + Real.exp t ≠ 0 := by positivity
  simp only [silu, neg_neg, Real.exp_neg]
  field_simp
  ring

/-- Two opposed SwiGLU gates give a product, without a small-input limit
or an approximation error (M9.4b). -/
theorem swiglu_sub_neg_gate (gate value : ℝ) :
    swiglu gate value - swiglu (-gate) value = gate * value := by
  rw [swiglu, swiglu, ← sub_mul, silu_sub_neg]

variable {V K T : Type*} [Fintype K] [Fintype T]

/-- Opposed gate projections, two per bilinear term. -/
def pairedGate (g : Channels K → T → ℝ) (i : Channels K) (j : T × Fin 2) : ℝ :=
  if j.2 = 0 then g i j.1 else -g i j.1

/-- Both members of a pair use the same value projection. -/
def pairedValue (u : Channels K → T → ℝ) (i : Channels K) (j : T × Fin 2) : ℝ :=
  u i j.1

/-- Opposed output coefficients subtract the two gates after activation. -/
def pairedOutput (d : T → Channels K → ℝ) (j : T × Fin 2) (i : Channels K) : ℝ :=
  if j.2 = 0 then d j.1 i else -d j.1 i

/-- A concrete parameter vector of the full tied model; the embedding and
all three branch matrices occupy their original free coordinates. -/
noncomputable def productWeights (E : V → Channels K → ℝ)
    (g u : Channels K → T → ℝ) (d : T → Channels K → ℝ) :
    GatedWeights V K (T × Fin 2) :=
  WithLp.toLp 2 fun coordinate => match coordinate with
    | .inl (q, i) => E q i
    | .inr (.inl (i, j)) => pairedGate g i j
    | .inr (.inr (.inl (i, j))) => pairedValue u i j
    | .inr (.inr (.inr (j, i))) => pairedOutput d j i

/-- A paired branch implements a sum of products of linear forms in its
input, with exactly two gates per term (M9.4b, nine terms use 18 gates). -/
theorem pairedBranch_eq_products (g u : Channels K → T → ℝ)
    (d : T → Channels K → ℝ) (r : Channels K → ℝ) (i : Channels K) :
    dense (pairedOutput d)
      (fun j => swiglu (dense (pairedGate g) r j) (dense (pairedValue u) r j)) i =
        ∑ t, dense g r t * dense u r t * d t i := by
  unfold dense
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro t ht
  rw [Fin.sum_univ_two]
  simp only [pairedGate, pairedValue, pairedOutput, Fin.isValue,
    ↓reduceIte, Fin.reduceEq, mul_neg, Finset.sum_neg_distrib]
  calc
    _ = (swiglu (∑ k, r k * g k t) (∑ k, r k * u k t) -
          swiglu (-(∑ k, r k * g k t)) (∑ k, r k * u k t)) * d t i := by ring
    _ = _ := by rw [swiglu_sub_neg_gate]

/-- The compiled weights inhabit the actual residual model, including its
shared E and positive-epsilon normalization formula. No normalization is
dropped to obtain the quadratic branch identity (M9.4b). -/
theorem gatedState_productWeights (eps : ℝ) (E : V → Channels K → ℝ)
    (g u : Channels K → T → ℝ) (d : T → Channels K → ℝ)
    (a b : V) (i : Channels K) :
    gatedState eps (productWeights E g u d) a b i =
      tiedInput E a b i + ∑ t,
        dense g (rmsNorm eps (tiedInput E a b)) t *
          dense u (rmsNorm eps (tiedInput E a b)) t * d t i := by
  change tiedInput E a b i + dense (pairedOutput d)
    (fun j => swiglu (dense (pairedGate g) (rmsNorm eps (tiedInput E a b)) j)
      (dense (pairedValue u) (rmsNorm eps (tiedInput E a b)) j)) i = _
  rw [pairedBranch_eq_products]

/-- The same compiled parameters retain the second normalization and the
transposed shared embedding in every logit. The character witness uses this
full formula, rather than a separate clock readout (M9.4b). -/
theorem gatedLogit_productWeights (eps : ℝ) (E : V → Channels K → ℝ)
    (g u : Channels K → T → ℝ) (d : T → Channels K → ℝ) (a b q : V) :
    gatedLogit eps (productWeights E g u d) a b q =
      tiedReadout E (rmsNorm eps (fun i => tiedInput E a b i + ∑ t,
        dense g (rmsNorm eps (tiedInput E a b)) t *
          dense u (rmsNorm eps (tiedInput E a b)) t * d t i)) q := by
  have h : gatedState eps (productWeights E g u d) a b =
      fun i => tiedInput E a b i + ∑ t,
        dense g (rmsNorm eps (tiedInput E a b)) t *
          dense u (rmsNorm eps (tiedInput E a b)) t * d t i := by
    funext i
    exact gatedState_productWeights eps E g u d a b i
  change tiedReadout E (rmsNorm eps (gatedState eps (productWeights E g u d) a b)) q = _
  rw [h]

-- Nonconstant gates, values and coefficients witness the exact product.
example : swiglu (2 : ℝ) 3 - swiglu (-2) 3 = 6 := by
  rw [swiglu_sub_neg_gate]
  norm_num

example (r : Channels (Fin 1) → ℝ) :
    dense (pairedOutput (fun (_ : Fin 1) (i : Channels (Fin 1)) => if i.2 = 0 then 1 else -1))
      (fun j => swiglu
        (dense (pairedGate (fun i (_ : Fin 1) => if i.2 = 0 then 1 else 0)) r j)
        (dense (pairedValue (fun i (_ : Fin 1) => if i.2 = 1 then 1 else 0)) r j)) (0, 0) =
      ∑ t : Fin 1,
        dense (fun (i : Channels (Fin 1)) (_ : Fin 1) => if i.2 = 0 then 1 else 0) r t *
          dense (fun (i : Channels (Fin 1)) (_ : Fin 1) => if i.2 = 1 then 1 else 0) r t *
            (if (0 : Fin 2) = 0 then 1 else -1) := by
  exact pairedBranch_eq_products _ _ _ r (0, 0)

example : gatedState (1 / 100000)
    (productWeights (fun (q : Fin 3) (i : Channels (Fin 1)) => (q.val : ℝ) + i.2.val)
      (fun i (t : Fin 1) => (i.2.val : ℝ) + t.val + 1)
      (fun i (t : Fin 1) => (i.2.val : ℝ) - t.val + 1)
      (fun (t : Fin 1) i => (t.val : ℝ) + i.2.val + 1)) 0 1 (0, 0) =
    tiedInput (fun (q : Fin 3) (i : Channels (Fin 1)) => (q.val : ℝ) + i.2.val) 0 1 (0, 0) +
      ∑ t : Fin 1,
        dense (fun i (s : Fin 1) => (i.2.val : ℝ) + s.val + 1)
          (rmsNorm (1 / 100000)
            (tiedInput (fun (q : Fin 3) (i : Channels (Fin 1)) => (q.val : ℝ) + i.2.val) 0 1)) t *
        dense (fun i (s : Fin 1) => (i.2.val : ℝ) - s.val + 1)
          (rmsNorm (1 / 100000)
            (tiedInput (fun (q : Fin 3) (i : Channels (Fin 1)) => (q.val : ℝ) + i.2.val) 0 1)) t *
        ((t.val : ℝ) + (0 : Fin 2).val + 1) := by
  exact gatedState_productWeights _ _ _ _ _ _ _ _

end Transformer.Landscape.Neural

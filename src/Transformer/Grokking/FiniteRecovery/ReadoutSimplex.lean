module

public import Transformer.Grokking.FiniteRecovery.ArgandReachability
public import Transformer.Grokking.FiniteRecovery.ZeroEmbeddings

/-! A target row inside a full-dimensional simplex of competing output
rows can never be a strict winner. A quantitative version uses only the
coordinate errors in the tied embeddings, so it holds for every hidden
state of the full decoder. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Finset
open scoped InnerProductSpace

variable {I A : Type*} [Fintype I]

private theorem row_error (u v h : I → ℝ) (δ : ℝ) (H : ∀ i, |u i-v i| ≤ δ) :
    |(∑ i, u i*h i) - ∑ i, v i*h i| ≤ δ*∑ i, |h i| := by
  rw [← sum_sub_distrib,Finset.mul_sum]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (sum_le_sum fun i _ => ?_)
  rw [← sub_mul,abs_mul]
  exact mul_le_mul_of_nonneg_right (H _) (abs_nonneg _)

variable [DecidableEq I]

/-- The ideal rival rows are the scaled standard basis and the negative
all-ones row. If every row moves at most δ per coordinate, no feature vector
can make the perturbed origin strictly beat every rival. -/
theorem simplex_prevents_winner (row : A → I → ℝ) (target neg : A) (basis : I → A)
    (hne : neg ≠ target) (hb : ∀ i, basis i ≠ target) (δ K : ℝ) (hd : 0 ≤ δ)
    (large : (2*(Fintype.card I : ℝ)+1)*(2*δ) ≤ K)
    (atTarget : ∀ i, |row target i| ≤ δ) (atNeg : ∀ i, |row neg i-(-K)| ≤ δ)
    (atBasis : ∀ j i, |row (basis j) i - (if i=j then K else 0)| ≤ δ) (h : I → ℝ) :
    ¬ ∀ b, b ≠ target → (∑ i, row b i*h i) < ∑ i, row target i*h i := by
  intro win
  let S := ∑ i, |h i|
  have hS : 0 ≤ S := sum_nonneg fun i _ => abs_nonneg (h i)
  have hK : 0 ≤ K := (by positivity : 0 ≤ (2*(Fintype.card I : ℝ)+1)*(2*δ)).trans large
  have targetBound : (∑ i, row target i*h i) ≤ δ*S := by
    have he := row_error (row target) 0 h δ (fun i => by simpa using atTarget i)
    simpa [S] using (le_abs_self _).trans he
  have bound (u : A) (ideal : I → ℝ) (hu : u ≠ target) (close : ∀ i, |row u i-ideal i| ≤ δ) :
      (∑ i, ideal i*h i) < 2*δ*S := by
    have e := (abs_le.mp (row_error (row u) ideal h δ close)).1
    change -(δ*S) ≤ _ at e
    linarith [win u hu]
  have each (j : I) : max (K*h j) 0 ≤ 2*δ*S := by
    apply max_le _ (by positivity)
    have he := bound (basis j) _ (hb j) (atBasis j)
    simpa [ite_mul] using he.le
  have negRow := bound neg (fun _ => -K) hne atNeg
  rw [← Finset.mul_sum,neg_mul] at negRow
  have posSum : (∑ i, max (K*h i) 0) ≤ (Fintype.card I : ℝ)*(2*δ*S) := by
    simpa using sum_le_sum (s := (univ : Finset I)) (fun i _ => each i)
  have split : K*S = 2*(∑ i, max (K*h i) 0) - K*∑ i, h i := by
    simp only [S,Finset.mul_sum,← sum_sub_distrib]
    apply sum_congr rfl
    intro i _
    rw [← abs_of_nonneg hK,← abs_mul,abs_of_nonneg hK]
    rcases le_total 0 (K*h i) with hp | hn
    · rw [max_eq_left hp,abs_of_nonneg hp]; ring
    · rw [max_eq_right hn,abs_of_nonpos hn]; ring
  have incompatible := mul_le_mul_of_nonneg_right large hS
  linarith only [split,negRow,posSum,incompatible]

abbrev ReadoutCoord (c : ℕ) := Fin c × Bool

def readoutRow {H d L : ℕ} (w : Weights A H d L) (a : A) (i : ReadoutCoord (H*d)) : ℝ :=
  w (.inl (.embedding a i.1,i.2))

def realFeature {c : ℕ} (h : Transformer.PhasorV2.CSpace c) (i : ReadoutCoord c) : ℝ :=
  if i.2 then (h i.1).im else (h i.1).re

/-- The native tied complex readout is half the real dot product across
the real and imaginary parts of its embedding row. -/
theorem readout_real_dot {H d L : ℕ} (w : Weights A H d L) (h : Transformer.PhasorV2.CSpace (H*d)) (a : A) :
    (Model.fromWeights w).readout h a = (∑ i, readoutRow w a i * realFeature h i)/2 := by
  unfold Model.readout
  congr 1
  rw [PiLp.inner_apply,Complex.re_sum,Fintype.sum_prod_type]
  apply sum_congr rfl
  intro j _
  simp [realFeature,readoutRow,Model.fromWeights,complexEntry,RCLike.inner_apply,
    Complex.mul_re,add_comm,mul_comm]

/-- With zero input/output embeddings every actual minibatch gradient is
zero. Decoupled decay preserves these rows and both moment buffers. -/
theorem adamW_zero_embeddings [Fintype A] {H d L : ℕ} (r : ℕ) (batch : ℕ → List (LabeledPosition A))
    (rate : ℕ → ℝ) (draw : ComplexParameter A H d L → ℂ) (start : ZeroEmbeddings (sgdInitialWeights true draw))
    (t : ℕ) : let s := adamWRun r batch rate draw t
    ZeroEmbeddings s.weights ∧ s.first = 0 ∧ s.second = 0 := by
  unfold adamWRun
  induction t with
  | zero => exact ⟨start,rfl,rfl⟩
  | succ t ih =>
    obtain ⟨rows,first,second⟩ := ih
    have grad := batchGradient_eq_zero_of_zeroEmbeddings r (batch t) _ rows
    simp only [AdamW.run,grad,first,second,Transformer.AMSGradW.Kernite.clip,Pi.zero_apply,zero_div,
      mul_zero,add_zero,zero_pow (by decide : 2≠0),AdamW.direction,sub_zero]
    refine ⟨?_,rfl,rfl⟩
    intro a j b
    exact mul_eq_zero_of_right _ (rows a j b)

/-- The all-zero embedding draw is a point in Gaussian support. Its
unchanged tied logits prove failure from this fixed start for unlimited
ordinary updates on the real division minibatches, for any WSD rate. -/
theorem adamW_zero_draw_logits [Fintype A] {H d L T : ℕ} (r : ℕ)
    (batch : ℕ → List (LabeledPosition A)) (rate : ℕ → ℝ) (t : ℕ) (x : Word A T) (i : Fin T) (a : A) :
    (Model.fromWeights ((adamWRun (H := H) (d := d) (L := L) r batch rate (fun _ => 0) t).weights)).logits r x i a = 0 := by
  have hw := (adamW_zero_embeddings (H := H) (d := d) (L := L) r batch rate (fun _ => 0)
    (fun _ _ b => by cases b <;> rfl) t).1
  let w := (adamWRun (H := H) (d := d) (L := L) r batch rate (fun _ => 0) t).weights
  have he : (Model.fromWeights w).embedding a = 0 := by
    ext j; exact Complex.ext (hw a j false) (hw a j true)
  change (Model.fromWeights w).logits r x i a = 0
  simp [Model.logits,Model.readout,he]

end Transformer.Grokking.FiniteRecovery.Argand

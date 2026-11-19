module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRetrieval.Binding

/-!
# Two content-dependent reads and a learned complex phase interaction

Source: `review/convex/moments.md`, CM113. Each object indexes three learned
phase vectors. Cosine differences are the real inner products of unit complex
coordinates. Destination scores depend on observed content, its predecessor
and learned relative potentials. Every position and output class participates.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

open scoped BigOperators
open Perspective

/-- CM113. The coordinate width is `d + 1`. The boundary bias and start phases
are learned along with the object-indexed and positional parameters. -/
structure Parameters (O : Type*) (d : ℕ) where
  key : O → Fin (d + 1) → ℝ
  value : O → Fin (d + 1) → ℝ
  phase : O → Fin (d + 1) → ℝ
  start : Fin (d + 1) → ℝ
  bias : Option O → ℝ
  scale : ℝ
  recency : ℝ
  relation : ℤ → ℝ
  depth : ℝ
  product : ℝ

variable {O : Type*} {d n : ℕ}

/-- CM113. Gates range over the closed interval. All other coordinates are
unconstrained reals and the induced complex coordinates have modulus one. -/
def Admissible (p : Parameters O d) : Prop :=
  (0 ≤ p.depth ∧ p.depth ≤ 1) ∧ 0 ≤ p.product ∧ p.product ≤ 1

/-- The coordinate mean of executed complex inner products, CM113. -/
noncomputable def average (u : Fin (d + 1) → ℝ) : ℝ := (∑ f, u f) / (d + 1 : ℕ)

/-- Real inner product after rotating each value coordinate by a learned
angle. Zero angle gives the tied value/class kernel, CM113. -/
noncomputable def decode (p : Parameters O d) (v turn : Fin (d + 1) → ℝ) (c : O) : ℝ :=
  average (fun f => Real.cos (v f + turn f - p.value c f))

/-- The immediately preceding raw object in the causal prefix, CM113. -/
def previous (x : Fin n → O) (j : Fin n) : Option O :=
  if j.val = 0 then none else some (x ⟨j.val - 1, by have := j.isLt; omega⟩)

/-- Actual learned content match for a query object and a candidate source
position. Repeated objects may have different contextual scores, CM113. -/
noncomputable def keyScore (p : Parameters O d) (x : Fin n → O) (o : O) (j : Fin n) : ℝ :=
  p.scale * average (fun f => Real.cos (p.key o f - p.key (x j) f)) +
    p.bias (previous x j) + p.recency * j.val

/-- Unrestricted learned signed-offset score in each row, CM113. -/
def linkScore (p : Parameters O d) (j k : Fin n) : ℝ := p.relation ((k : ℤ) - (j : ℤ))

/-- From an object to a destination through every possible source position,
CM113. The intermediate variable is summed out. -/
noncomputable def jump (p : Parameters O d) (x : Fin n → O) (o : O) (k : Fin n) : ℝ :=
  ∑ j, softmaxWeight (keyScore p x o) j * softmaxWeight (linkScore p j) k

/-- A second normalized read uses the object reached by the first read as
its query. Both reads share the learned memory operator, CM113. -/
noncomputable def trail (p : Parameters O d) (x : Fin n → O) (q : O) (kl : Fin n × Fin n) : ℝ :=
  jump p x q kl.1 * jump p x (x kl.1) kl.2

/-- One learned read optionally rotates its retrieved value before the full
class readout, CM113. -/
noncomputable def readOne (p : Parameters O d) (x : Fin n → O) (q : O)
    (turn : Fin (d + 1) → ℝ) (c : O) : ℝ :=
  ∑ k, jump p x q k * decode p (p.value (x k)) turn c

/-- Complete two-hop output, including every intermediate and destination,
CM113. -/
noncomputable def readTwo (p : Parameters O d) (x : Fin n → O) (q c : O) : ℝ :=
  ∑ kl, trail p x q kl * decode p (p.value (x kl.2)) 0 c

/-- A learned gate interpolates complete bounded scores, CM113. -/
def blend (t a b : ℝ) : ℝ := (1 - t) * a + t * b

/-- All three routes execute from the same original prefix, CM113. -/
noncomputable def logit (p : Parameters O d) (x : Fin n → O) (q c : O) : ℝ :=
  blend p.product (blend p.depth (readOne p x q 0 c) (readTwo p x q c))
    (readOne p x q (p.phase q) c)

/-- Serialization of the entire original prefix, CM113. The learned boundary
represents an empty memory. A single object can already rotate this state. -/
noncomputable def score (p : Parameters O d) (xs : List O) (c : O) : ℝ :=
  match xs.reverse with
  | [] => decode p p.start 0 c
  | [q] => decode p p.start (p.phase q) c
  | q :: a :: rest =>
    let past := (a :: rest).reverse
    logit p (fun j : Fin past.length => past[j]) q c

/-- The full coordinate mean is bounded whenever each coordinate is bounded,
CM113. -/
theorem average_bound (u : Fin (d + 1) → ℝ) (h : ∀ f, |u f| ≤ 1) : |average u| ≤ 1 := by
  have hd : (0 : ℝ) < (d + 1 : ℕ) := by positivity
  rw [average, abs_div, abs_of_pos hd, div_le_one hd]
  calc
    |∑ f, u f| ≤ ∑ f, |u f| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _ : Fin (d + 1), (1 : ℝ) := Finset.sum_le_sum (fun f _ => h f)
    _ = _ := by simp

/-- Complete learned angular readouts have unit capacity for any rotation,
CM113. -/
theorem decode_bound (p : Parameters O d) (v turn : Fin (d + 1) → ℝ) (c : O) :
    |decode p v turn c| ≤ 1 := average_bound _ (fun _ => Real.abs_cos_le_one _)

/-- An object's score at its own unrotated value always attains one, CM113. -/
theorem self_decode (p : Parameters O d) (o : O) : decode p (p.value o) 0 o = 1 := by
  have hd : (d : ℝ) + 1 ≠ 0 := by positivity
  simp [decode, average, hd]

/-- Each normalized first-read weight is nonnegative, CM113. -/
theorem jump_nonneg (p : Parameters O d) (x : Fin n → O) (o : O) (k : Fin n) :
    0 ≤ jump p x o k :=
  Finset.sum_nonneg (fun j _ => mul_nonneg (softmaxWeight_nonneg _ j) (softmaxWeight_nonneg _ k))

/-- Summing both actual softmax rows preserves mass, CM113. -/
theorem sum_jump (hn : 0 < n) (p : Parameters O d) (x : Fin n → O) (o : O) :
    ∑ k, jump p x o k = 1 := by
  unfold jump
  rw [Finset.sum_comm]
  simp only [← Finset.mul_sum, sum_softmaxWeight hn, mul_one]

/-- Each two-read path has nonnegative mass, CM113. -/
theorem trail_nonneg (p : Parameters O d) (x : Fin n → O) (q : O) (kl : Fin n × Fin n) :
    0 ≤ trail p x q kl := mul_nonneg (jump_nonneg ..) (jump_nonneg ..)

/-- Both content-dependent reads together preserve mass, CM113. -/
theorem sum_trail (hn : 0 < n) (p : Parameters O d) (x : Fin n → O) (q : O) :
    ∑ kl, trail p x q kl = 1 := by
  simp only [trail, Fintype.sum_prod_type, ← Finset.mul_sum, sum_jump hn, mul_one]

/-- Each normalized read contracts the uniform score bound, CM113. -/
theorem weighted_bound {I : Type*} [Fintype I] (w v : I → ℝ)
    (hw : ∀ i, 0 ≤ w i) (hs : ∑ i, w i = 1) (hv : ∀ i, |v i| ≤ 1) :
    |∑ i, w i * v i| ≤ 1 := by
  calc
    _ ≤ ∑ i, |w i * v i| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, w i := by
      apply Finset.sum_le_sum
      intro i _
      rw [abs_mul, abs_of_nonneg (hw i)]
      simpa using mul_le_mul_of_nonneg_left (hv i) (hw i)
    _ = 1 := hs

/-- A pair of bounded routes remains bounded after the actual gate, CM113. -/
theorem blend_bound {t a b : ℝ} (ht : 0 ≤ t ∧ t ≤ 1) (ha : |a| ≤ 1) (hb : |b| ≤ 1) :
    |blend t a b| ≤ 1 := by
  calc
    _ ≤ |(1 - t) * a| + |t * b| := abs_add_le _ _
    _ ≤ (1 - t) * 1 + t * 1 := by
      rw [abs_mul, abs_mul, abs_of_nonneg (sub_nonneg.mpr ht.2), abs_of_nonneg ht.1]
      exact add_le_add (mul_le_mul_of_nonneg_left ha (sub_nonneg.mpr ht.2))
        (mul_le_mul_of_nonneg_left hb ht.1)
    _ = 1 := by ring

/-- Every original class score has unit capacity at every sequence length,
including the empty and single-object inputs, CM113. -/
theorem score_bound (p : Parameters O d) (hp : Admissible p) (xs : List O) (c : O) :
    |score p xs c| ≤ 1 := by
  unfold score
  split
  · exact decode_bound ..
  · exact decode_bound ..
  · rename_i a rest he
    have hn : 0 < (a :: rest).reverse.length := by simp
    apply blend_bound hp.2
    · apply blend_bound hp.1
      · exact weighted_bound _ _ (fun _ => jump_nonneg ..) (sum_jump hn ..) (fun _ => decode_bound ..)
      · exact weighted_bound _ _ (fun _ => trail_nonneg ..) (sum_trail hn ..) (fun _ => decode_bound ..)
    · exact weighted_bound _ _ (fun _ => jump_nonneg ..) (sum_jump hn ..) (fun _ => decode_bound ..)

example : |average (fun _ : Fin 1 => (1 : ℝ))| ≤ 1 := average_bound _ (by norm_num)
example : |∑ _ : Fin 2, (1 / 2 : ℝ) * (-1)| ≤ 1 :=
  weighted_bound (fun _ => 1 / 2) (fun _ => -1) (by norm_num) (by norm_num [Fin.sum_univ_two])
    (by norm_num)
example : |blend (1 / 2) (-1) 1| ≤ 1 := blend_bound (by norm_num) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

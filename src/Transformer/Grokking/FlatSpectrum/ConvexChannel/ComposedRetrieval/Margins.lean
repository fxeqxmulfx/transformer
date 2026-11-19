module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval.Basic

/-!
# Sufficient margins for composed learned reads

Source: `review/convex/moments.md`, CM113. The four gaps describe actual
learned scores on an evaluated memory. They are measured outside the forward
and fitting. The second query is the raw object reached by the first jump.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

open scoped BigOperators
open Perspective

variable {O : Type*} {d n : ℕ}

/-- Indices and proposed lower bounds used in a diagnostic, CM113. -/
structure Hop (n : ℕ) where
  source : Fin n
  dest : Fin n
  keyGap : ℝ
  linkGap : ℝ

/-- Both actual source and link scores beat all competing positions, CM113. -/
def Hop.Separated (s : Hop n) (p : Parameters O d) (x : Fin n → O) (o : O) : Prop :=
  (∀ j, j ≠ s.source → s.keyGap ≤ keyScore p x o s.source - keyScore p x o j) ∧
  (∀ k, k ≠ s.dest → s.linkGap ≤ linkScore p s.source s.dest - linkScore p s.source k)

/-- Conservative missing mass from the two softmax denominators, CM113. -/
noncomputable def Hop.error (s : Hop n) : ℝ := n * (Real.exp (-s.keyGap) + Real.exp (-s.linkGap))

private theorem weight_le_one {I : Type*} [Fintype I] {w : I → ℝ}
    (hw : ∀ i, 0 ≤ w i) (hs : ∑ i, w i = 1) (i : I) : w i ≤ 1 :=
  hs ▸ Finset.single_le_sum (fun j _ => hw j) (Finset.mem_univ i)

private theorem miss_mul {a b : ℝ} (ha : a ≤ 1) (hb : b ≤ 1) :
    1 - a * b ≤ (1 - a) + (1 - b) := by
  nlinarith [mul_nonneg (sub_nonneg.mpr ha) (sub_nonneg.mpr hb)]

/-- A good source gives a lower bound on the entire destination mass.
Other sources can also reach that destination, CM113. -/
theorem jump_error (p : Parameters O d) (x : Fin n → O) (q : O) (s : Hop n)
    (hs : s.Separated p x q) : 1 - jump p x q s.dest ≤ s.error := by
  let a := softmaxWeight (keyScore p x q)
  let b := softmaxWeight (linkScore p s.source)
  have hn := Nat.zero_lt_of_lt s.source.isLt
  have hpart : a s.source * b s.dest ≤ jump p x q s.dest :=
    Finset.single_le_sum (fun j _ => mul_nonneg (softmaxWeight_nonneg _ j)
      (softmaxWeight_nonneg _ s.dest)) (Finset.mem_univ s.source)
  have ha : a s.source ≤ 1 := weight_le_one (softmaxWeight_nonneg _) (sum_softmaxWeight hn _) _
  have hb : b s.dest ≤ 1 := weight_le_one (softmaxWeight_nonneg _) (sum_softmaxWeight hn _) _
  calc
    _ ≤ 1 - a s.source * b s.dest := by linarith
    _ ≤ (1 - a s.source) + (1 - b s.dest) := miss_mul ha hb
    _ ≤ n * Real.exp (-s.keyGap) + n * Real.exp (-s.linkGap) :=
      add_le_add (LearnedRetrieval.softmax_error _ _ _ hs.1) (LearnedRetrieval.softmax_error _ _ _ hs.2)
    _ = _ := by rw [Hop.error, mul_add]

/-- Four score gaps bound the lost mass along the genuine two-hop chain.
The second match uses the first destination's object, CM113. -/
theorem trail_error (p : Parameters O d) (x : Fin n → O) (q : O) (s t : Hop n)
    (hs : s.Separated p x q) (ht : t.Separated p x (x s.dest)) :
    1 - trail p x q (s.dest, t.dest) ≤ s.error + t.error := by
  have hn := Nat.zero_lt_of_lt s.dest.isLt
  apply (miss_mul (weight_le_one (fun _ => jump_nonneg ..) (sum_jump hn p x q) s.dest)
    (weight_le_one (fun _ => jump_nonneg ..) (sum_jump hn p x (x s.dest)) t.dest)).trans
  exact add_le_add (jump_error p x q s hs) (jump_error p x (x s.dest) t ht)

/-- The first read alone can identify a different intermediate object before
the second read is applied, CM113. -/
theorem readOne_margin (p : Parameters O d) (x : Fin n → O) (q c : O) (s : Hop n)
    (m : ℝ) (hm : 0 ≤ m) (hs : s.Separated p x q)
    (hc : m ≤ 1 - decode p (p.value (x s.dest)) 0 c) :
    m - (m+2)*s.error ≤ readOne p x q 0 (x s.dest) - readOne p x q 0 c := by
  let v (k : Fin n) := decode p (p.value (x k)) 0 (x s.dest) - decode p (p.value (x k)) 0 c
  have hv (k : Fin n) : -2 ≤ v k := by
    dsimp [v]
    linarith [(abs_le.mp (decode_bound p (p.value (x k)) 0 (x s.dest))).1,
      (abs_le.mp (decode_bound p (p.value (x k)) 0 c)).2]
  have good : m ≤ v s.dest := by simpa [v, self_decode] using hc
  have hw := LearnedRetrieval.weighted_gap (jump p x q) v s.dest m
    (fun _ => jump_nonneg ..) (sum_jump (Nat.zero_lt_of_lt s.source.isLt) ..) hv good
  have hsum : (∑ k, jump p x q k * v k) = readOne p x q 0 (x s.dest) - readOne p x q 0 c := by
    simp only [v, readOne, mul_sub, Finset.sum_sub_distrib]
  rw [hsum] at hw
  linarith [mul_le_mul_of_nonneg_left (jump_error p x q s hs) (show 0 ≤ m+2 by linarith)]

/-- With unrotated decoder gap `m` at the final linked object, the actual
two-hop read retains the following margin against a rival, CM113. -/
theorem readTwo_margin (p : Parameters O d) (x : Fin n → O) (q c : O) (s t : Hop n)
    (m : ℝ) (hm : 0 ≤ m) (hs : s.Separated p x q) (ht : t.Separated p x (x s.dest))
    (hc : m ≤ 1 - decode p (p.value (x t.dest)) 0 c) :
    m - (m + 2) * (s.error + t.error) ≤ readTwo p x q (x t.dest) - readTwo p x q c := by
  let v (kl : Fin n × Fin n) := decode p (p.value (x kl.2)) 0 (x t.dest) -
    decode p (p.value (x kl.2)) 0 c
  have hv (kl : Fin n × Fin n) : -2 ≤ v kl := by
    have ht' := (abs_le.mp (decode_bound p (p.value (x kl.2)) 0 (x t.dest))).1
    have hc' := (abs_le.mp (decode_bound p (p.value (x kl.2)) 0 c)).2
    dsimp [v]
    linarith
  have good : m ≤ v (s.dest, t.dest) := by simpa [v, self_decode] using hc
  have hw := LearnedRetrieval.weighted_gap (trail p x q) v (s.dest,t.dest) m
    (fun _ => trail_nonneg ..) (sum_trail (Nat.zero_lt_of_lt s.source.isLt) ..) hv good
  have he := mul_le_mul_of_nonneg_left (trail_error p x q s t hs ht) (show 0 ≤ m+2 by linarith)
  have hsum : (∑ kl, trail p x q kl * v kl) = readTwo p x q (x t.dest) - readTwo p x q c := by
    simp only [v, readTwo, mul_sub, Finset.sum_sub_distrib]
  rw [hsum] at hw
  linarith

/-- Lower bound for the complete mixed logit difference, CM113. The two
other routes can each contribute a margin as low as minus two. -/
noncomputable def guaranteed (p : Parameters O d) (s t : Hop n) (m : ℝ) : ℝ :=
  let w := (1 - p.product) * p.depth
  w * (m - (m + 2) * (s.error + t.error)) - 2 * (1 - w)

private theorem blend_gap {u h a a' b b' e e' M : ℝ} (hu : 0 ≤ u ∧ u ≤ 1)
    (hh : 0 ≤ h ∧ h ≤ 1) (ha : -2 ≤ a-a') (hb : M ≤ b-b') (he : -2 ≤ e-e') :
    (1-u)*h*M - 2*(1-(1-u)*h) ≤ blend u (blend h a b) e - blend u (blend h a' b') e' := by
  calc
    _ = (1-u)*(1-h)*(-2) + (1-u)*h*M + u*(-2) := by ring
    _ ≤ (1-u)*(1-h)*(a-a') + (1-u)*h*(b-b') + u*(e-e') :=
      add_le_add (add_le_add (mul_le_mul_of_nonneg_left ha
        (mul_nonneg (sub_nonneg.mpr hu.2) (sub_nonneg.mpr hh.2)))
        (mul_le_mul_of_nonneg_left hb (mul_nonneg (sub_nonneg.mpr hu.2) hh.1)))
        (mul_le_mul_of_nonneg_left he hu.1)
    _ = _ := by unfold blend; ring

/-- The bound applies to the complete trained output including learned gates
and the competing one-read and phase-product routes, CM113. -/
theorem logit_margin (p : Parameters O d) (hp : Admissible p) (x : Fin n → O)
    (q c : O) (s t : Hop n) (m : ℝ) (hm : 0 ≤ m)
    (hs : s.Separated p x q) (ht : t.Separated p x (x s.dest))
    (hc : m ≤ 1 - decode p (p.value (x t.dest)) 0 c) :
    guaranteed p s t m ≤ logit p x q (x t.dest) - logit p x q c := by
  have hn := Nat.zero_lt_of_lt s.source.isLt
  have bound (turn : Fin (d+1) → ℝ) (o : O) : |readOne p x q turn o| ≤ 1 :=
    weighted_bound _ _ (fun _ => jump_nonneg ..) (sum_jump hn ..) (fun _ => decode_bound ..)
  have gap (turn : Fin (d+1) → ℝ) : -2 ≤ readOne p x q turn (x t.dest) - readOne p x q turn c := by
    linarith [(abs_le.mp (bound turn (x t.dest))).1, (abs_le.mp (bound turn c)).2]
  exact blend_gap hp.2 hp.1 (gap 0) (readTwo_margin p x q c s t m hm hs ht hc) (gap _)

/-- Positive complete margin supplies the original object against every rival.
No path or label is provided to the predictor, CM113. -/
theorem correct_object (p : Parameters O d) (hp : Admissible p) (x : Fin n → O) (q : O)
    (s t : Hop n) (m : ℝ) (hm : 0 ≤ m) (hs : s.Separated p x q)
    (ht : t.Separated p x (x s.dest))
    (hc : ∀ c, c ≠ x t.dest → m ≤ 1 - decode p (p.value (x t.dest)) 0 c)
    (hg : 0 < guaranteed p s t m) :
    ∀ c, c ≠ x t.dest → logit p x q c < logit p x q (x t.dest) := by
  intro c hne
  linarith [logit_margin p hp x q c s t m hm hs ht (hc c hne)]

/-- The same actual output margin bounds ordinary CE over every original
class after a nonnegative capacity scaling, CM113. -/
theorem crossEntropy_bound [Fintype O] (p : Parameters O d) (hp : Admissible p)
    (x : Fin n → O) (q : O) (s t : Hop n) (m B : ℝ) (hm : 0 ≤ m) (hB : 0 ≤ B)
    (hs : s.Separated p x q) (ht : t.Separated p x (x s.dest))
    (hc : ∀ c, c ≠ x t.dest → m ≤ 1 - decode p (p.value (x t.dest)) 0 c) :
    Landscape.Neural.crossEntropy (fun c => B * logit p x q c) (x t.dest) ≤
      Real.log (1 + (Fintype.card O - 1) * Real.exp (-(B * guaranteed p s t m))) := by
  apply (Training.crossEntropy_le_of_le_gap (μ := B * guaranteed p s t m) ?_).1
  intro c hne
  simpa only [mul_sub] using mul_le_mul_of_nonneg_left (logit_margin p hp x q c s t m hm hs ht (hc c hne)) hB

end Transformer.Grokking.FlatSpectrum.ConvexChannel.ComposedRetrieval

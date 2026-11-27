module

public import Transformer.Grokking.FiniteRecovery.ArgandMomentum

/-! A native draw witnessing large binary logit gaps when the target
only uses the token at the readout position. The witness is used to
locate an open set of Gaussian draws for the full update. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped BigOperators InnerProductSpace

variable {T layers : ℕ}

/-- Scalar embedding coordinates for the identity and constant witnesses. -/
def binaryEmbedding (r : ℝ) (mode : Option Bool) (b : Bool) : ℝ :=
  match mode with
  | none => if b then r else -r
  | some a => if b = a then 2 * r else r

/-- Only the freely drawn input/output embeddings are nonzero in this
witness. The native score, frequency and gate initializers still apply. -/
def binaryWitnessDraw (r : ℝ) (mode : Option Bool) : ComplexParameter Bool 1 1 layers → ℂ
  | .embedding b _ => binaryEmbedding r mode b
  | _ => 0

private def scalarEmbedding (x : ℝ) : CSpace 1 :=
  WithLp.toLp 2 fun _ => (x : ℂ)

private theorem argandRMS_scalarEmbedding (eps x : ℝ) :
    argandRMS eps (scalarEmbedding x) = scalarEmbedding (x / (|x| + eps)) := by
  have hnorm : ‖scalarEmbedding x‖ = |x| := by
    rw [PiLp.norm_eq_of_L2, Fin.sum_univ_one, Real.sqrt_sq_eq_abs]
    simp [scalarEmbedding, Real.norm_eq_abs]
  rw [argandRMS, hnorm]
  ext i
  rw [PiLp.smul_apply]
  change (↑(Real.sqrt (↑(1 : ℕ)) / (|x| + eps)) : ℂ) • (x : ℂ) = ↑(x / (|x| + eps))
  rw [smul_eq_mul, Nat.cast_one, Real.sqrt_one, ← Complex.ofReal_mul]
  congr 1
  simp [div_eq_mul_inv, mul_comm]

set_option maxHeartbeats 1000000 in
private theorem witness_logits (r : ℝ) (mode : Option Bool) (repeats : ℕ) (word : Word Bool T)
    (i : Fin T) (a : Bool) :
    let eps : ℝ := 1 / 1000000
    let x := binaryEmbedding r mode (word i)
    let y := x / (|x| + eps)
    (Model.fromWeights (sgdInitialWeights true (binaryWitnessDraw (layers := layers) r mode))).logits
        repeats word i a = y / (|y| + eps) * binaryEmbedding r mode a / 2 := by
  let draw : ComplexParameter Bool 1 1 layers → ℂ := binaryWitnessDraw r mode
  let m := Model.fromWeights (initialWeights draw)
  let eps : ℝ := 1 / 1000000
  let scale := 1 / Real.sqrt (2 * (layers : ℝ) * repeats)
  have hi : sgdInitialWeights true draw = initialWeights draw := by
    ext (⟨p, b⟩ | k)
    · cases p <;> cases b <;> rfl
    · rfl
  have he (b : Bool) : m.embedding b = scalarEmbedding (binaryEmbedding r mode b) := by
    ext j
    exact Complex.ext rfl rfl
  have hop (l : Fin layers) : (m.layer l).attentionProjections.WO = 0 := by
    have hc (h : Fin 1) (j : Fin 1) :
        (WithLp.toLp 2 (fun k : Fin 1 => complexEntry (initialWeights draw)
          (.attentionOut l h j k)) : CSpace 1) = 0 := by
      ext k
      exact Complex.ext rfl rfl
    change headProjection _ = 0
    simp [headProjection, vectorProjection, hc]
  have hfp (l : Fin layers) : (m.layer l).feedForward.WF = 0 := by
    have hc (j : Fin 4) : (WithLp.toLp 2 (fun k : Fin 1 =>
        complexEntry (initialWeights draw) (.feedForwardOut l j k)) : CSpace 1) = 0 := by
      ext k
      exact Complex.ext rfl rfl
    change vectorProjection (fun j : Fin 4 => (WithLp.toLp 2 (fun k : Fin 1 =>
      complexEntry (initialWeights draw) (.feedForwardOut l j k)) : CSpace 1)) = 0
    simp [vectorProjection, hc]
  have hl (l : Fin layers) (z : Word (CSpace 1) T) :
      (m.layer l).forwardWith eps scale (causalGeometry m.alpha) z = z := by
    obtain ⟨_, _, hag, hfg⟩ := initialWeights_layer draw l
    change (m.layer l).attentionGates = _ at hag
    change (m.layer l).feedForwardGates = _ at hfg
    funext j
    simp [Layer.forwardWith, sublayer, attentionWith, hop, hfp, hag, hfg, argandRMS]
  let slide (z : Word (CSpace 1) T) (l : Fin layers) :=
    (m.layer l).forwardWith eps scale (causalGeometry m.alpha) z
  have hpass : (List.finRange layers).foldl slide = id := by
    have hp (ls : List (Fin layers)) (z : Word (CSpace 1) T) : ls.foldl slide z = z := by
      induction ls generalizing z with
      | nil => rfl
      | cons l ls ih => simpa only [List.foldl_cons, slide, hl] using ih z
    funext z
    exact hp _ z
  change (Model.fromWeights (sgdInitialWeights true draw)).logits repeats word i a = _
  rw [hi]
  change m.logits repeats word i a = _
  have hh : m.hidden repeats word i =
      argandRMS eps (argandRMS eps (m.embedding (word i))) := by
    simp only [Model.hidden, Model.hiddenWith, List.ofFn_eq_map, List.foldl_map]
    change argandRMS eps (((List.finRange layers).foldl slide)^[repeats]
      (fun j => argandRMS eps (m.embedding (word j))) i) = _
    rw [hpass, Function.iterate_id, id_eq]
  rw [Model.logits, hh, he, argandRMS_scalarEmbedding, argandRMS_scalarEmbedding]
  simp only [Model.readout, he, PiLp.inner_apply, Fin.sum_univ_one, scalarEmbedding,
    RCLike.inner_apply, Complex.mul_re, Complex.conj_re, Complex.conj_im,
    Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero, eps]

/-- Scaling the embeddings gives arbitrarily large gaps despite the two
RMS normalizations, since the tied readout uses the raw embedding. -/
theorem binaryWitness_gap {r : ℝ} (hr : 1 ≤ r) (mode : Option Bool) (repeats : ℕ)
    (word : Word Bool T) (i : Fin T) :
    let model := Model.fromWeights (sgdInitialWeights true (binaryWitnessDraw (layers := layers) r mode))
    let label := mode.getD (word i)
    r / 4 ≤ model.logits repeats word i label - model.logits repeats word i (!label) := by
  let eps : ℝ := 1 / 1000000
  let radial (x : ℝ) := x / (|x| + eps)
  have half (x : ℝ) (hx : eps ≤ x) : 1 / 2 ≤ radial x := by
    have heps : 0 < eps := by norm_num [eps]
    unfold radial
    rw [abs_of_nonneg (heps.le.trans hx)]
    apply (le_div_iff₀ (by linarith : 0 < x + eps)).mpr
    calc
      1 / 2 * (x + eps) ≤ (1 / 2 : ℝ) * (x + x) := by gcongr
      _ = x := by ring
  have twice (x : ℝ) (hx : 1 ≤ x) : 1 / 2 ≤ radial (radial x) :=
    half _ ((by norm_num [eps] : eps ≤ 1 / 2).trans
      (half x ((by norm_num [eps] : eps ≤ 1).trans hx)))
  have neg (x : ℝ) : radial (-x) = -radial x := by simp [radial, neg_div]
  have bound (x : ℝ) (hx : 1 ≤ x) : r / 2 ≤ radial (radial x) * r := by
    simpa [div_eq_mul_inv, mul_comm r] using mul_le_mul_of_nonneg_right (twice x hx) (by linarith : 0 ≤ r)
  have h1 := bound r hr
  have h2 := bound (2 * r) (by linarith)
  dsimp only
  rw [witness_logits, witness_logits]
  change r / 4 ≤ radial (radial (binaryEmbedding r mode (word i))) *
    binaryEmbedding r mode (mode.getD (word i)) / 2 -
    radial (radial (binaryEmbedding r mode (word i))) * binaryEmbedding r mode (!(mode.getD (word i))) / 2
  cases mode with
  | none =>
    cases h : word i <;> simp [binaryEmbedding, neg] <;> linarith
  | some a =>
    cases a <;> cases h : word i <;> simp [binaryEmbedding] <;>
      nlinarith only [h1, h2]

example (word : Word Bool 16) :=
  binaryWitness_gap (layers := 2) (r := 16) (by norm_num) none 3 word 0

end Transformer.Grokking.FiniteRecovery.Argand

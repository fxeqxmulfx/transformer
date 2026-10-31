module

public import Transformer.CRASP.Rotary.Source

/-!
# Uniform counting formulas for the full normalized source architecture

Source: diff_gpt b98c665b, `gpt.py`, `block.py`, `mha.py`, `feed_forward.py`,
`rope.py`; `review/flat_spectrum.md`, *Uniform nonperiodic RoPE*, U3--U4.
One formula handles every finite input word with the exact geometric grid.
The explicit fixed-point rounding model of `Normalized` is retained. Extending
the mathematical grid to all naturals does not extend the Python KV cache.
-/

@[expose] public section
noncomputable section

namespace Transformer.CRASP.Rotary

open GPTMini SequenceModel.Attention

/-- Put a finite residual sequence into the previous position-tagged state space. -/
def boundedStream {cfg : Config} {T p s : ℕ}
    (x : Fin T → Fin cfg.d_model → Fx p s) (j : ℕ) : Normalized.State cfg T p s :=
  if h : 0 < j ∧ j ≤ T then
    some (⟨j - 1, by omega⟩, x ⟨j - 1, by omega⟩) else none

/-- The same residuals with only the first-position flag retained. -/
def stream {cfg : Config} {T p s : ℕ}
    (x : Fin T → Fin cfg.d_model → Fx p s) (j : ℕ) : State cfg p s :=
  if h : 0 < j ∧ j ≤ T then
    tagged (⟨j - 1, by omega⟩ : Fin T) (x ⟨j - 1, by omega⟩) else (false, fun _ => 0)

/-- Original finite coordinates are retained at their one-based logic positions. -/
theorem boundedStream_at {cfg : Config} {T p s : ℕ}
    (x : Fin T → Fin cfg.d_model → Fx p s) (i : Fin T) :
    boundedStream x (i.val + 1) = some (i, x i) := by
  have h : 0 < i.val + 1 ∧ i.val + 1 ≤ T := by have := i.isLt; omega
  simp only [boundedStream, dite_eq_left h, Nat.add_sub_cancel, Fin.eta]

/-- Tag erasure retains the correct first-position flag and residual. -/
theorem stream_at {cfg : Config} {T p s : ℕ}
    (x : Fin T → Fin cfg.d_model → Fx p s) (i : Fin T) :
    stream x (i.val + 1) = tagged i (x i) := by
  have h : 0 < i.val + 1 ∧ i.val + 1 ≤ T := by have := i.isLt; omega
  simp only [stream, dite_eq_left h, Nat.add_sub_cancel, Fin.eta]

/-- The earlier zero fallback is precisely totalized real division followed
by rounding; its zero-valued sentinel contributes no spurious average. -/
theorem normalized_aggregate {Q C : Type*} {p s : ℕ}
    (L : Normalized.Layer Q C p s) (bos : Q) (q : ℕ → Q) (i : ℕ) (c : C) :
    L.aggregate bos q i c = Fx.round p s
      ((CountCells.sum bos (L.numerator (q i) c) q i : ℝ) /
        (CountCells.sum bos (L.denominator (q i) c) q i : ℝ)) := by
  unfold Normalized.Layer.aggregate CountCells.value
  split_ifs with h
  · rw [h]
    simp [CountCells.sum]
  · rfl

/-- Reindexing keeps the whole prefix sum, including each token's own value (U3). -/
theorem coefficient_sum {cfg : Config} {T p s : ℕ}
    (x : Fin T → Fin cfg.d_model → Fx p s) (i : Fin T)
    (f : Normalized.State cfg T p s → Fx p s) (g : State cfg p s → ℕ → Fx p s)
    (hz : f none = 0)
    (hfg : ∀ j : Fin T, f (some (j, x j)) = g (tagged j (x j)) (j.val + 1)) :
    CountCells.sum none f (boundedStream x) (i.val + 1) =
      ∑ j ∈ Finset.Icc 1 (i.val + 1), (g (stream x j) j).m := by
  rw [CountCells.sum, hz]
  simp only [Fx.m_zero, zero_add]
  apply Finset.sum_congr rfl
  intro j hj
  have hjb := Finset.mem_Icc.mp hj
  have hjt : 0 < j ∧ j ≤ T := by have := i.isLt; omega
  let idx : Fin T := ⟨j - 1, by omega⟩
  have he : idx.val + 1 = j := by dsimp [idx]; omega
  have h := congrArg Fx.m (hfg idx)
  simpa only [boundedStream, stream, dite_eq_left hjt, idx, he] using h

/-- Equality of all exact RoPE coefficients yields equality of the complete
rounded attention quotient, at every head coordinate and finite prefix (U3). -/
theorem aggregate_restrict (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : ℕ → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → Fin cfg.d_model → Fx p s) (i : Fin T) (c : Normalized.Coordinate cfg) :
    (Normalized.sourceLayer cfg B eps xeps p s (fun a : Fin T => angle a.val)).aggregate
      none (boundedStream x) (i.val + 1) c =
    (sourceLayer cfg B eps xeps p s angle).aggregate (tagged i (x i))
      (stream x) (i.val + 1) c := by
  let old := Normalized.sourceLayer cfg B eps xeps p s (fun a : Fin T => angle a.val)
  let new := sourceLayer cfg B eps xeps p s angle
  have hd := coefficient_sum x i (old.denominator (some (i, x i)) c)
    (fun a j => new.denominator (tagged i (x i)) c a (i.val + 1) j) rfl
    (fun j => (denominator_restrict cfg B eps xeps p s angle i j (x i) (x j) c).symm)
  have hn := coefficient_sum x i (old.numerator (some (i, x i)) c)
    (fun a j => new.numerator (tagged i (x i)) c a (i.val + 1) j) rfl
    (fun j => (numerator_restrict cfg B eps xeps p s angle i j (x i) (x j) c).symm)
  change old.aggregate none (boundedStream x) (i.val + 1) c = _
  rw [normalized_aggregate, boundedStream_at, hd, hn]
  rfl

/-- The entire normalized block commutes with position-tag erasure at every
finite context, including XSA, both residuals and the normalized SwiGLU branch. -/
theorem step_restrict (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : ℕ → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → Fin cfg.d_model → Fx p s) (i : Fin T) :
    (Normalized.sourceLayer cfg B eps xeps p s (fun a : Fin T => angle a.val)).step
      none (boundedStream x) (i.val + 1) =
      some (i, ((sourceLayer cfg B eps xeps p s angle).step (stream x) (i.val + 1)).2) := by
  have ha : (Normalized.sourceLayer cfg B eps xeps p s
      (fun a : Fin T => angle a.val)).aggregate none (boundedStream x) (i.val + 1) =
      (sourceLayer cfg B eps xeps p s angle).aggregate (tagged i (x i))
        (stream x) (i.val + 1) := by
    funext c
    exact aggregate_restrict cfg B eps xeps p s angle x i c
  rw [Normalized.Layer.step, boundedStream_at, ha, Layer.step, stream_at]
  exact finish_restrict cfg B eps xeps p s angle i (x i) _

/-- The sum-agreement hypotheses have nonempty finite source witnesses. -/
example : (fun _ : Normalized.State witnessConfig 2 2 0 => (0 : Fx 2 0)) none = 0 ∧
    (∀ j : Fin 2,
      (fun _ : Normalized.State witnessConfig 2 2 0 => (0 : Fx 2 0))
        (some (j, fun _ => 0)) =
      (fun (_ : State witnessConfig 2 0) (_ : ℕ) => (0 : Fx 2 0))
        (tagged j (fun _ => 0)) (j.val + 1)) := ⟨rfl, fun _ => rfl⟩

/-- Execute the rounded source forward on the original input vocabulary. -/
def predict (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) (p s : ℕ)
    (enabled : Fin cfg.n_layers → Bool) (w : List (Fin cfg.vocab_size))
    (i : ℕ) (target : Fin cfg.vocab_size) : Bool :=
  readout cfg θ eps target
    (run (layers cfg θ eps xeps p s enabled)
      (input (embed cfg θ eps p s) (false, fun _ => 0) w) cfg.n_layers i)

/-- The first-position bit survives all layers; exact RoPE positions remain
external to the finite residual state (U3). -/
theorem first_flag_preserved (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ)
    (p s : ℕ) (enabled : Fin cfg.n_layers → Bool) (q : ℕ → State cfg p s)
    (n i : ℕ) : (run (layers cfg θ eps xeps p s enabled) q n i).1 = (q i).1 := by
  induction n with
  | zero => rfl
  | succ n ih => exact ih

/-- A single syntactically finite formula, fixed before the input word,
simulates the whole normalized source model on all finite lengths. The exact
source grid requires no rational-angle, period, or context-size assumption (U4). -/
theorem uniform_source_classifier (cfg : Config) (θ : ModelParams cfg)
    (eps xeps : ℝ) (p s : ℕ) (enabled : Fin cfg.n_layers → Bool)
    (target : Fin cfg.vocab_size) :
    ∃ φ : Form (Fin cfg.vocab_size) (Key (State cfg p s) (Normalized.Coordinate cfg)),
      φ.depth ≤ cfg.n_layers ∧
      ∀ (w : List (Fin cfg.vocab_size)) (i : ℕ), 0 < i → i ≤ w.length →
        φ.sat (kernel (layers cfg θ eps xeps p s enabled)) w i =
          predict cfg θ eps xeps p s enabled w i target := by
  let L := layers cfg θ eps xeps p s enabled
  refine ⟨classifier L (embed cfg θ eps p s) (readout cfg θ eps target) cfg.n_layers,
    classifier_depth _ _ _ _, ?_⟩
  intro w i hi hw
  apply classifier_correct L
  · intro l
    exact sourceLayer_nonnegative _ _ _ _ _ _ _
  · exact hi
  · exact hw

/-- The classifier covers a concrete three-token input with all rotary layers
enabled, at source RMS/XSA epsilons, without a context-size premise. -/
example (θ : ModelParams witnessConfig) :
    ∃ φ : Form (Fin witnessConfig.vocab_size)
        (Key (State witnessConfig 8 4) (Normalized.Coordinate witnessConfig)),
      φ.depth ≤ 2 ∧ φ.sat
        (kernel (layers witnessConfig θ (1 / 8388608) (1 / 1000000000000) 8 4
          (fun _ => true))) [0, 1, 2] 3 =
        predict witnessConfig θ (1 / 8388608) (1 / 1000000000000) 8 4
          (fun _ => true) [0, 1, 2] 3 1 := by
  obtain ⟨φ, hd, hs⟩ := uniform_source_classifier witnessConfig θ
    (1 / 8388608) (1 / 1000000000000) 8 4 (fun _ => true) 1
  exact ⟨φ, hd, hs [0, 1, 2] 3 (by decide) (by decide)⟩

end Transformer.CRASP.Rotary

import Transformer.GPTMini.Semantics.DepthQKV
import Transformer.GPTMini.Semantics.DepthNormalization
import Transformer.GPTMini.Semantics.DepthUniformProjection

/-!
# Complete original depth attention and output projection

Source: original attnSubLayer, fused QKV, head merge and W_o at
f11b6e2. A simultaneous ordinary output matrix reads the first
coordinates of heads zero and one into specified residual channels.
Together with DepthQKV this is a complete original attention sublayer.

Its signal is derived from the actual coordinate of rmsNormEps(x_j),
the true diagonal-inclusive causal mean and original XSA attenuation.
There is no supplied position scale, Boolean presence or modified
attention operator. All actual positions, RoPE and QKNorm remain.

The complete sublayer formula and every unwritten residual coordinate
are proved on arbitrary real input arrays. Distinct output coordinates
read their own signal exactly. Actual amplitude/norm bounds and the
ordered feature recurrence must still be derived from raw inputs.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators
open Transformer.Basis

/-- The actual merged coordinate of each assigned head's first value component.
Source: inverse of the original four-head view in GPTMini.Reshape. -/
noncomputable def depthMergeIndex (mode : Mode) (branch : Fin 2) : Fin (depthConfig mode).d_model :=
  (headSplit (depthConfig mode)).symm (depthHead mode branch, depthHeadZero mode)

/-- One genuine shared W_o reads both assigned merged head components into ordinary residual coordinates.
Source: two rank-one rows in the original bias-free output-matrix type. -/
noncomputable def depthUniformOut (mode : Mode) (target : Fin 2 → Fin (depthConfig mode).d_model) :
    EucSpace (depthConfig mode).d_model →L[ℝ] EucSpace (depthConfig mode).d_model :=
  ∑ branch : Fin 2, (EuclideanSpace.proj (depthMergeIndex mode branch)).smulRight
    (EuclideanSpace.single (target branch) 1)

/-- The true output matrix reads its two intended components of the actual head merge.
Source: ordinary rank-one sum and the original inverse head reshape. -/
theorem depthUniformOut_merge (mode : Mode) (target : Fin 2 → Fin (depthConfig mode).d_model)
    (heads : Fin (depthConfig mode).n_heads → EucSpace (depthConfig mode).head_dim) :
    depthUniformOut mode target (headMerge (depthConfig mode) heads) =
      ∑ branch : Fin 2, heads (depthHead mode branch) (depthHeadZero mode) • EuclideanSpace.single (target branch) 1 := by
  unfold depthUniformOut
  simp only [sum_apply, ContinuousLinearMap.smulRight_apply, EuclideanSpace.proj, PiLp.proj_apply]
  apply Finset.sum_congr rfl
  intro branch hb
  rw [depthMergeIndex, headMerge_apply, Equiv.apply_symm_apply]

/-- A complete ordinary attention parameter record for both uniform depth heads.
Source: unchanged GPTMini AttnParams, with one simultaneous fused matrix and one actual output matrix. -/
noncomputable def depthAttention (mode : Mode) (source target : Fin 2 → Fin (depthConfig mode).d_model) :
    AttnParams (depthConfig mode) where
  W_qkv := depthValueQKV mode source
  W_o := depthUniformOut mode target
  log_alpha := fun _ => 0

/-- Every scalar V amplitude is the coordinate of the genuine current prenormed residual.
Source: evaluated original V rows; neither a token feature nor a normalization oracle is supplied. -/
noncomputable def depthFeatureAmplitude (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) (branch : Fin 2) (j : Fin T) : ℝ :=
  rmsNormEps eps (x j) (source branch)

/-- Reading the same true prenorm yields its actual position-dependent multiplier times the original coordinate.
Source: genuine rmsNormEps coordinate arithmetic. -/
theorem depthFeatureAmplitude_scale (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) (branch : Fin 2) (j : Fin T) :
    depthFeatureAmplitude mode source eps x branch j = depthScale mode eps (x j) * x j (source branch) := by
  rw [depthFeatureAmplitude, depthScale_rms]
  simp only [PiLp.smul_apply, smul_eq_mul]

/-- The continuous scalar signal of the complete original uniform head at this actual residual array.
Source: genuine prenorm coordinates, finite causal mean and original XSA coefficient. -/
noncomputable def depthAttentionSignal (mode : Mode) (source : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (x : Fin T → EucSpace (depthConfig mode).d_model) (branch : Fin 2) (i : Fin T) : ℝ :=
  depthPrefixMass (depthFeatureAmplitude mode source eps x branch) i *
    (1 - (depthSelfFraction eps (depthFeatureAmplitude mode source eps x branch i)) ^ 2)

/-- The genuine full attention sublayer has exactly these two residual contributions, with all original operators retained.
Source: complete fused-matrix slices, actual head merge/output and proved uniform RoPE/QKNorm/softmax/XSA formula. -/
theorem depthAttention_apply (mode : Mode) (source target : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) :
    attnSubLayer (depthConfig mode) (depthAttention mode source target) eps positions x i =
      ∑ branch : Fin 2, depthAttentionSignal mode source eps x branch i • EuclideanSpace.single (target branch) 1 := by
  unfold attnSubLayer
  change depthUniformOut mode target (headMerge (depthConfig mode) _) = _
  rw [depthUniformOut_merge mode target]
  apply Finset.sum_congr rfl
  intro branch hb
  have hq : (fun j => headSlice (depthConfig mode) (qkvSlice (depthConfig mode) (qkvQ (depthConfig mode))
      (depthValueQKV mode source (rmsNormEps eps (x j)))) (depthHead mode branch)) = fun _ => 0 := by
    funext j
    exact (depthValueQKV_qk mode source (rmsNormEps eps (x j)) (depthHead mode branch)).1
  have hk : (fun j => headSlice (depthConfig mode) (qkvSlice (depthConfig mode) (qkvK (depthConfig mode))
      (depthValueQKV mode source (rmsNormEps eps (x j)))) (depthHead mode branch)) = fun _ => 0 := by
    funext j
    exact (depthValueQKV_qk mode source (rmsNormEps eps (x j)) (depthHead mode branch)).2
  have hv : (fun j => headSlice (depthConfig mode) (qkvSlice (depthConfig mode) (qkvV (depthConfig mode))
      (depthValueQKV mode source (rmsNormEps eps (x j)))) (depthHead mode branch)) =
      fun j => depthFeatureAmplitude mode source eps x branch j • depthHeadDirection mode := by
    funext j
    exact depthValueQKV_branch mode source (rmsNormEps eps (x j)) branch
  change attentionHead (depthConfig mode) 0 eps _ _ _ positions i (depthHeadZero mode) •
    EuclideanSpace.single (target branch) 1 = _
  dsimp only [depthAttention]
  rw [hq, hk, hv, depthUniformHead_collinear (depthConfig mode) 0 eps positions
    (depthFeatureAmplitude mode source eps x branch) (depthHeadDirection mode) (depthHeadDirection_norm mode) i]
  simp only [PiLp.smul_apply, depthHeadDirection, PiLp.single_apply, ite_true, smul_eq_mul, mul_one]
  rfl

/-- Every residual coordinate outside the two output targets receives exactly zero actual attention contribution.
Source: the complete original sublayer formula and disjoint single-coordinate output columns. -/
theorem depthAttention_protected (mode : Mode) (source target : Fin 2 → Fin (depthConfig mode).d_model)
    (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace (depthConfig mode).d_model)
    (i : Fin T) (c : Fin (depthConfig mode).d_model) (ht : ∀ branch, c ≠ target branch) :
    attnSubLayer (depthConfig mode) (depthAttention mode source target) eps positions x i c = 0 := by
  rw [depthAttention_apply]
  simp only [WithLp.ofLp_sum, Finset.sum_apply, PiLp.smul_apply, PiLp.single_apply, smul_eq_mul]
  apply Finset.sum_eq_zero
  intro branch hb
  rw [ite_eq_right (ht branch), mul_zero]

example : ∀ branch : Fin 2, (0 : Fin 64) ≠ (if branch = 0 then 3 else 4) := by
  intro branch
  split_ifs <;> decide

/-- Distinct actual output targets read their own genuine scalar signal exactly.
Source: full original attention output, with injectivity preventing interference between the two branches. -/
theorem depthAttention_target (mode : Mode) (source target : Fin 2 → Fin (depthConfig mode).d_model)
    (ht : Function.Injective target) (eps : ℝ) {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace (depthConfig mode).d_model) (i : Fin T) (branch : Fin 2) :
    attnSubLayer (depthConfig mode) (depthAttention mode source target) eps positions x i (target branch) =
      depthAttentionSignal mode source eps x branch i := by
  rw [depthAttention_apply]
  simp only [WithLp.ofLp_sum, Finset.sum_apply, PiLp.smul_apply, PiLp.single_apply, smul_eq_mul]
  rw [Fintype.sum_eq_single branch]
  · simp only [ite_true, mul_one]
  · intro other ho
    have hne : target branch ≠ target other := fun he => ho (ht he).symm
    rw [ite_eq_right hne, mul_zero]

example : Function.Injective (fun b : Fin 2 => (⟨3 + b.val, by have hb := b.isLt; omega⟩ : Fin 64)) := by
  intro a b hab
  apply Fin.ext
  have h := congrArg Fin.val hab
  change 3 + a.val = 3 + b.val at h
  omega

end Transformer.GPTMini.Semantics

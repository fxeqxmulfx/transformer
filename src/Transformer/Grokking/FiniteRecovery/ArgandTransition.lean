module

public import Transformer.Grokking.FiniteRecovery.Argand

/-! A finite-state transition is realized by a single ordinary Argand layer.
Use tied orthogonal embeddings, a zero attention write, and the product FFN
to move the current state's basis vector to the next state's basis vector.
The skip gate and input projections account for both RMS factors. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped InnerProductSpace

namespace Transition

noncomputable def basis {c : ℕ} (a : Fin c) : CSpace c :=
  EuclideanSpace.single a 1

noncomputable def rmsAmplitude (c : ℕ) (a : ℝ) (eps : ℝ := 1) : ℝ :=
  Real.sqrt c / (a + eps) * a

theorem rmsAmplitude_pos_at {eps : ℝ} (he : 0 < eps) {c : ℕ} (hc : 0 < c) {a : ℝ} (ha : 0 < a) :
    0 < rmsAmplitude c a eps :=
  mul_pos (div_pos (Real.sqrt_pos.mpr (Nat.cast_pos.mpr hc)) (add_pos ha he)) ha

private theorem rmsAmplitude_pos {c : ℕ} (hc : 0 < c) {a : ℝ} (ha : 0 < a) :
    0 < rmsAmplitude c a :=
  rmsAmplitude_pos_at zero_lt_one hc ha

theorem argandRMS_basis_at (eps : ℝ) {c : ℕ} (i : Fin c) {a : ℝ} (ha : 0 ≤ a) :
    argandRMS eps ((a : ℂ) • basis i) = ((rmsAmplitude c a eps : ℝ) : ℂ) • basis i := by
  have hnorm : ‖basis i‖ = 1 := by simp [basis, PiLp.norm_single]
  rw [argandRMS, norm_smul, hnorm, mul_one, Complex.norm_real, Real.norm_of_nonneg ha, smul_smul]
  rw [← Complex.ofReal_mul]
  rfl

def slot {c : ℕ} (a : Fin c) : Fin (4 * c) :=
  ⟨a.val, by omega⟩

private theorem slot_injective (c : ℕ) : Function.Injective (slot (c := c)) :=
  fun _ _ h => Fin.ext (congrArg (fun j : Fin (4 * c) => j.val) h)

noncomputable def coordinates (c : ℕ) : CSpace c →ₗ[ℂ] ℂ :=
  ∑ i : Fin c, PiLp.projₗ 2 (fun _ : Fin c => ℂ) i

noncomputable def liftBasis (c : ℕ) : CSpace c →ₗ[ℂ] (Fin (4 * c) → ℂ) :=
  ∑ i : Fin c, (PiLp.projₗ 2 (𝕜 := ℂ) (fun _ : Fin c => ℂ) i).smulRight (Pi.single (slot i) 1)

noncomputable def sendBasis {c : ℕ} (next : Fin c → Fin c) :
    (Fin (4 * c) → ℂ) →ₗ[ℂ] CSpace c :=
  ∑ i : Fin c, (LinearMap.proj (R := ℂ) (φ := fun _ => ℂ) (slot i)).smulRight (basis (next i))

private theorem coordinates_basis {c : ℕ} (i : Fin c) : coordinates c (basis i) = 1 := by
  simp [coordinates, basis, LinearMap.sum_apply, PiLp.single_apply]

private theorem liftBasis_basis {c : ℕ} (i : Fin c) :
    liftBasis c (basis i) = Pi.single (slot i) 1 := by
  simp [liftBasis, basis, LinearMap.sum_apply, PiLp.single_apply, ite_smul]

private theorem sendBasis_single {c : ℕ} (next : Fin c → Fin c) (i : Fin c) :
    sendBasis next (Pi.single (slot i) 1) = basis (next i) := by
  simp [sendBasis, LinearMap.sum_apply, Pi.single_apply, (slot_injective c).eq_iff, ite_smul]

private theorem single_mul_single_one {c : ℕ} (i : Fin (4 * c)) :
    ((Pi.single i 1 : Fin (4 * c) → ℂ) * Pi.single i 1) = Pi.single i 1 := by
  funext j
  by_cases h : j = i <;> simp [h]

noncomputable def transitionLayer {H d : ℕ}
    (next : Fin (H * d) → Fin (H * d)) (u : ℝ) : Layer H d :=
  let project := ((u : ℂ)⁻¹) • liftBasis (H * d)
  ⟨⟨0, 0, 0, 0⟩, ⟨project, project, sendBasis next⟩, 1,
    ⟨0, 0⟩, ⟨(-(u : ℂ)⁻¹) • coordinates (H * d), 0⟩⟩

theorem transitionLayer_forward_at {P : Type*} [Fintype P] {H d : ℕ} (eps : ℝ) (he : 0 < eps)
    (hc : 0 < H * d) (next : Fin (H * d) → Fin (H * d)) (geometry : AttentionGeometry P d)
    (scale : ℝ) {a : ℝ} (ha : 0 < a) (word : P → Fin (H * d)) :
    (transitionLayer next (rmsAmplitude (H * d) a eps)).forwardWith eps scale geometry
      (fun i => (a : ℂ) • basis (word i)) =
        fun i => ((scale * rmsAmplitude (H * d) 1 eps : ℝ) : ℂ) • basis (next (word i)) := by
  let c := H * d
  let u := rmsAmplitude c a eps
  let x : P → CSpace c := fun i => (a : ℂ) • basis (word i)
  have huv : (u : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr (rmsAmplitude_pos_at he hc ha).ne'
  have hx : sublayer ⟨0, 0⟩ eps scale
      (attentionWith geometry (⟨0, 0, 0, 0⟩ : Attention c H d) eps 1) x = x := by
    funext i
    simp [sublayer, attentionWith, argandRMS]
  have hnormalized (i : P) : argandRMS eps (x i) = (u : ℂ) • basis (word i) :=
    argandRMS_basis_at eps (word i) ha.le
  have hproject (i : P) :
      (((u : ℂ)⁻¹) • liftBasis c) (argandRMS eps (x i)) = Pi.single (slot (word i)) 1 := by
    rw [hnormalized, LinearMap.smul_apply, map_smul, smul_smul, inv_mul_cancel₀ huv,
      one_smul, liftBasis_basis]
  have hskip (i : P) : ((-(u : ℂ)⁻¹) • coordinates c) (argandRMS eps (x i)) = -1 := by
    rw [hnormalized, LinearMap.smul_apply, map_smul, coordinates_basis, smul_eq_mul, smul_eq_mul,
      mul_one, neg_mul, inv_mul_cancel₀ huv]
  funext i
  change sublayer ⟨(-(u : ℂ)⁻¹) • coordinates c, 0⟩ eps scale
    (fun z j => sendBasis next
      ((((u : ℂ)⁻¹) • liftBasis c) (z j) * (((u : ℂ)⁻¹) • liftBasis c) (z j)))
    (sublayer ⟨0, 0⟩ eps scale
      (attentionWith geometry (⟨0, 0, 0, 0⟩ : Attention c H d) eps 1) x) i = _
  rw [hx]
  simp only [sublayer, hskip, hproject, single_mul_single_one, sendBasis_single, add_neg_cancel,
    zero_smul, LinearMap.zero_apply, add_zero, mul_one, zero_add]
  have hout := argandRMS_basis_at eps (next (word i)) (a := 1) zero_le_one
  rw [Complex.ofReal_one, one_smul] at hout
  rw [hout, smul_smul, ← Complex.ofReal_mul]

noncomputable def transitionScale : ℝ := 1 / Real.sqrt 2

private theorem transitionScale_pos : 0 < transitionScale :=
  div_pos one_pos (Real.sqrt_pos.mpr (by norm_num))

noncomputable def outputAmplitude (c : ℕ) : ℝ :=
  rmsAmplitude c (transitionScale * rmsAmplitude c 1)

private theorem outputAmplitude_pos {c : ℕ} (hc : 0 < c) :
    0 < outputAmplitude c :=
  rmsAmplitude_pos hc (mul_pos transitionScale_pos (rmsAmplitude_pos hc one_pos))

end Transition

open Transition

/-- The single-layer ordinary Argand model for a deterministic law on its
last token. The positive margin fixes the gap in the tied readout. -/
noncomputable def transitionModel {H d : ℕ} (next : Fin (H * d) → Fin (H * d))
    (margin : ℝ) : Model (Fin (H * d)) H d 1 :=
  let c := H * d
  let a := 2 * margin / outputAmplitude c
  ⟨fun i => (a : ℂ) • basis i,
    fun _ => transitionLayer next (rmsAmplitude c (rmsAmplitude c a)), 0, 1⟩

/-- Every input length and every prefix use the same compiled transition.
The exact next-state logit is the margin, and every competing logit is zero. -/
theorem transitionModel_logits {H d T : ℕ} (hc : 0 < H * d)
    (next : Fin (H * d) → Fin (H * d)) {margin : ℝ} (hm : 0 < margin)
    (word : Word (Fin (H * d)) T) (i : Fin T) (token : Fin (H * d)) :
    (transitionModel next margin).logits 1 word i token =
      if next (word i) = token then margin else 0 := by
  let c := H * d
  let a := 2 * margin / outputAmplitude c
  have ha : 0 < a := div_pos (by positivity) (outputAmplitude_pos hc)
  have hnorm : (fun j => argandRMS 1 ((a : ℂ) • basis (word j))) =
      fun j => ((rmsAmplitude c a : ℝ) : ℂ) • basis (word j) := by
    funext j
    exact argandRMS_basis_at 1 (word j) ha.le
  have hhidden : (transitionModel next margin).hidden 1 word i =
      ((outputAmplitude c : ℝ) : ℂ) • basis (next (word i)) := by
    unfold Model.hidden Model.hiddenWith transitionModel
    dsimp only
    rw [Function.iterate_one]
    simp only [List.ofFn_succ, List.ofFn_zero, List.foldl_cons, List.foldl_nil,
      Nat.cast_one, mul_one]
    change argandRMS 1 ((transitionLayer next (rmsAmplitude c (rmsAmplitude c a))).forwardWith
      1 transitionScale (causalGeometry 0) (fun j => argandRMS 1 ((a : ℂ) • basis (word j))) i) = _
    rw [hnorm, transitionLayer_forward_at 1 zero_lt_one hc next _ _ (rmsAmplitude_pos hc ha)]
    exact argandRMS_basis_at 1 _ (mul_pos transitionScale_pos (rmsAmplitude_pos hc one_pos)).le
  rw [Model.logits, hhidden, Model.readout]
  change (⟪(a : ℂ) • basis token, ((outputAmplitude c : ℝ) : ℂ) • basis (next (word i))⟫_ℂ).re / 2 = _
  rw [inner_smul_left, inner_smul_right]
  simp only [basis, EuclideanSpace.inner_single_left, map_one, one_mul,
    PiLp.single_apply, Complex.conj_ofReal]
  by_cases h : next (word i) = token
  · simp only [h, ite_true, mul_one, ← Complex.ofReal_mul, Complex.ofReal_re]
    rw [show a * outputAmplitude c = 2 * margin from
      div_mul_cancel₀ _ (outputAmplitude_pos hc).ne']
    ring
  · simp [h, Ne.symm h]

example (next : Fin 256 → Fin 256) (word : Word (Fin 256) 32) (token : Fin 256) :=
  transitionModel_logits (H := 4) (d := 64) (by decide) next (by norm_num : (0 : ℝ) < 1)
    word 31 token

end Transformer.Grokking.FiniteRecovery.Argand

module

public import Transformer.Phasor.TwoCopies.Constants
public import Transformer.Phasor.TwoCopies.Read

/-!
# Two copies: the layers of the construction

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

The construction `twoModel` is a one-stream model of main's shape. The four tokens
`x₀, y₀, x₁, y₁` sit at the positions `0, 1, 2, 3`, and everything is read at the position `3`.

* Layer `0` (`apply_layer0`). The heads read the bias and the codes; the output projection writes
  the first three tokens into their codes (`readOut_heads`), so the state holds all four operands.
  The first branch writes `ψ(w_c)`, `w_c = q_c(x_c, y_c)`, into the plane of each copy.
* Layer `1` (`apply_layer1`). The attention is zero; the second branch writes `ψ(w₀ + w₁)` into
  the output plane.
* Layers `2, 3, …` are zero and change nothing (`oneStep_attention_zero`, `oneStep_ffn_zero`).

Each residual adds a multiple of a state to a state, so after each layer the stream at the
position `3` is a state again, with the constants of `Constants`. Exact real arithmetic, every
`ε ≥ 0` and every depth scale `s`.
-/

@[expose] public section

open Finset Real
open Transformer.ComplexFFN Transformer.AttentionResiduals Transformer.LeRoPEBlock
open scoped RealInnerProductSpace

namespace Transformer.Phasor.TwoCopies

variable {D d F p : ℕ}

/-- The first layer's attention: the DFT heads, decoded at the logit scales of the first read. -/
noncomputable def attnTwo (e : Slot p → EucSpace D) (f : Fin p → EucSpace d) (eps b : ℝ) :
    Attention D d 4 :=
  attention0 e f (readOut e (d / 2) f (dftScale d eps (aRead D eps b * b)))

/-- **The construction**: the embedding `e(v)` with the bias `b`, LeRoPE's scales at the DFT
frequencies, the first layer's heads and phase branch, the second layer's product with the gain
`g`, and zero layers above. -/
noncomputable def twoModel (e : Slot p → EucSpace D) (f : Fin p → EucSpace d)
    (ι₀ : Fin 2 × Fin p ↪ Fin F) (ι₁ : Unit ↪ Fin F) (q : Fin 2 → Fin p → Fin p → Fin p)
    (eps b g : ℝ) : OneModel D d 4 F p where
  embedding v := vec e (embC b v)
  alpha := dftAlpha d
  layer l := match l with
    | 0 => ⟨attnTwo e f eps b, ffn0 e ι₀ q⟩
    | 1 => ⟨Attention.zero D d 4, ffn1 e ι₁ g⟩
    | _ + 2 => ⟨Attention.zero D d 4, FFN.zero D F⟩

/-- A residual whose branch is zero changes nothing. -/
theorem oneStep_of_eq_zero {eps s : ℝ} {raw : EucSpace D → EucSpace D} (h : ∀ z, raw z = 0)
    (x : EucSpace D) : oneStep eps s raw x = x := by
  simp [oneStep, h, rmsNormEps_eq_smul]

example (x : EucSpace 128) : oneStep (1 / 8388608) 1 (fun _ => 0) x = x :=
  oneStep_of_eq_zero (fun _ => rfl) x

/-- The zero attention changes nothing. -/
theorem oneStep_attention_zero {H : ℕ} (eps s : ℝ) (y : Fin H → EucSpace d) (x : EucSpace D) :
    oneStep eps s (fun _ => (Attention.zero D d H).output y) x = x :=
  oneStep_of_eq_zero (fun _ => rfl) x

/-- The zero feed-forward branch changes nothing. -/
theorem oneStep_ffn_zero (eps s : ℝ) (x : EucSpace D) :
    oneStep eps s (plainFFN (FFN.zero D F).gate (FFN.zero D F).value (FFN.zero D F).outRe
      (FFN.zero D F).outIm) x = x :=
  oneStep_of_eq_zero (fun _ => by simp [plainFFN, complexReadout, FFN.zero]) x

/-- Equal coefficients give equal states. -/
theorem vec_stateC_congr (e : Slot p → EucSpace D) (b : ℝ) (t : Fin 4 → Fin p)
    (w : Fin 2 → Fin p) (u : Fin p) {k k' a a' M M' : ℝ} {P P' : Fin 2 → ℝ} (hk : k = k')
    (ha : a = a') (hP : P = P') (hM : M = M') :
    vec e (stateC b k a t P w M u) = vec e (stateC b k' a' t P' w M' u) := by
  subst hk ha hP hM; rfl

example (e : Slot 29 → EucSpace 128) (t : Fin 4 → Fin 29) (w : Fin 2 → Fin 29) :=
  vec_stateC_congr e 1 t w 0 (zero_add (1 : ℝ)) rfl rfl rfl (P := 0) (M := 0) (a := 0)

/-- The tokens at the positions `0, …, 3`. -/
def toks (tokens : ℕ → Fin p) : Fin 4 → Fin p := fun j => tokens j

/-- The values `w_c = q_c(x_c, y_c)` of the two copies. -/
abbrev copyVals (q : Fin 2 → Fin p → Fin p → Fin p) (tokens : ℕ → Fin p) : Fin 2 → Fin p :=
  fun c => q c (toks tokens (xPos c)) (toks tokens (yPos c))

variable {e : Slot p → EucSpace D} {f : Fin p → EucSpace d} {ι₀ : Fin 2 × Fin p ↪ Fin F}
  {ι₁ : Unit ↪ Fin F} {q : Fin 2 → Fin p → Fin p → Fin p}

/-- **Layer `0` at the position `3`**: on the embeddings, the four operands in their codes and
`ψ(w_c)` in the plane of each copy. -/
theorem apply_layer0 (he : Orthonormal ℝ e) (hf : Orthonormal ℝ f) (hd : 4 ≤ d) (hD : 0 < D)
    {eps b : ℝ} (heps : 0 ≤ eps) (hb : 0 < b) (s : ℝ) (tokens : ℕ → Fin p) (x : ℕ → EucSpace D)
    (hx : ∀ j, x j = vec e (kEmb D eps b • embC b (tokens j))) :
    OneLayer.apply ⟨attnTwo e f eps b, ffn0 e ι₀ q⟩ eps s (dftAlpha d) x 3 =
      vec e (stateC b (kEmb D eps b) (aAttn D eps b s) (toks tokens) (ampPlanes D eps b s)
        (copyVals q tokens) 0 (copyVals q tokens 0 + copyVals q tokens 1)) := by
  have hd2 : 2 ≤ d := by omega
  have hx' (j : ℕ) : GPTMini.rmsNormEps eps (x j) = vec e (aRead D eps b • embC b (tokens j)) := by
    rw [hx, rmsNormEps_vec_embC he]; rfl
  have hheads : (attnTwo e f eps b).oneHeads eps (dftAlpha d) x 3 = fun h => attend
      (if h = 3 then fun _ => 0 else dftScale d eps (aRead D eps b * b) h •
        dftScores (d / 2) (3 - h) 3 (fun _ => ()))
      (fun j : Fin 4 => aRead D eps b • f (tokens j)) :=
    funext (oneHeads_attention0 he _ hd2 eps _ b tokens x hx')
  have hattn : (attnTwo e f eps b).output ((attnTwo e f eps b).oneHeads eps (dftAlpha d) x 3) =
      vec e (stateC b 0 (aRead D eps b) (toks tokens) 0 (copyVals q tokens) 0
        (copyVals q tokens 0 + copyVals q tokens 1)) := by
    rw [hheads]
    exact readOut_heads hf hd (dftScale_pos hd2 heps (mul_pos (aRead_pos hD heps) hb)) _ b _ _ _
  have hatt : oneStep eps s (fun _ => vec e (stateC b 0 (aRead D eps b) (toks tokens) 0
      (copyVals q tokens) 0 (copyVals q tokens 0 + copyVals q tokens 1)))
      (vec e (kEmb D eps b • embC b (toks tokens 3))) = vec e (stateC b (kEmb D eps b)
        (aAttn D eps b s) (toks tokens) 0 (copyVals q tokens) 0
          (copyVals q tokens 0 + copyVals q tokens 1)) := by
    simp only [oneStep, rmsNormEps_vec he, sum_stateC_sq, smul_embC_eq_stateC b (toks tokens)
      (copyVals q tokens) (copyVals q tokens 0 + copyVals q tokens 1), ← vec_smul, ← vec_add,
      smul_stateC, stateC_add]
    refine vec_stateC_congr e b _ _ _ ?_ ?_ ?_ ?_ <;> simp [aAttn, mul_assoc]
  have hr0 : GPTMini.rmsNormEps eps (vec e (stateC b (kEmb D eps b) (aAttn D eps b s)
      (toks tokens) 0 (copyVals q tokens) 0 (copyVals q tokens 0 + copyVals q tokens 1))) =
      vec e (stateC b (kFFN0 D eps b s * kEmb D eps b) (kFFN0 D eps b s * aAttn D eps b s)
        (toks tokens) 0 (copyVals q tokens) 0 (copyVals q tokens 0 + copyVals q tokens 1)) := by
    rw [rmsNormEps_vec he, sum_stateC_sq, smul_stateC]
    refine vec_stateC_congr e b _ _ _ ?_ ?_ ?_ ?_ <;> simp [kFFN0]
  have hffn : oneStep eps s (plainFFN (ffn0 e ι₀ q).gate (ffn0 e ι₀ q).value (ffn0 e ι₀ q).outRe
      (ffn0 e ι₀ q).outIm) (vec e (stateC b (kEmb D eps b) (aAttn D eps b s) (toks tokens) 0
        (copyVals q tokens) 0 (copyVals q tokens 0 + copyVals q tokens 1))) =
      vec e (stateC b (kEmb D eps b) (aAttn D eps b s) (toks tokens) (ampPlanes D eps b s)
        (copyVals q tokens) 0 (copyVals q tokens 0 + copyVals q tokens 1)) := by
    rw [oneStep, hr0, plainFFN_ffn0 he, rmsNormEps_vec he, sum_stateC_sq, smul_stateC,
      ← vec_smul, smul_stateC, ← vec_add, stateC_add]
    refine vec_stateC_congr e b _ _ _ ?_ ?_ ?_ ?_
    · simp
    · simp
    · funext c; simp [ampPlanes, ampFFN0, mul_assoc]
    · simp
  simp only [OneLayer.apply]
  rw [hattn, hx 3, show tokens 3 = toks tokens 3 from rfl, hatt]
  exact hffn

/-- At main's widths, `D = 128`, `d = 32` and `F = 171`, with real-linear projections, `ε = 2⁻²³`,
depth scale `1/√8` and `p = 29`. -/
example (ι₀ : Fin 2 × Fin 29 ↪ Fin 171) (q : Fin 2 → Fin 29 → Fin 29 → Fin 29)
    (tokens : ℕ → Fin 29) :=
  apply_layer0 (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128))
    (orthonormal_coordCodes (by norm_num : 29 ≤ 32)) (ι₀ := ι₀) (q := q) (by norm_num)
    (by norm_num) (by norm_num : (0 : ℝ) ≤ 1 / 8388608) one_pos (1 / Real.sqrt 8) tokens _
    (fun _ => rfl)

/-- **Layer `1` at the position `3`**: the product of the two planes, `ψ(w₀ + w₁)`, on the output
plane with the amplitude `ampOut`. -/
theorem apply_layer1 (he : Orthonormal ℝ e) {eps b : ℝ} (s g : ℝ) (tokens : ℕ → Fin p)
    (x : ℕ → EucSpace D) (hx : x 3 = vec e (stateC b (kEmb D eps b) (aAttn D eps b s) (toks tokens)
      (ampPlanes D eps b s) (copyVals q tokens) 0 (copyVals q tokens 0 + copyVals q tokens 1))) :
    OneLayer.apply ⟨Attention.zero D d 4, ffn1 e ι₁ g⟩ eps s (dftAlpha d) x 3 =
      vec e (stateC b (kEmb D eps b) (aAttn D eps b s) (toks tokens) (ampPlanes D eps b s)
        (copyVals q tokens) (ampOut D eps b s g) (copyVals q tokens 0 + copyVals q tokens 1)) := by
  simp only [OneLayer.apply]
  rw [oneStep_attention_zero, hx, oneStep, rmsNormEps_vec he, sum_stateC_sq, smul_stateC, mul_zero,
    plainFFN_ffn1 he, rmsNormEps_vec he, sum_stateC_out_sq, smul_stateC, ← vec_smul, smul_stateC,
    ← vec_add, stateC_add]
  refine vec_stateC_congr e b _ _ _ ?_ ?_ ?_ ?_
  · simp
  · simp
  · simp
  · simp only [zero_add, ampOut, ampProd, kFFN1]; ring

/-- The same sizes, on any state of the form that layer `0` leaves. -/
example (ι₁ : Unit ↪ Fin 171) (q : Fin 2 → Fin 29 → Fin 29 → Fin 29) (tokens : ℕ → Fin 29) :=
  apply_layer1 (d := 32) (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128)) (ι₁ := ι₁)
    (q := q) (eps := 1 / 8388608) (b := 1) (1 / Real.sqrt 8) 1 tokens (fun _ => _) rfl

end Transformer.Phasor.TwoCopies

module

public import Transformer.Phasor.TwoCopies.States
public import Transformer.Phasor.TwoCopies.Pairs

/-!
# Two copies: the heads of the first layer

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

Every token's state is `a e(t)` with the same bias `a b`, so a head that reads the bias slot into a
fixed vector `u` gets the same `headScale · u` from every token after the head RMS
(`headVector_slotMap_vec`). In the attention `attention0` the heads `h = 0, 1, 2` read the bias into
the DFT query and into the key turned by `3 − h`, and the head `3` has the zero query. At the
position `3`, with LeRoPE's scales at the DFT frequencies, the head `h ≠ 3` is RS5's DFT head turned
by `3 − h` with its logits scaled by `dftScale > 0`, and the head `3` is uniform. Every head reads
the code `(3, ·)` of the token into orthonormal codes `f`, so each averages `a f(t_j)` over `j ≤ 3`
(`oneHeads_attention0`).
-/

@[expose] public section

open Finset Matrix
open Transformer.MHC.PeriRMS Transformer.LeRoPEBlock Transformer.AttentionResiduals
  Transformer.RoPE.RoFormer

namespace Transformer.Phasor.TwoCopies

variable {D d p : ℕ} {e : Slot p → EucSpace D}

/-- **A head that reads the bias into `u`**, on a state with the bias `c ⋆`: `headScale · u` after
the head RMS, in the pair layout. -/
theorem headVector_slotMap_vec (he : Orthonormal ℝ e) (eps : ℝ) (u : EucSpace d)
    (c : Slot p → ℝ) :
    headVector eps (slotMap e (.inl ()) u) (vec e c) =
      headScale d eps (c (.inl ())) u • pairLayout d u := by
  rw [headVector, slotMap_vec he, rmsNormEps_smul, pairLayout_smul]

/-- The RMS of a token's state `a e(v)`: the factor `rmsScale D ε (a² (b² + 2)) · a`. -/
theorem rmsNormEps_vec_embC (he : Orthonormal ℝ e) (eps a b : ℝ) (v : Fin p) :
    GPTMini.rmsNormEps eps (vec e (a • embC b v)) =
      vec e ((rmsScale D eps (a ^ 2 * (b ^ 2 + 2)) * a) • embC b v) := by
  rw [rmsNormEps_vec_smul he, sum_embC_sq]

/-- Main's sizes, `D = 128`, `d = 32`, `p = 29`, `ε = 2⁻²³`, on the coordinate slots. -/
example (c : Slot 29 → ℝ) :=
  headVector_slotMap_vec (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128)) (1 / 8388608)
    (dftQuery 32) c

example (a b : ℝ) (v : Fin 29) :=
  rmsNormEps_vec_embC (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128)) (1 / 8388608) a b v

/-- The attention of the first layer, with the output projection `WO`: the heads `h = 0, 1, 2`
read the bias into the DFT query and the key turned by `3 − h`, the head `3` has the zero query,
and every head reads the code `(3, ·)` of the token into the codes `f`. -/
noncomputable def attention0 (e : Slot p → EucSpace D) (f : Fin p → EucSpace d)
    (WO : (Fin 4 → EucSpace d) →ₗ[ℝ] EucSpace D) : Attention D d 4 where
  query h := if h = 3 then 0 else slotMap e (.inl ()) (dftQuery d)
  key h := slotMap e (.inl ()) (dftKey d (3 - h))
  value _ := ∑ a, slotMap e (.inr (.inl (3, a))) (f a)
  coefficient _ := 0
  output := WO

/-- The logit scale of the head `h ≠ 3` when every read has the bias `m`. -/
noncomputable def dftScale (d : ℕ) (eps m : ℝ) (h : Fin 4) : ℝ :=
  headScale d eps m (dftQuery d) * headScale d eps m (dftKey d (3 - h)) *
    (2 * ((d / 2 : ℕ) : ℝ) / d)

theorem dftScale_pos {eps m : ℝ} (hd : 2 ≤ d) (heps : 0 ≤ eps) (hm : 0 < m) (h : Fin 4) :
    0 < dftScale d eps m h := by
  have h2 : (0 : ℝ) < (d / 2 : ℕ) := Nat.cast_pos.2 (by omega)
  have hd' : (0 : ℝ) < d := Nat.cast_pos.2 (by omega)
  unfold dftScale
  exact mul_pos (mul_pos (headScale_pos (by omega) heps hm (dftQuery_ne_zero hd))
    (headScale_pos (by omega) heps hm (dftKey_ne_zero hd _))) (by positivity)

example (h : Fin 4) : 0 < dftScale 32 (1 / 8388608) 1 h :=
  dftScale_pos (by norm_num) (by norm_num) one_pos h

variable {f : Fin p → EucSpace d}

/-- Every head reads the token `v` of the state `a e(v)` as `a f(v)`. -/
theorem value_vec_embC (he : Orthonormal ℝ e) (WO : (Fin 4 → EucSpace d) →ₗ[ℝ] EucSpace D)
    (h : Fin 4) (a b : ℝ) (v : Fin p) :
    (attention0 e f WO).value h (vec e (a • embC b v)) = a • f v := by
  simp only [attention0, LinearMap.sum_apply, slotMap_vec he]
  rw [Finset.sum_eq_single v (fun c _ hc => by simp [embC, hc]) (by simp)]
  simp [embC]

/-- **The heads of the first layer at the position `3`**: when every read is
`vec e (a • embC b (t_j))`, the head `3` is uniform and the head `h ≠ 3` is the DFT head turned by
`3 − h` with its logits scaled by `dftScale`, every head averaging `a f(t_j)` over `j ≤ 3`. -/
theorem oneHeads_attention0 (he : Orthonormal ℝ e) (WO : (Fin 4 → EucSpace d) →ₗ[ℝ] EucSpace D)
    (hd : 2 ≤ d) (eps a b : ℝ) (t : ℕ → Fin p) (x : ℕ → EucSpace D)
    (hx : ∀ j, GPTMini.rmsNormEps eps (x j) = vec e (a • embC b (t j))) (h : Fin 4) :
    (attention0 e f WO).oneHeads eps (dftAlpha d) x 3 h =
      attend (if h = 3 then fun _ => 0 else
          dftScale d eps (a * b) h • dftScores (d / 2) (3 - h) 3 (fun _ => ()))
        (fun j : Fin 4 => a • f (t j)) := by
  have h2 : (0 : ℝ) < (d / 2 : ℕ) := Nat.cast_pos.2 (by omega)
  have hd' : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  change attend (fun j : Fin 4 => (attention0 e f WO).oneLogit eps (dftAlpha d) h 3 (x 3) (x j)
      (3 - j)) (fun j => (attention0 e f WO).value h (GPTMini.rmsNormEps eps (x j))) = _
  congr 1
  · funext j
    simp only [Attention.oneLogit, leropeFrequencies_dftAlpha, leropeLogit, phaseLogit, attention0,
      phaseCoefficient, zero_div, zero_mul, add_zero, hx]
    split_ifs with h3
    · simp [headVector, pairLayout_zero, rmsNormEps_eq_smul]
    · rw [headVector_slotMap_vec he, headVector_slotMap_vec he]
      simp only [Pi.smul_apply, smul_eq_mul, dftScores, leropeLogit, phaseLogit, zero_mul,
        zero_div, add_zero, mulVec_smul, smul_dotProduct, dotProduct_smul, embC, dftScale,
        pairLayout_pairUnlayout, dftQuery, dftKey]
      field_simp
  · funext j
    rw [hx, value_vec_embC he]

/-- Main's sizes on the coordinate slots and codes, with the embeddings of bias `1` read by the
heads: the heads of the first layer at the position `3`. -/
example (t : ℕ → Fin 29) (WO : (Fin 4 → EucSpace 32) →ₗ[ℝ] EucSpace 128) (h : Fin 4) :=
  oneHeads_attention0 (f := coordCodes (by norm_num : 29 ≤ 32))
    (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128)) WO (by norm_num) (1 / 8388608) _ 1 t
    (fun j => vec (coordSlots _) ((1 : ℝ) • embC 1 (t j)))
    (fun j => rmsNormEps_vec_embC (orthonormal_coordSlots _) _ 1 1 (t j)) h

/-- The values of the heads at main's sizes. -/
example (WO : (Fin 4 → EucSpace 32) →ₗ[ℝ] EucSpace 128) (h : Fin 4) (a b : ℝ) (v : Fin 29) :=
  value_vec_embC (f := coordCodes (by norm_num : 29 ≤ 32))
    (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128)) WO h a b v

end Transformer.Phasor.TwoCopies

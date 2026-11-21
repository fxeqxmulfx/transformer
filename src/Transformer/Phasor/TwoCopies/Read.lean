module

public import Transformer.Phasor.TwoCopies.Heads

/-!
# Two copies: the output projection writes the operands into their slots

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

At the position `3` the heads of the first layer average the codes `a f(t_j)` of the four tokens:
the head `h ≠ 3` with RS5's weights `P_c / (P_c + 3)` on the key `h` and `1 / (P_c + 3)` on the
others, the head `3` uniformly. RS5's decoder subtracts `(4 / (P_c + 3))` times the uniform head
and divides by `p_c − q_c`, which leaves `a f(t_h)` (`dft_read_uniform_smul`), and the inner product
with the code `f c` reads `a` if `c = t_h` and `0` otherwise. The output projection `readOut`
writes these readings into the codes `(h, c)`, `h < 3`, so it turns the heads into the state with
the coefficient `a` on the code of each of the first three tokens and nothing else
(`readOut_heads`). The fourth token is the query's own and already sits in its code `(3, t_3)`.

The decoder is exact at every logit scale `c > 0`, so at every `ε`. Its coefficients change with
the query position (RS5): the projection decodes at the position `3` only, which is all the
construction reads.
-/

@[expose] public section

open Finset
open Transformer.LeRoPEBlock Transformer.AttentionResiduals
open scoped RealInnerProductSpace

namespace Transformer.Phasor.TwoCopies

variable {D d p : ℕ}

/-- `vec e` as a linear map of the coefficients. -/
noncomputable def vecL {ι : Type*} [Fintype ι] (e : ι → EucSpace D) :
    (ι → ℝ) →ₗ[ℝ] EucSpace D where
  toFun := vec e
  map_add' := vec_add e
  map_smul' := vec_smul e

/-- RS5's decoder of the head `j` against the uniform head `3` at the position `3`, for the logit
scale `c` on `2R + 1` points: `(p_c − q_c)⁻¹ (o_j − (4 / (P_c + 3)) o_3)`. -/
noncomputable def headDecode (R : ℕ) (c : ℝ) (j : Fin 4) :
    (Fin 4 → EucSpace d) →ₗ[ℝ] EucSpace d :=
  (dftGapAt c R 3)⁻¹ • (LinearMap.proj j -
    ((((3 : ℕ) : ℝ) + 1) / (dftPowAt c R 3 + ((3 : ℕ) : ℝ))) • LinearMap.proj 3)

/-- The coefficients that the output projection writes: on the code `(j, c)`, `j < 3`, the
reading of the code `f c` from the decoded head `j`; nothing elsewhere. -/
noncomputable def readCoef (R : ℕ) (f : Fin p → EucSpace d) (cs : Fin 4 → ℝ) :
    Slot p → (Fin 4 → EucSpace d) →ₗ[ℝ] ℝ
  | .inl _ => 0
  | .inr (.inl (j, c)) => if j = 3 then 0 else innerₛₗ ℝ (f c) ∘ₗ headDecode R (cs j) j
  | .inr (.inr _) => 0

/-- The output projection of the first layer, for the logit scales `cs` of the heads. -/
noncomputable def readOut (e : Slot p → EucSpace D) (R : ℕ) (f : Fin p → EucSpace d)
    (cs : Fin 4 → ℝ) : (Fin 4 → EucSpace d) →ₗ[ℝ] EucSpace D :=
  vecL e ∘ₗ LinearMap.pi (readCoef R f cs)

variable {e : Slot p → EucSpace D} {f : Fin p → EucSpace d}

/-- **The output projection writes the operands into their slots**: on the heads of the first
layer with the logit scales `cs > 0`, `readOut` gives the state with the coefficient `a` on the
code `(j, t_j)` of every `j < 3`, and nothing else. -/
theorem readOut_heads (hf : Orthonormal ℝ f) (hd : 4 ≤ d) {cs : Fin 4 → ℝ} (hcs : ∀ h, 0 < cs h)
    (a b : ℝ) (t : Fin 4 → Fin p) (w : Fin 2 → Fin p) (u : Fin p) :
    readOut e (d / 2) f cs (fun h => attend (if h = 3 then fun _ => 0 else
        cs h • dftScores (d / 2) (3 - h) 3 (fun _ => ())) (fun j : Fin 4 => a • f (t j))) =
      vec e (stateC b 0 a t 0 w 0 u) := by
  have hR : 1 ≤ d / 2 := by omega
  have hi : 3 ≤ 2 * (d / 2) := by omega
  have hread (j : Fin 4) (hj : j ≠ 3) : headDecode (d / 2) (cs j) j (fun h => attend
      (if h = 3 then fun _ => 0 else cs h • dftScores (d / 2) (3 - h) 3 (fun _ => ()))
      (fun j : Fin 4 => a • f (t j))) = a • f (t j) := by
    have h := dft_read_uniform_smul (X := Unit) hR (by norm_num) hi (fun _ => ()) (hcs j)
      (fun j : Fin 4 => a • f (t j)) j
    simpa [headDecode, hj] using h
  change vec e _ = vec e _
  congr 1
  funext s
  rcases s with _ | ⟨j, c⟩ | ⟨c, r⟩ | r
  · simp [readCoef, stateC]
  · by_cases hj : j = 3
    · subst hj
      simp [readCoef, stateC]
    · simp only [LinearMap.pi_apply, readCoef, hj, ite_false, LinearMap.comp_apply, hread j hj,
        innerₛₗ_apply_apply, real_inner_smul_right, orthonormal_iff_ite.1 hf c (t j), stateC]
      split_ifs <;> simp
  · simp [readCoef, stateC]
  · simp [readCoef, stateC]

/-- Main's head width `d = 32` with the coordinate codes of `p = 29`, at the logit scales `1`. -/
example (a b : ℝ) (t : Fin 4 → Fin 29) (w : Fin 2 → Fin 29) (u : Fin 29) :=
  readOut_heads (e := coordSlots (by norm_num : 4 * 29 + 7 ≤ 128))
    (orthonormal_coordCodes (by norm_num : 29 ≤ 32)) (by norm_num) (cs := fun _ => 1)
    (fun _ => one_pos) a b t w u

end Transformer.Phasor.TwoCopies

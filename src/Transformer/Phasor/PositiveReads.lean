module

public import Transformer.Phasor.StreamSum

/-!
# Without phases nothing is hidden from a read

T3d of `review/residual_connections.md`, *What the extra streams can hold*. Call four streams a
*positive combination* of contents `v_k` when every stream is `∑_k a_sk v_k` with all `a_sk > 0`
(`IsPosComb`). Four equal streams are one, of the single content they hold (`isPosComb_const`),
as the embedding state is. A block of mHC with zero phases keeps this, with one content more:
its finite Sinkhorn map has positive entries and its write gates are positive, so the streams after
it are a positive combination of the old contents and of the branch output
(`isPosComb_phaseMHCStep`). Every read of a positive combination is `∑_k c_k v_k` with all
`c_k > 0` (`IsPosComb.exists_read`).

The step holds at any logits, so it holds along every forward pass, where the router computes them
from the state. By induction over the sublayers (done by hand; the induction over the model is not
formalized here), with zero phases every read of the model is a combination of the embedding and
of all earlier branch outputs at that position, each with a positive coefficient. No content is
ever exactly out of sight of a read. In the ordinary residual every coefficient is `1` or `α`;
without phases mHC only changes the weights. Extreme logits can make a weight small, so
approximate hiding is not excluded, only exact hiding.
-/

@[expose] public section

open Finset

namespace Transformer.Phasor

open Transformer Transformer.MHC Transformer.MHC.PeriRMS Transformer.Residual

variable {ι κ : Type*} [Fintype ι] [Fintype κ] {E : Type*} [NormedAddCommGroup E]
  [NormedSpace ℝ E]

/-- The streams `x` are a combination of the contents `v` with positive coefficients only. -/
def IsPosComb (v : κ → E) (x : ι → E) : Prop :=
  ∃ a : ι → κ → ℝ, (∀ s k, 0 < a s k) ∧ ∀ s, x s = ∑ k, a s k • v k

omit [Fintype ι] in
/-- Equal streams are a positive combination of the one content they hold. -/
theorem isPosComb_const (h : E) : IsPosComb (ι := ι) (fun _ : Unit => h) (fun _ => h) :=
  ⟨fun _ _ => 1, fun _ _ => one_pos, fun _ => by simp⟩

/-- A mix with positive entries keeps a positive combination. -/
theorem IsPosComb.mix [Nonempty ι] {v : κ → E} {x : ι → E} (hx : IsPosComb v x)
    {H : Matrix ι ι ℝ} (hH : ∀ s t, 0 < H s t) : IsPosComb v (mix H x) := by
  obtain ⟨a, ha, hxa⟩ := hx
  refine ⟨fun s k => ∑ t, H s t * a t k,
    fun s k => Finset.sum_pos (fun t _ => mul_pos (hH s t) (ha t k)) Finset.univ_nonempty,
    fun s => ?_⟩
  simp only [Residual.mix, weightedSum, hxa, Finset.smul_sum, smul_smul, Finset.sum_smul]
  exact Finset.sum_comm

/-- Main's initial skip map keeps four equal streams a positive combination. -/
example (h : EucSpace 128) :=
  (isPosComb_const (ι := Fin 4) h).mix (H := finiteSinkhorn (2 • 1) 20)
    (sinkhornIterate_pos _ (fun _ _ => Real.exp_pos _) 20)

/-- **A block with zero phases keeps a positive combination**, with its branch output as one
content more (`none`): the finite Sinkhorn map has positive entries, the write gates are
positive. -/
theorem isPosComb_phaseMHCStep [Nonempty ι] (P : PhaseAction E) (readLogits writeLogits : ι → ℝ)
    (skipLogits : Matrix ι ι ℝ) (F : E → E) {v : κ → E} {x : ι → E} (hx : IsPosComb v x) :
    IsPosComb (Option.elim' (F (weightedSum (fun j => gate (readLogits j)) x)) v)
      (phaseMHCStep P readLogits writeLogits skipLogits (fun _ => 0) F x) := by
  obtain ⟨b, hb, hxb⟩ := hx.mix (H := finiteSinkhorn skipLogits 20)
    (sinkhornIterate_pos _ (fun _ _ => Real.exp_pos _) 20)
  refine ⟨fun s o => o.elim (writeGate (writeLogits s)) (b s), fun s o => ?_, fun s => ?_⟩
  · cases o with
    | none => exact (writeGate_bounds _).1
    | some k => exact hb s k
  · rw [Fintype.sum_option]
    simp only [phaseMHCStep, phaseMix_zero, hxb, Option.elim_none, Option.elim_some,
      Option.elim'_none, Option.elim'_some]
    rw [add_comm]

/-- Main's block at initialization, at its width, on four equal streams and any branch. -/
example (F : EucSpace 128 → EucSpace 128) (h : EucSpace 128) :=
  isPosComb_phaseMHCStep (headPhaseAction 128) 0 0 (2 • 1) F (isPosComb_const (ι := Fin 4) h)

/-- **Every read of a positive combination sees every content**, with a positive coefficient. -/
theorem IsPosComb.exists_read [Nonempty ι] {v : κ → E} {x : ι → E} (hx : IsPosComb v x)
    (readLogits : ι → ℝ) : ∃ c : κ → ℝ, (∀ k, 0 < c k) ∧
      weightedSum (fun j => gate (readLogits j)) x = ∑ k, c k • v k := by
  obtain ⟨a, ha, hxa⟩ := hx
  refine ⟨fun k => ∑ j, gate (readLogits j) * a j k,
    fun k => Finset.sum_pos (fun j _ => mul_pos (gate_pos _) (ha j k)) Finset.univ_nonempty, ?_⟩
  simp only [weightedSum, hxa, Finset.smul_sum, smul_smul, Finset.sum_smul]
  exact Finset.sum_comm

/-- After main's initial block, any read of the streams sees both the input and the branch. -/
example (F : EucSpace 128 → EucSpace 128) (h : EucSpace 128) (readLogits : Fin 4 → ℝ) :=
  (isPosComb_phaseMHCStep (headPhaseAction 128) 0 0 (2 • 1) F
    (isPosComb_const (ι := Fin 4) h)).exists_read readLogits

end Transformer.Phasor

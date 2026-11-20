module

public import Transformer.LeRoPEBlock.PiHead
public import Mathlib.Topology.Instances.Matrix

/-!
# The blocks with no positional encoding lie in the closure of LeRoPE's

Source: the blocks of `Transformer.LeRoPEBlock.Defs` (DiffGPT `5a37c51`). LeRoPE's frequencies are
`10000^{−2m/d} e^{α_m} > 0`: never exactly zero, so a block with no positional encoding is not
literally a LeRoPE block. It is a limit of them: let the frequencies go to `0` and every rotation
`R^{−δ θ}` of the finitely many distances `δ ≤ i` goes to the identity, and with it the scores,
the attentions and every continuous `Φ` of them.

* `continuous_leropeLogit`, `continuous_attend`, `continuous_block`: the logit, the attention and
  the block are continuous in the frequencies;
* `exists_lerope_close`: **for every block with no positional encoding** (the main model's, with
  its `Φ`, projections, `β`, `d` and values), every finite set of sequences, every query and
  every `ε > 0`, there are frequencies, all positive, such that the LeRoPE block with the same
  `Φ`, projections, `β`, `d` and values is within `ε` of it on the whole set.

`Transformer.LeRoPEBlock.TwoPi` has the exact statement, at the frequency `2π`, which turns a
natural distance by a multiple of `2π`. This module's is the reading that survives a perturbation
of the frequencies: small frequencies, as the high planes of RoPE's grid, already give a block
within `ε` of the one with none.
-/

@[expose] public section

open Matrix Filter Topology
open Transformer.AttentionResiduals Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {X E Y : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {H p : ℕ}

/-- The turned key varies continuously with the frequencies. -/
theorem continuous_rotary_mulVec (m : ℝ) (k : Fin 2 × Fin p → ℝ) :
    Continuous fun θ : Fin p → ℝ => rotary θ m *ᵥ k := by
  have h : (fun θ : Fin p → ℝ => rotary θ m *ᵥ k) = fun θ ri =>
      k ri * Real.cos (m * θ ri.2) + rotateHalf k ri * Real.sin (m * θ ri.2) :=
    funext fun θ => rotary_mulVec θ m k
  rw [h]
  refine continuous_pi fun ri => ?_
  have hθ : Continuous fun θ : Fin p → ℝ => θ ri.2 := continuous_apply _
  fun_prop

/-- The multiplication by `i` is continuous. -/
theorem continuous_rotateHalf :
    Continuous (rotateHalf : (Fin 2 × Fin p → ℝ) → Fin 2 × Fin p → ℝ) := by
  refine continuous_pi fun ri => ?_
  show Continuous fun x : Fin 2 × Fin p → ℝ => if ri.1 = 0 then -x (1, ri.2) else x (0, ri.2)
  split_ifs
  · exact (continuous_apply _).neg
  · exact continuous_apply _

/-- **The head's logit is continuous in the frequencies.** -/
theorem continuous_leropeLogit (β d : ℝ) (Q K : X → Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X)
    (δ : ℕ) : Continuous fun θ : Fin p → ℝ => leropeLogit θ β d Q K i x y δ := by
  have hk := continuous_rotary_mulVec (-(δ : ℝ)) (K y)
  have h1 : Continuous fun θ : Fin p → ℝ => Q x ⬝ᵥ (rotary θ (-(δ : ℝ)) *ᵥ K y) :=
    continuous_const.dotProduct hk
  have h2 : Continuous fun θ : Fin p → ℝ =>
      Q x ⬝ᵥ rotateHalf (rotary θ (-(δ : ℝ)) *ᵥ K y) :=
    continuous_const.dotProduct (continuous_rotateHalf.comp hk)
  unfold leropeLogit phaseLogit
  fun_prop

/-- **An attention is continuous in its scores.** -/
theorem continuous_attend {ι : Type*} [Fintype ι] [Nonempty ι] (u : ι → E) :
    Continuous fun s : ι → ℝ => attend s u := by
  simp only [attend, Transformer.Residual.weightedSum, depthWeight, depthPartition]
  refine continuous_finsetSum _ fun j _ => Continuous.smul ?_ continuous_const
  refine Continuous.div ?_ ?_ fun s => (Finset.sum_pos (fun k _ => Real.exp_pos _)
    Finset.univ_nonempty).ne'
  · exact Real.continuous_exp.comp (continuous_apply j)
  · exact continuous_finsetSum _ fun k _ => Real.continuous_exp.comp (continuous_apply k)

/-- **The block is continuous in the frequencies**, for every continuous `Φ`. -/
theorem continuous_block (Φ : X → (Fin H → E) → Y) [TopologicalSpace Y]
    (hΦ : ∀ a, Continuous (Φ a)) (β : Fin H → ℝ) (d : ℝ) (Q K : Fin H → X → Fin 2 × Fin p → ℝ)
    (v : Fin H → X → E) (x : ℕ → X) (i : ℕ) :
    Continuous fun θ : Fin p → ℝ => block Φ (fun h => leropeLogit θ (β h) d (Q h) (K h)) v x i := by
  unfold block
  refine (hΦ _).comp (continuous_pi fun h => ?_)
  refine (continuous_attend _).comp (continuous_pi fun j => ?_)
  exact continuous_leropeLogit _ _ _ _ _ _ _ _

/-- One head of width `2` and its projection, on one sequence of reals, at the query `3`. -/
example (x : ℕ → ℝ) :=
  continuous_block (H := 1) (p := 1) (fun _ a => a 0) (fun _ => continuous_apply 0)
    (fun _ => 0) 2 (fun _ _ => fun _ => 1) (fun _ _ => fun _ => 1) (fun _ y => y) x 3

/-- **LeRoPE's blocks approach every block with no positional encoding**: for the main model's
`Φ` (continuous), projections `Q h`, `K h`, coordinates `β h`, width `d` and values `v`, every
finite set `S` of sequences, every query `i` and every `ε > 0`, there are positive frequencies
such that the block with LeRoPE is within `ε` of the block with none, on every sequence of `S`. -/
theorem exists_lerope_close [NormedAddCommGroup Y] (Φ : X → (Fin H → E) → Y)
    (hΦ : ∀ a, Continuous (Φ a)) (β : Fin H → ℝ) (d : ℝ) (Q K : Fin H → X → Fin 2 × Fin p → ℝ)
    (v : Fin H → X → E) (i : ℕ) (S : Finset (ℕ → X)) {ε : ℝ} (hε : 0 < ε) :
    ∃ θ : Fin p → ℝ, (∀ l, 0 < θ l) ∧ ∀ x ∈ S,
      ‖block Φ (fun h => leropeLogit θ (β h) d (Q h) (K h)) v x i -
        block Φ (fun h _ a b _ => plainLogit (β h) d (Q h) (K h) i a b) v x i‖ < ε := by
  have hev : ∀ᶠ t in 𝓝[>] (0 : ℝ), (∀ x ∈ S,
      ‖block Φ (fun h => leropeLogit (fun _ => t) (β h) d (Q h) (K h)) v x i -
        block Φ (fun h _ a b _ => plainLogit (β h) d (Q h) (K h) i a b) v x i‖ < ε) ∧ 0 < t := by
    refine (Filter.Eventually.filter_mono nhdsWithin_le_nhds
      ((Filter.eventually_all_finset S).2 fun x _ => ?_)).and self_mem_nhdsWithin
    have hc : Continuous fun t : ℝ =>
        block Φ (fun h => leropeLogit (fun _ : Fin p => t) (β h) d (Q h) (K h)) v x i :=
      (continuous_block Φ hΦ β d Q K v x i).comp (continuous_pi fun _ => continuous_id)
    have h0 : block Φ (fun h => leropeLogit (fun _ : Fin p => (0 : ℝ)) (β h) d (Q h) (K h)) v x i =
        block Φ (fun h _ a b _ => plainLogit (β h) d (Q h) (K h) i a b) v x i := by
      simp only [block]
      congr 1
      funext h
      simp only [show (fun _ : Fin p => (0 : ℝ)) = 0 from rfl, leropeLogit_zero]
    have := (hc.tendsto 0) (Metric.ball_mem_nhds _ hε)
    simp only [h0] at this
    filter_upwards [this] with t ht
    rwa [Set.mem_preimage, Metric.mem_ball, dist_eq_norm] at ht
  obtain ⟨t, ht, ht0⟩ := hev.exists
  exact ⟨fun _ => t, fun _ => ht0, ht⟩

/-- One head of width `2` and its projection, on one sequence of reals, at the query `3`. -/
example (x : ℕ → ℝ) :=
  exists_lerope_close (H := 1) (p := 1) (fun _ a => a 0) (fun _ => continuous_apply 0)
    (fun _ => 0) 2 (fun _ _ => fun _ => 1) (fun _ _ => fun _ => 1) (fun _ y => y) 3 {x}
    (ε := 1) one_pos

end Transformer.LeRoPEBlock

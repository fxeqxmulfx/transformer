module

public import Transformer.Grokking.FiniteRecovery.ArgandCausality

/-! All visible values in a homogeneous word agree. Each split softmax
has total mass one, even though rotary phases can change its weights. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.PhasorV2
open scoped BigOperators

variable {A : Type*} {T H d layers : ℕ}

private theorem sum_maskedSoftmax {P : Type*} [Fintype P] (keep : P → Bool)
    (hne : ∃ j, keep j = true) (z : P → ℝ) : ∑ j, maskedSoftmax keep z j = 1 := by
  have hpos : 0 < ∑ k ∈ Finset.univ.filter (fun j => keep j), Real.exp (z k) := by
    obtain ⟨j, hj⟩ := hne
    exact Finset.sum_pos' (fun _ _ => (Real.exp_pos _).le)
      ⟨j, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hj⟩, Real.exp_pos _⟩
  simp only [maskedSoftmax, ← Finset.sum_filter, ← Finset.sum_div]
  exact div_self hpos.ne'

private theorem attention_const (projection : Attention (H * d) H d) (eps : ℝ)
    (multiplier : ℂ) (alpha : Fin d → ℝ) (z : CSpace (H * d)) (i : Fin T) :
    attention projection eps multiplier alpha (fun _ => z) i =
      projection.WO (fun h m => (1 + Complex.I) * projection.WV z h m) := by
  have hsum (x y : Fin T → ℝ) :
      ∑ j : Fin T, (⟨maskedSoftmax ((causalGeometry alpha).visible i) x j,
                     maskedSoftmax ((causalGeometry alpha).visible i) y j⟩ : ℂ) = 1 + Complex.I := by
    have hm : ∃ j, (causalGeometry alpha).visible i j = true := ⟨i, by simp [causalGeometry]⟩
    apply Complex.ext
    · simp only [Complex.re_sum, Complex.add_re, Complex.one_re, Complex.I_re, add_zero]
      exact sum_maskedSoftmax _ hm x
    · simp only [Complex.im_sum, Complex.add_im, Complex.one_im, Complex.I_im, zero_add]
      exact sum_maskedSoftmax _ hm y
  unfold attention attentionWith
  dsimp only
  apply congrArg projection.WO
  funext h m
  rw [← Finset.sum_mul, hsum]

private theorem layer_const (layer : Layer H d) (eps scale : ℝ) (alpha : Fin d → ℝ)
    (z : CSpace (H * d)) :
    ∃ y, layer.forward eps scale alpha (fun _ : Fin T => z) = fun _ => y := by
  unfold Layer.forward Layer.forwardWith
  dsimp only
  change ∃ y, sublayer _ _ _ _ (sublayer _ _ _ (attention _ _ _ _) (fun _ => z)) = fun _ => y
  simp only [sublayer, attention_const]
  exact ⟨_, rfl⟩

/-- The last and first predictions on a homogeneous word coincide
at every depth and repeat count, for every setting of the weights. -/
theorem Model.logits_const_eq_first (model : Model A H d layers) (repeats : ℕ)
    (a : A) (i : Fin (T + 1)) (b : A) :
    model.logits repeats (fun _ : Fin (T + 1) => a) i b =
      model.logits repeats (fun _ : Fin (T + 1) => a) 0 b := by
  let slide (z : Word (CSpace (H * d)) (T + 1)) (l : Layer H d) :=
    l.forward model.eps (1 / Real.sqrt (2 * (layers : ℝ) * repeats)) model.alpha z
  let pass := (List.ofFn model.layer).foldl slide
  have hp (ls : List (Layer H d)) (z : CSpace (H * d)) :
      ∃ y, ls.foldl slide (fun _ : Fin (T + 1) => z) = fun _ => y := by
    induction ls generalizing z with
    | nil => exact ⟨z, rfl⟩
    | cons l ls ih =>
      obtain ⟨y, hy⟩ := layer_const (T := T + 1) l model.eps
        (1 / Real.sqrt (2 * (layers : ℝ) * repeats)) model.alpha z
      rw [List.foldl_cons, show slide (fun _ => z) l = fun _ => y from hy]
      exact ih y
  let x := argandRMS model.eps (model.embedding a)
  have hr (n : ℕ) : ∃ z, (pass^[n]) (fun _ : Fin (T + 1) => x) = fun _ => z := by
    induction n with
    | zero => exact ⟨x, rfl⟩
    | succ n ih =>
      obtain ⟨z, hz⟩ := ih
      rw [Function.iterate_succ_apply', hz]
      exact hp _ z
  obtain ⟨z, hz⟩ := hr repeats
  change model.readout (argandRMS model.eps ((pass^[repeats]) (fun _ => x) i)) b =
    model.readout (argandRMS model.eps ((pass^[repeats]) (fun _ => x) 0)) b
  rw [hz]

example (model : Model Bool 4 16 2) (b : Bool) :=
  model.logits_const_eq_first 3 true (Fin.last 31) b

end Transformer.Grokking.FiniteRecovery.Argand

module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PathLoss
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Transport

/-!
# Stability with nonlinear features preserved along each path

Source: `review/convex_depth.md`, CD6. A bounded nonlinear branch and an isometric
flat-spectrum skip give additive state bounds on every executed path. Nonnegative
normalized joint weights retain that bound. The ensemble itself need not be an
isometry: correlations learn predictions, while averaging can cancel hidden states.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators

variable {J : Type} {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- A fixed nonexpansive skip and a bounded, possibly nonlinear branch have a
one-step additive bound (CD6). The branch reads the actual current hidden state. -/
theorem connectionBlocks_bound (U : ℕ → E →L[ℝ] E) (F : ℕ → J → E → E)
    (β : ℕ → ℝ) (hU : ∀ k h, ‖U k h‖ ≤ ‖h‖)
    (hF : ∀ k j h, ‖F k j h‖ ≤ β k) (k : ℕ) (j : J) (h : E) :
    ‖connectionBlocks U F k j h‖ ≤ ‖h‖ + β k := by
  exact (norm_add_le _ _).trans (add_le_add (hU k h) (hF k j h))

omit [NormedSpace ℝ E] in
/-- The state budget is proved through every nonlinear path, without replacing
any layer by evaluation on a mean state (CD6). -/
theorem pathRun_norm_le (B : ℕ → J → E → E) (β : ℕ → ℝ) (d : ℕ)
    (hB : ∀ k < d, ∀ j h, ‖B k j h‖ ≤ ‖h‖ + β k) (p : Path J d) (x : E) :
    ‖pathRun B d p x‖ ≤ ‖x‖ + ∑ k ∈ Finset.range d, β k := by
  induction d with
  | zero => simp [pathRun]
  | succ d ih =>
    have hp := ih (fun k hk => hB k (Nat.lt_trans hk (Nat.lt_succ_self d))) p.2
    calc
      _ ≤ ‖pathRun B d p.2 x‖ + β d := hB d (Nat.lt_succ_self d) p.1 _
      _ ≤ (‖x‖ + ∑ k ∈ Finset.range d, β k) + β d := add_le_add hp le_rfl
      _ = _ := by rw [Finset.sum_range_succ]; ring

variable [Fintype J]

/-- Learning joint correlations does not enlarge a common bound on the path
states (CD6). A norm-preserving claim for their average would be false. -/
theorem pathPredict_norm_le (B : ℕ → J → E → E) (β : ℕ → ℝ) (d : ℕ)
    (hB : ∀ k < d, ∀ j h, ‖B k j h‖ ≤ ‖h‖ + β k)
    (w : Path J d → ℝ) (hw : w ∈ pathDomain d) (x : E) :
    ‖pathPredict B d w x‖ ≤ ‖x‖ + ∑ k ∈ Finset.range d, β k := by
  calc
    _ ≤ ∑ p, ‖w p • pathRun B d p x‖ := norm_sum_le _ _
    _ = ∑ p, w p * ‖pathRun B d p x‖ := by
      apply Finset.sum_congr rfl
      intro p _
      rw [norm_smul, Real.norm_of_nonneg (hw.1 p)]
    _ ≤ ∑ p, w p * (‖x‖ + ∑ k ∈ Finset.range d, β k) :=
      Finset.sum_le_sum fun p _ =>
        mul_le_mul_of_nonneg_left (pathRun_norm_le B β d hB p x) (hw.1 p)
    _ = _ := by rw [← Finset.sum_mul, hw.2, one_mul]

/-- The new phase-Fourier connection accepts nonlinear branch interactions and
retains its bound while the joint path tensor trains (CD6). No double stochasticity
or finite Sinkhorn assumption is used. -/
theorem fourier_path_bound (b : ℕ → Fin 3 → ℝ)
    (F : ℕ → J → EuclideanSpace ℂ (Fin 4) → EuclideanSpace ℂ (Fin 4))
    (β : ℕ → ℝ) (hF : ∀ k j h, ‖F k j h‖ ≤ β k)
    (d : ℕ) (w : Path J d → ℝ) (hw : w ∈ pathDomain d) (x : EuclideanSpace ℂ (Fin 4)) :
    ‖pathPredict (connectionBlocks
      (fun k => transport (phaseConjugate fourier4 (phaseVector (b k)))) F) d w x‖ ≤
      ‖x‖ + ∑ k ∈ Finset.range d, β k := by
  apply pathPredict_norm_le _ β d _ w hw
  intro k hk j h
  apply connectionBlocks_bound _ F β _ hF
  intro i y
  exact (fourier_phase_transport_norm (b i) y).le

/-- Plain residual connections have the same additive path bound for bounded
nonlinear branches (CD6), with no extra streams or mixing parameters. -/
theorem residual_path_bound (F : ℕ → J → E → E)
    (β : ℕ → ℝ) (hF : ∀ k j h, ‖F k j h‖ ≤ β k)
    (d : ℕ) (w : Path J d → ℝ) (hw : w ∈ pathDomain d) (x : E) :
    ‖pathPredict (residualBlocks F) d w x‖ ≤ ‖x‖ + ∑ k ∈ Finset.range d, β k := by
  apply pathPredict_norm_le _ β d _ w hw
  intro k hk j h
  exact (norm_add_le _ _).trans (add_le_add le_rfl (hF k j h))

/-- A nonzero bounded branch witnesses the connection hypotheses. -/
example := connectionBlocks_bound (fun _ => ContinuousLinearMap.id ℝ ℝ)
  (fun _ (_ : Unit) (_ : ℝ) => (1 : ℝ)) (fun _ => 1)
  (by intro k h; exact le_rfl) (by intro k j h; norm_num) 0 () 2

example := pathRun_norm_le (fun _ (_ : Unit) (x : ℝ) => x + 1) (fun _ => 1) 2
  (by intro k hk j h; simpa using norm_add_le h (1 : ℝ))
  (choicePath (fun _ => ()) 2) 1

example := pathPredict_norm_le (fun _ (_ : Unit) (x : ℝ) => x + 1) (fun _ => 1) 2
  (by intro k hk j h; simpa using norm_add_le h (1 : ℝ))
  (pathVertex (choicePath (fun _ => ()) 2)) (pathVertex_mem _) 1

example := residual_path_bound (fun _ (_ : Unit) (_ : ℝ) => 1) (fun _ => 1)
  (by intro k j h; norm_num) 2
  (pathVertex (choicePath (fun _ => ()) 2)) (pathVertex_mem _) 1

example := fourier_path_bound (J := Unit) (fun _ _ => 0)
  (fun _ _ _ => WithLp.toLp 2 (fun _ : Fin 4 => (1 : ℂ))) (fun _ => 2)
  (by intro k j h; norm_num [EuclideanSpace.norm_eq]) 2
  (pathVertex (choicePath (fun _ => ()) 2)) (pathVertex_mem _)
  (WithLp.toLp 2 (fun _ : Fin 4 => (1 : ℂ)))

end Transformer.Grokking.FlatSpectrum.ConvexDepth

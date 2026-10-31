module

public import Transformer.Rotate90.Forcing.Basic

/-!
# What decay and averaging cannot add to the initial parameter span

Source: `review/flat_spectrum.md`, *Gradient forcing*, F3. The initial span
is a subspace of actual parameter vectors, not a rank approximation to the
task matrix. A normalized-gradient input outside it is necessary for escape,
but escape is not sufficient for learning or generalization. No eigenvalue,
positivity or infinite-size assumption enters the source invariance proof.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Forcing

open ScheduleFree

variable {I : Type*}

/-- The homogeneous source matrix preserves any common buffer subspace. -/
theorem transition_mem (p : AdamW.Parameters) (t : ℕ) (h : AdamW.History I)
    (S : Submodule ℝ (I → ℝ)) (v : Buffers I) (hy : v.1 ∈ S) (hz : v.2 ∈ S) :
    (transition p t h v).1 ∈ S ∧ (transition p t h v).2 ∈ S := by
  change (_ : ℝ) • v.1 + (_ : ℝ) • v.2 ∈ S ∧ (_ : ℝ) • v.1 + v.2 ∈ S
  exact ⟨S.add_mem (S.smul_mem _ hy) (S.smul_mem _ hz), S.add_mem (S.smul_mem _ hy) hz⟩

/-- A normalized-gradient input in the subspace also has both impulses there. -/
theorem impulse_mem (p : AdamW.Parameters) (t : ℕ) (h : AdamW.History I)
    (S : Submodule ℝ (I → ℝ)) (v : I → ℝ) (hv : v ∈ S) :
    (impulse p t h v).1 ∈ S ∧ (impulse p t h v).2 ∈ S :=
  ⟨S.smul_mem _ hv, S.smul_mem _ hv⟩

/-- F3's invariant applies to the actual optimizer step, including decay. -/
theorem step_mem (p : AdamW.Parameters) (t : ℕ) (s : AdamW.TrainingState I)
    (g : I → ℝ) (S : Submodule ℝ (I → ℝ))
    (hy : s.point ∈ S) (hz : s.base ∈ S) (hg : normalizedInput p t s g ∈ S) :
    (AdamW.trainingStep p t s g).point ∈ S ∧ (AdamW.trainingStep p t s g).base ∈ S := by
  have hm := transition_mem p t s.history S (buffers s) hy hz
  have hi := impulse_mem p t s.history S _ hg
  change (buffers (AdamW.trainingStep p t s g)).1 ∈ S ∧
    (buffers (AdamW.trainingStep p t s g)).2 ∈ S
  rw [step_split]
  exact ⟨S.add_mem hm.1 hi.1, S.add_mem hm.2 hi.2⟩

/-- Any finite source trajectory stays in a subspace containing its initial
buffers and every intervening normalized-gradient input. -/
theorem run_mem (p : AdamW.Parameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I)
    (S : Submodule ℝ (I → ℝ)) (hy : s.point ∈ S) (hz : s.base ∈ S) (n : ℕ)
    (hg : ∀ j < n, normalizedInput p (offset + j) (run p offset oracle s j)
      (oracle (offset + j) (run p offset oracle s j).point) ∈ S) :
    (run p offset oracle s n).point ∈ S ∧ (run p offset oracle s n).base ∈ S := by
  induction n with
  | zero => exact ⟨hy, hz⟩
  | succ n ih =>
    have hprevious := ih (fun j hj => hg j (by omega))
    exact step_mem p (offset + n) _ _ S hprevious.1 hprevious.2 (hg n (by omega))

/-- Source evaluation cannot leave a common query/base subspace either. -/
theorem evaluation_mem (p : AdamW.Parameters) (s : AdamW.TrainingState I)
    (S : Submodule ℝ (I → ℝ)) (hy : s.point ∈ S) (hz : s.base ∈ S) :
    AdamW.evaluation p s ∈ S := by
  change s.point + (1 - 1 / p.adam.betaOne) • (s.base - s.point) ∈ S
  exact S.add_mem hy (S.smul_mem _ (S.sub_mem hz hy))

/-- A genuine new evaluation direction requires a normalized-gradient input
outside the subspace; raw gradients alone are not the correct premise. -/
theorem escape_requires_forcing (p : AdamW.Parameters) (offset : ℕ)
    (oracle : ℕ → (I → ℝ) → (I → ℝ)) (s : AdamW.TrainingState I)
    (S : Submodule ℝ (I → ℝ)) (hy : s.point ∈ S) (hz : s.base ∈ S) (n : ℕ)
    (hout : AdamW.evaluation p (run p offset oracle s n) ∉ S) :
    ∃ j < n, normalizedInput p (offset + j) (run p offset oracle s j)
      (oracle (offset + j) (run p offset oracle s j).point) ∉ S := by
  by_contra h
  push Not at h
  have hm := run_mem p offset oracle s S hy hz n h
  exact hout (evaluation_mem p _ S hm.1 hm.2)

/-- The linear span of the two actual saved buffers. -/
def initialSpan (s : AdamW.TrainingState I) : Submodule ℝ (I → ℝ) :=
  Submodule.span ℝ {s.point, s.base}

/-- F3: arbitrary decay/rates/averaging with zero gradient tensors cannot leave
the two-vector initial span. Raw second moments need not be zero or reset. -/
theorem zero_evaluation_span (p : AdamW.Parameters) (offset : ℕ)
    (s : AdamW.TrainingState I) (n : ℕ) :
    AdamW.evaluation p (run p offset (fun _ _ => 0) s n) ∈ initialSpan s := by
  have hy : s.point ∈ initialSpan s := Submodule.subset_span (by simp)
  have hz : s.base ∈ initialSpan s := Submodule.subset_span (by simp)
  have hm := run_mem p offset (fun _ _ => 0) s (initialSpan s) hy hz n
    (by intro j hj; simpa only [normalizedInput_zero] using (initialSpan s).zero_mem)
  exact evaluation_mem p _ _ hm.1 hm.2

/-- The zero-gradient control retains and decays its saved raw second moment;
zero forcing never means that its optimizer state was reinitialized. -/
theorem zero_second (p : AdamW.Parameters) (offset : ℕ)
    (s : AdamW.TrainingState I) (n : ℕ) (i : I) :
    (run p offset (fun _ _ => 0) s n).history.second i =
      p.adam.betaTwo ^ n * s.history.second i := by
  induction n with
  | zero => simp [run]
  | succ n ih =>
    change p.adam.betaTwo * (run p offset (fun _ _ => 0) s n).history.second i +
      (1 - p.adam.betaTwo) * (0 : ℝ) ^ 2 = _
    rw [ih, pow_succ]
    ring

/- Nontrivial witnesses use a proper coordinate plane and two nonzero buffers. -/
section Witness

private def plane : Submodule ℝ (Fin 3 → ℝ) where
  carrier := {v | v 2 = 0}
  zero_mem' := rfl
  add_mem' := by intro a b ha hb; change a 2 + b 2 = 0; rw [ha, hb, add_zero]
  smul_mem' := by intro a b hb; change a * b 2 = 0; rw [hb, mul_zero]

local notation "p₀" => (⟨⟨1, 0, 0, 1/2, 0, 1⟩, 0, 2⟩ : AdamW.Parameters)
private abbrev witnessState : AdamW.TrainingState (Fin 3) :=
  ⟨fun i => if i = 0 then 1 else 0, fun i => if i = 1 then 1 else 0, ⟨fun _ => 1, 1, 1⟩⟩
private abbrev witnessGradient : Fin 3 → ℝ := fun i => if i = 2 then 1 else 0
local notation "s₀" => witnessState
local notation "g₀" => (fun (_ : ℕ) (_ : Fin 3 → ℝ) => witnessGradient)

example := transition_mem p₀ 0 (s₀).history plane (buffers s₀)
  (by norm_num [plane, buffers]) (by norm_num [plane, buffers])
example := impulse_mem p₀ 0 (s₀).history plane (s₀).point (by norm_num [plane])
example := step_mem p₀ 0 s₀ 0 plane (by norm_num [plane]) (by norm_num [plane])
  (by simpa only [normalizedInput_zero] using plane.zero_mem)
example := run_mem p₀ 0 (fun _ _ => 0) s₀ plane (by norm_num [plane]) (by norm_num [plane]) 10
  (by intro j hj; simpa only [normalizedInput_zero] using plane.zero_mem)
example := evaluation_mem p₀ s₀ plane (by norm_num [plane]) (by norm_num [plane])

/-- A positive-rate source update actually escapes the proper plane. -/
example : ∃ j < 1, normalizedInput p₀ j (run p₀ 0 g₀ s₀ j)
    (g₀ j (run p₀ 0 g₀ s₀ j).point) ∉ plane := by
  have hout : AdamW.evaluation p₀ (run p₀ 0 g₀ s₀ 1) ∉ plane := by
    change AdamW.evaluation p₀ (AdamW.trainingStep p₀ 0 s₀ (g₀ 0 (s₀).point)) 2 ≠ 0
    norm_num [AdamW.evaluation, AdamW.trainingStep, AdamW.direction, AdamW.denominator,
      AdamW.nextSecond, AdamW.correction, AdamW.coefficient, AdamW.weight,
      AdamW.nextMaximum, AdamW.rate, warmupFactor, Real.rpow_two, witnessGradient]
  simpa only [Nat.zero_add] using
    (escape_requires_forcing p₀ 0 g₀ s₀ plane (by norm_num [plane]) (by norm_num [plane]) 1 hout)

end Witness

end Transformer.Rotate90.Forcing

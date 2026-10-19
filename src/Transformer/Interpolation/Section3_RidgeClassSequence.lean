module

public import Transformer.Interpolation.Section3_RidgeClassProgress

/-!
# Finite actual ridge sequences make original means non-collinear

ArXiv:2411.04551v3, §3.2, preprocessing for `prop: separation`.
Compose the actual globally well-posed single-ridge probability maps.
Every map is bijective; reversing the phase order and clock gives its
actual inverse. No transport or weak-solution assumption is introduced.

For an injective finite family of positive-quadrant laws, each unfinished
step strictly lowers the class deficit while preserving the source data.
Induction on this natural number constructs at most the original number
of indices many phases. Their final supports remain in the quadrant and
their original barycenters are pairwise non-collinear. The empty sequence
is used only when no refinement is necessary.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Composition of the actual ridge probability maps in phase order.
Source: arXiv:2411.04551v3, §3.2, successive perceptron phases. -/
noncomputable def ridgeSequenceLaw (L : List (EucSpace d × EucSpace d × ℝ)) (τ : ℝ)
    (ν : Perspective.ProbSphere d) : Perspective.ProbSphere d :=
  L.foldl (fun ρ phase => ridgeLaw phase.1 phase.2.1 phase.2.2 ρ τ) ν

/-- The empty actual sequence is the identity.
Source: arXiv:2411.04551v3, §3.2, the induction base. -/
theorem ridgeSequenceLaw_nil (τ : ℝ) (ν : Perspective.ProbSphere d) :
    ridgeSequenceLaw [] τ ν = ν := rfl

/-- The first phase acts before the remaining actual probability maps.
Source: arXiv:2411.04551v3, §3.2, composing solution maps. -/
theorem ridgeSequenceLaw_cons (p u : EucSpace d) (b : ℝ)
    (L : List (EucSpace d × EucSpace d × ℝ)) (τ : ℝ) (ν : Perspective.ProbSphere d) :
    ridgeSequenceLaw ((p, u, b) :: L) τ ν = ridgeSequenceLaw L τ (ridgeLaw p u b ν τ) := rfl

/-- Appending phases composes the two actual solution maps.
Source: arXiv:2411.04551v3, §3.2, composing controls. -/
theorem ridgeSequenceLaw_append (L M : List (EucSpace d × EucSpace d × ℝ))
    (τ : ℝ) (ν : Perspective.ProbSphere d) :
    ridgeSequenceLaw (L ++ M) τ ν = ridgeSequenceLaw M τ (ridgeSequenceLaw L τ ν) := by
  exact List.foldl_append

/-- Every finite actual ridge sequence preserves distinct probabilities.
Source: arXiv:2411.04551v3, §1 and §3.2, invertible flow maps. -/
theorem bijective_ridgeSequenceLaw (L : List (EucSpace d × EucSpace d × ℝ)) (τ : ℝ) :
    Function.Bijective (ridgeSequenceLaw L τ) := by
  induction L with
  | nil => exact Function.bijective_id
  | cons phase L ih =>
    exact ih.comp (bijective_ridgeLaw phase.1 phase.2.1 phase.2.2 τ)

/-- At clock zero every actual phase fixes the original law.
Source: arXiv:2411.04551v3, §1, the original initial condition. -/
theorem ridgeSequenceLaw_zero_time (L : List (EucSpace d × EucSpace d × ℝ))
    (ν : Perspective.ProbSphere d) : ridgeSequenceLaw L 0 ν = ν := by
  induction L generalizing ν with
  | nil => rfl
  | cons phase L ih =>
    rw [ridgeSequenceLaw_cons, ridgeLaw_zero]
    exact ih ν

/-- Reversing the order and time inverts the actual finite sequence.
Source: arXiv:2411.04551v3, §3.2, flow reversal. -/
theorem ridgeSequenceLaw_reverse_inverse (L : List (EucSpace d × EucSpace d × ℝ))
    (τ : ℝ) (ν : Perspective.ProbSphere d) :
    ridgeSequenceLaw L.reverse (-τ) (ridgeSequenceLaw L τ ν) = ν := by
  induction L generalizing ν with
  | nil => rfl
  | cons phase L ih =>
    rw [List.reverse_cons, ridgeSequenceLaw_append, ridgeSequenceLaw_cons,
      ridgeSequenceLaw_cons, ridgeSequenceLaw_nil, ih, ridgeLaw_inverse]

/-- A finite actual sequence makes every normalized original mean
distinct, using at most the initial class deficit many phases. Source:
arXiv:2411.04551v3, §3.2, the perturbation stage of `prop: separation`. -/
theorem exists_ridge_class_sequence {ι : Type*} [Fintype ι]
    (i : Fin d) (ν : ι → Perspective.ProbSphere d)
    (hν : ∀ k, (ν k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective ν) (τ : ℝ) (hτ : 0 < τ) :
    ∃ L : List (EucSpace d × EucSpace d × ℝ),
      L.length ≤ Fintype.card ι - Nat.card (Set.range (fun k => barycenterDirection (ν k))) ∧
      (∀ k, (ridgeSequenceLaw L τ (ν k) : Measure (SSphere d)).support ⊆ positiveQuadrant d) ∧
      Function.Injective (fun k => ridgeSequenceLaw L τ (ν k)) ∧
      Function.Injective (fun k => barycenterDirection (ridgeSequenceLaw L τ (ν k))) := by
  classical
  generalize hdef : Fintype.card ι - Nat.card (Set.range (fun k => barycenterDirection (ν k))) = n
  induction n using Nat.strong_induction_on generalizing ν with
  | h n ih =>
    by_cases hclasses : Function.Injective (fun k => barycenterDirection (ν k))
    · exact ⟨[], Nat.zero_le _, hν, hinj, hclasses⟩
    · obtain ⟨p, u, b, -, hQ, hnewinj, hp, hs, -⟩ :=
        exists_prescribed_time_ridge_class_refinement i ν hν hinj hclasses τ hτ
      let η (k : ι) := ridgeLaw p u b (ν k) τ
      have hlt : Fintype.card ι - Nat.card (Set.range (fun k => barycenterDirection (η k))) < n := by
        have hh := refinement_defect_lt _ _ hp hs
        rwa [hdef] at hh
      obtain ⟨L, hlen, hLQ, hLinj, hLclasses⟩ := ih _ hlt η hQ hnewinj rfl
      refine ⟨(p, u, b) :: L, ?_, hLQ, hLinj, hLclasses⟩
      simp only [List.length_cons]
      omega

/-- At most the number of inputs many actual ridge phases give the
original pairwise non-collinear barycenters, retaining every quadrant
support. Source: arXiv:2411.04551v3, §3.2, `prop: separation`. -/
theorem exists_ridge_sequence_noncolinear {ι : Type*} [Fintype ι]
    (i : Fin d) (ν : ι → Perspective.ProbSphere d)
    (hν : ∀ k, (ν k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hinj : Function.Injective ν) (τ : ℝ) (hτ : 0 < τ) :
    ∃ L : List (EucSpace d × EucSpace d × ℝ), L.length ≤ Fintype.card ι ∧
      (∀ k, (ridgeSequenceLaw L τ (ν k) : Measure (SSphere d)).support ⊆ positiveQuadrant d) ∧
      Function.Injective (fun k => ridgeSequenceLaw L τ (ν k)) ∧
      ∀ k l, k ≠ l → ∀ γ : ℝ, barycenter d (ridgeSequenceLaw L τ (ν k)) ≠
        γ • barycenter d (ridgeSequenceLaw L τ (ν l)) := by
  obtain ⟨L, hlen, hQ, hnewinj, hclasses⟩ := exists_ridge_class_sequence i ν hν hinj τ hτ
  refine ⟨L, hlen.trans (Nat.sub_le _ _), hQ, hnewinj, fun k l hkl => ?_⟩
  exact (barycenterDirection_ne_iff_noncolinear i _ _ (hQ k) (hQ l)).mp
    (fun he => hkl (hclasses he))

/-- Distinct equal-mean atomic laws require a genuinely nonempty actual
sequence. The construction retains their original quadrant inputs and
gives an injective family of final barycenter directions within two phases. -/
example : ∃ (ν : Fin 2 → Perspective.ProbSphere 3)
    (L : List (EucSpace 3 × EucSpace 3 × ℝ)),
    (∀ k, (ν k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3) ∧
    Function.Injective ν ∧ (0 : ℝ) < 1 ∧ L ≠ [] ∧ L.length ≤ 2 ∧
    Function.Injective (fun k => barycenterDirection (ridgeSequenceLaw L 1 (ν k))) := by
  obtain ⟨μ, σ, hne, hμ, hσ, hbary, -⟩ := perturbation_equal_barycenter_counterexample
  let ν : Fin 2 → Perspective.ProbSphere 3 := ![μ, σ]
  have hQ (k : Fin 2) : (ν k : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 := by
    fin_cases k
    · exact hμ
    · exact hσ
  have hinj : Function.Injective ν := by
    intro k l he
    fin_cases k <;> fin_cases l
    · rfl
    · exact (hne he).elim
    · exact (hne he.symm).elim
    · rfl
  have hdir : barycenterDirection (ν 0) = barycenterDirection (ν 1) := by
    change barycenterDirection μ = barycenterDirection σ
    unfold barycenterDirection
    rw [hbary]
  obtain ⟨L, hlen, -, -, hclasses⟩ := exists_ridge_class_sequence (0 : Fin 3) ν hQ hinj 1 zero_lt_one
  refine ⟨ν, L, hQ, hinj, zero_lt_one, ?_, ?_, hclasses⟩
  · intro hL
    subst L
    have hh : (0 : Fin 2) = 1 := hclasses hdir
    exact (by decide : (0 : Fin 2) ≠ 1) hh
  · exact hlen.trans (Nat.sub_le _ _)

end Transformer.Interpolation

module

public import Transformer.Interpolation.Section3_MeanAxisPhases
public import Transformer.Interpolation.Section1_PhaseConcatenation

/-!
# Composing actual mean-attention and perceptron phases

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.2,
Steps 1–3 of `lem: induction.barycenter`. An initial rank-one attention
phase followed by a constant two-neuron phase has actual global weak
existence and uniqueness. Appending another rank-one phase gives the
exact composition formula for every original weak solution.

The law before the final switch is identified by extending it with the
old constant perceptron tail and applying proved prefix uniqueness. The
last-phase formula then applies with its actual elapsed time. The three
constant phases have an explicit three-piece description on their horizon.
-/

@[expose] public section

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original attention–perceptron prefix has global weak uniqueness.
Auxiliary for arXiv:2411.04551v3, §3.2, Steps 1 and 2. -/
theorem cauchyPB_meanAxis_twoRidge_unique (w : SSphere d) (κ S : ℝ) (hS : 0 < S)
    (i j : Fin d) (hij : i ≠ j) (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (μ ν : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (concatenate S (fun _ => meanAxisParams (w : EucSpace d) κ)
      (fun _ => twoRidgeParams i j p u b q v c)) μ)
    (hν : cauchyPB d (concatenate S (fun _ => meanAxisParams (w : EucSpace d) κ)
      (fun _ => twoRidgeParams i j p u b q v c)) ν) (h0 : μ 0 = ν 0) : μ = ν := by
  let θ : TimeParams d := concatenate S (fun _ => meanAxisParams (w : EucSpace d) κ)
    (fun _ => twoRidgeParams i j p u b q v c)
  have hpre (t : ℝ) (ht : t ≤ S) : μ t = ν t := by
    apply cauchyPB_meanAxis_prefix_unique w κ S hS θ ?_ μ ν hμ hν h0 t ht
    intro s hs
    simp only [θ, concatenate, ite_eq_left hs]
  have hmatch : μ S = ν S := hpre S le_rfl
  funext t
  by_cases ht : t ≤ S
  · exact hpre t ht
  · have hst : S ≤ t := (lt_of_not_ge ht).le
    rw [cauchyPB_twoRidge_concatenate_tail i j hij p u b q v c S _ μ hμ t hst,
      cauchyPB_twoRidge_concatenate_tail i j hij p u b q v c S _ ν hν t hst, hmatch]

/-- Every initial probability has one actual weak law for the original
attention–perceptron prefix. Source: arXiv:2411.04551v3, §3.2. -/
theorem exists_unique_cauchyPB_meanAxis_twoRidge (w : SSphere d) (κ S : ℝ) (hS : 0 < S)
    (i j : Fin d) (hij : i ≠ j) (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (ρ : Perspective.ProbSphere d) :
    ∃! μ : ℝ → Perspective.ProbSphere d, μ 0 = ρ ∧
      cauchyPB d (concatenate S (fun _ => meanAxisParams (w : EucSpace d) κ)
        (fun _ => twoRidgeParams i j p u b q v c)) μ := by
  obtain ⟨μ, h0, hμ⟩ := exists_cauchyPB_concatenate S hS
    (fun _ => meanAxisParams (w : EucSpace d) κ) (fun _ => twoRidgeParams i j p u b q v c) ρ
    (exists_cauchyPB_meanAxis w κ ρ) (fun σ => ⟨twoRidgeLaw p u b q v c σ,
      twoRidgeLaw_zero _ _ _ _ _ _ _, cauchyPB_twoRidgeLaw i j hij _ _ _ _ _ _ _⟩)
  exact ⟨μ, ⟨h0, hμ⟩, fun η hη => cauchyPB_meanAxis_twoRidge_unique w κ S hS i j hij
    p u b q v c η μ hη.2 hμ (hη.1.trans h0.symm)⟩

/-- Every original weak endpoint equals the three actual probability
maps composed with their true phase clocks. Source: arXiv:2411.04551v3, §3.2. -/
theorem cauchyPB_meanAxis_twoRidge_meanAxis_endpoint (w z : SSphere d) (κ κ' S R Q : ℝ)
    (hS : 0 < S) (hR : 0 < R) (hQ : 0 ≤ Q)
    (i j : Fin d) (hij : i ≠ j) (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (μ : ℝ → Perspective.ProbSphere d)
    (hμ : cauchyPB d (concatenate (S + R)
      (concatenate S (fun _ => meanAxisParams (w : EucSpace d) κ) (fun _ => twoRidgeParams i j p u b q v c))
      (fun _ => meanAxisParams (z : EucSpace d) κ')) μ) :
    μ (S + R + Q) = meanAxisLaw z κ'
      (twoRidgeLaw p u b q v c (meanAxisLaw w κ (μ 0) S) R) Q := by
  let θ : TimeParams d := concatenate S (fun _ => meanAxisParams (w : EucSpace d) κ)
    (fun _ => twoRidgeParams i j p u b q v c)
  have hex (ρ : Perspective.ProbSphere d) : ∃ η : ℝ → Perspective.ProbSphere d,
      η 0 = ρ ∧ cauchyPB d θ η :=
    (exists_unique_cauchyPB_meanAxis_twoRidge w κ S hS i j hij p u b q v c ρ).exists
  have hu : ∀ α β : ℝ → Perspective.ProbSphere d, cauchyPB d θ α → cauchyPB d θ β →
      α 0 = β 0 → α = β := cauchyPB_meanAxis_twoRidge_unique w κ S hS i j hij p u b q v c
  obtain ⟨η, hη0, hη⟩ := hex (μ 0)
  have htail (s : ℝ) (hs : S + R ≤ s) : θ s = twoRidgeParams i j p u b q v c := by
    simp only [θ, concatenate, ite_eq_right (show ¬s < S by linarith)]
  have hpre := cauchyPB_concatenate_prefix_of_constant_tail i j hij p u b q v c (S + R)
    (by linarith) θ (fun _ => meanAxisParams (z : EucSpace d) κ') htail hu η μ hη hμ hη0.symm (S + R) le_rfl
  have hmid := cauchyPB_twoRidge_concatenate_tail i j hij p u b q v c S _ η hη (S + R) (by linarith)
  have hηS : η S = meanAxisLaw w κ (μ 0) S := by
    rw [cauchyPB_meanAxis_prefix w κ S hS θ (by intro s hs; simp only [θ, concatenate, ite_eq_left hs]) η hη S le_rfl, hη0]
  rw [hηS, show S + R - S = R by ring] at hmid
  have hend := cauchyPB_meanAxis_concatenate_tail z κ' (S + R) θ μ hμ (S + R + Q) (by linarith)
  rw [show S + R + Q - (S + R) = Q by ring, hpre, hmid] at hend
  exact hend

/-- The three original phases have an actual global weak law for every
initial probability. Source: arXiv:2411.04551v3, §3.2, Steps 1–3. -/
theorem exists_cauchyPB_meanAxis_twoRidge_meanAxis (w z : SSphere d) (κ κ' S R : ℝ)
    (hS : 0 < S) (hR : 0 < R) (i j : Fin d) (hij : i ≠ j)
    (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (ρ : Perspective.ProbSphere d) :
    ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ρ ∧
      cauchyPB d (concatenate (S + R)
        (concatenate S (fun _ => meanAxisParams (w : EucSpace d) κ) (fun _ => twoRidgeParams i j p u b q v c))
        (fun _ => meanAxisParams (z : EucSpace d) κ')) μ := by
  exact exists_cauchyPB_concatenate (S + R) (by linarith) _ _ ρ
    (exists_unique_cauchyPB_meanAxis_twoRidge w κ S hS i j hij p u b q v c ρ).exists
    (exists_cauchyPB_meanAxis z κ')

/-- Three positive constant-control phases have exactly the required
three-piece description. Source: arXiv:2411.04551v3, §3.2, switch count. -/
theorem piecewiseConstant_three_phases (θ₁ θ₂ θ₃ : Params d) (S R Q : ℝ)
    (hS : 0 < S) (hR : 0 < R) (hQ : 0 < Q) :
    PiecewiseConstant d (concatenate (S + R) (concatenate S (fun _ => θ₁) (fun _ => θ₂))
      (fun _ => θ₃)) (S + R + Q) 3 := by
  refine ⟨![0, S, S + R, S + R + Q], ?_, by simp, by simp, ?_⟩
  · intro a b hab
    fin_cases a <;> fin_cases b <;> simp at hab ⊢ <;> linarith
  · intro k s s' hs hs'
    fin_cases k
    · change s ∈ Ico 0 S at hs
      change s' ∈ Ico 0 S at hs'
      simp only [concatenate, ite_eq_left (show s < S + R by linarith [hs.2]),
        ite_eq_left (show s' < S + R by linarith [hs'.2]), ite_eq_left hs.2, ite_eq_left hs'.2]
    · change s ∈ Ico S (S + R) at hs
      change s' ∈ Ico S (S + R) at hs'
      simp only [concatenate, ite_eq_left hs.2, ite_eq_left hs'.2,
        ite_eq_right (not_lt_of_ge hs.1), ite_eq_right (not_lt_of_ge hs'.1)]
    · change s ∈ Ico (S + R) (S + R + Q) at hs
      change s' ∈ Ico (S + R) (S + R + Q) at hs'
      simp only [concatenate, ite_eq_right (not_lt_of_ge hs.1), ite_eq_right (not_lt_of_ge hs'.1)]

/-- Actual nonzero attention and perceptron controls, positive phase
durations, and constructed original weak curves jointly realize the
existence, uniqueness and three-phase endpoint hypotheses. -/
example (ρ : Perspective.ProbSphere 2) : ∃ (w : SSphere 2)
    (μ ν η : ℝ → Perspective.ProbSphere 2),
    (0 : Fin 2) ≠ 1 ∧ (0 : ℝ) < 1 ∧
    cauchyPB 2 (concatenate 1 (fun _ => meanAxisParams (w : EucSpace 2) 1)
      (fun _ => twoRidgeParams 0 1 (w : EucSpace 2) 0 1 (EuclideanSpace.single 1 1) 0 1)) μ ∧
    cauchyPB 2 (concatenate 1 (fun _ => meanAxisParams (w : EucSpace 2) 1)
      (fun _ => twoRidgeParams 0 1 (w : EucSpace 2) 0 1 (EuclideanSpace.single 1 1) 0 1)) ν ∧
    μ 0 = ν 0 ∧
    cauchyPB 2 (concatenate 2
      (concatenate 1 (fun _ => meanAxisParams (w : EucSpace 2) 1)
        (fun _ => twoRidgeParams 0 1 (w : EucSpace 2) 0 1 (EuclideanSpace.single 1 1) 0 1))
      (fun _ => meanAxisParams (w : EucSpace 2) (-1))) η ∧ η 0 = ρ := by
  let w : SSphere 2 := basePoint 1
  obtain ⟨μ, ⟨h0, hμ⟩, _⟩ := exists_unique_cauchyPB_meanAxis_twoRidge w 1 1 zero_lt_one
    (0 : Fin 2) 1 (by norm_num) (w : EucSpace 2) 0 1 (EuclideanSpace.single 1 1) 0 1 ρ
  obtain ⟨η, hη0, hη⟩ := exists_cauchyPB_meanAxis_twoRidge_meanAxis w w 1 (-1) 1 1
    zero_lt_one zero_lt_one (0 : Fin 2) 1 (by norm_num)
    (w : EucSpace 2) 0 1 (EuclideanSpace.single 1 1) 0 1 ρ
  norm_num only at hη
  exact ⟨w, μ, μ, η, by norm_num, zero_lt_one, hμ, hμ, rfl, hη, hη0⟩

end Transformer.Interpolation

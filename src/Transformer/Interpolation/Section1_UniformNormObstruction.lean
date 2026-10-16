/-
# Obstruction to a norm estimate uniform over the original data

ArXiv:2411.04551v3, §1, `thm: targets.atoms` and its displayed
parameter norm estimate. Its proof in §4, just before the technical
appendix, explicitly permits the separation-stage implicit constant
to depend on the initial supports. Quantifying the hidden norm constant before
all input and target data is false, already at d=3, N=2, T=1 and
accuracy 1/4. This file refutes that explicit uniform interpretation;
it does not refute existence of interpolating controls with a
constant that may depend on the data.

For each proposed finite parameter bound, choose distinct sufficiently
close unit Dirac inputs with a common actual support hole. Every
original weak solution stays Dirac, and the actual complete field
bounds the squared barycenter gap by its initial gap times a finite
exponential factor. Targets at antipodes have distance two, so their
original W₂ accuracy requires a barycenter gap at least 3/2. These
conclusions contradict the chosen initial gap, with any finite number
of complete control phases. All original matrix blocks are allowed.
-/

import Transformer.Interpolation.Section1_AtomicSeparation
import Transformer.Interpolation.Section1_CloseAtomicInputs

open Real MeasureTheory Set

namespace Transformer.Interpolation

/-- Original atomic interpolation requires an initial gap determined
by the complete parameter norm and the target separation. Source
auxiliary: arXiv:2411.04551v3, §1, `thm: targets.atoms` and its norm
estimate. The bound applies to every original weak solution. -/
theorem atomic_interpolation_requires_initial_gap {d : ℕ} (θ : TimeParams d)
    (μ ν : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) (hν : cauchyPB d θ ν)
    (p q r s : SSphere d) (hμ0 : μ 0 = Perspective.diracProb d p)
    (hν0 : ν 0 = Perspective.diracProb d q) (T ε M : ℝ) (K : ℕ)
    (hθ : PiecewiseConstant d θ T K)
    (hM : ∀ t ∈ Icc (0 : ℝ) T,
      ‖(θ t).V‖ + ‖(θ t).B‖ + ‖(θ t).W‖ + ‖(θ t).U‖ + ‖(θ t).b‖ ≤ M)
    (haccμ : W2 d (μ T : Measure (SSphere d)) (Measure.dirac r) ≤ ε)
    (haccν : W2 d (ν T : Measure (SSphere d)) (Measure.dirac s) ≤ ε)
    (hsep : 2 * ε ≤ ‖(r : EucSpace d) - (s : EucSpace d)‖) :
    (‖(r : EucSpace d) - (s : EucSpace d)‖ - 2 * ε) ^ 2 ≤
      ‖(p : EucSpace d) - (q : EucSpace d)‖ ^ 2 * exp (2 * (3 * M + 5 * M ^ 2) * T) := by
  have hreq := target_distance_le_barycenter_gap_add_errors (μ T) (ν T) r s ε haccμ haccν
  have hgap : ‖(r : EucSpace d) - (s : EucSpace d)‖ - 2 * ε ≤
      ‖barycenter d (μ T) - barycenter d (ν T)‖ := by linarith
  exact ((sq_le_sq₀ (sub_nonneg.mpr hsep) (norm_nonneg _)).mpr hgap).trans
    (squared_barycenter_gap_le_exp_of_dirac θ μ ν hμ hν p q hμ0 hν0 T M K hθ hM)

/-- For every proposed uniform parameter norm, legal original atomic
data defeat that bound, even with arbitrarily many complete phases.
Source: arXiv:2411.04551v3, §1, the data-uniform interpretation of the
parameter estimate in `thm: targets.atoms`. -/
theorem exists_atomic_data_obstructing_norm (M : ℝ) :
    ∃ (μ₀ : Idx 2 → Perspective.ProbSphere 3) (xtarget : Idx 2 → SSphere 3),
      Function.Injective μ₀ ∧ Function.Injective xtarget ∧
      (∃ w : SSphere 3, ∀ i, w ∉ (μ₀ i : Measure (SSphere 3)).support) ∧
      ¬ ∃ (θ : TimeParams 3) (K : ℕ) (μ : Idx 2 → ℝ → Perspective.ProbSphere 3),
        PiecewiseConstant 3 θ 1 K ∧
        (∀ s ∈ Icc (0 : ℝ) 1,
          ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖ ≤ M) ∧
        (∀ i, μ i 0 = μ₀ i ∧ cauchyPB 3 θ (μ i)) ∧
        ∀ i, W2 3 (μ i 1 : Measure (SSphere 3)) (Measure.dirac (xtarget i)) ≤ 1 / 4 := by
  obtain ⟨X, w, _, hX, hw, hsmall⟩ :=
    exists_close_injective_atomic_data (exp (2 * (3 * M + 5 * M ^ 2)))
  let p := X 0
  let Y : Idx 2 → SSphere 3 := ![p, antipode 3 p]
  have hY : Function.Injective Y := by
    intro i j he
    fin_cases i <;> fin_cases j
    · rfl
    · exact False.elim ((antipode_ne 3 p) he.symm)
    · exact False.elim ((antipode_ne 3 p) he)
    · rfl
  refine ⟨(fun i => Perspective.diracProb 3 (X i)), Y, hX, hY, ⟨w, hw⟩, ?_⟩
  rintro ⟨θ, K, μ, hθ, hM, hμ, hacc⟩
  have hgap := squared_barycenter_gap_le_exp_of_dirac θ (μ 0) (μ 1)
    (hμ 0).2 (hμ 1).2 (X 0) (X 1) (hμ 0).1 (hμ 1).1 1 M K hθ hM
  have hgaplt : ‖barycenter 3 (μ 0 1) - barycenter 3 (μ 1 1)‖ ^ 2 < 1 := by
    exact hgap.trans_lt (by simpa only [mul_one] using hsmall)
  have htarget := target_distance_le_barycenter_gap_add_errors (μ 0 1) (μ 1 1)
    p (antipode 3 p) (1 / 4) (hacc 0) (hacc 1)
  have hdist : ‖(p : EucSpace 3) - (antipode 3 p : EucSpace 3)‖ = 2 := by
    change ‖(p : EucSpace 3) - -(p : EucSpace 3)‖ = 2
    rw [sub_neg_eq_add, ← two_smul ℝ, norm_smul, mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  rw [hdist] at htarget
  nlinarith [sq_nonneg (‖barycenter 3 (μ 0 1) - barycenter 3 (μ 1 1)‖ - 1)]

/-- The full original `targets_atoms` quantifier order, with both
constants before all dimensions and data, is false. The counterexample
uses the fixed legal regime d=3, N=2, T=1, ε=1/4, so neither the
asymptotic restriction ε<1 nor the positive-piece condition repairs it.
Source: arXiv:2411.04551v3, §1, `thm: targets.atoms`; the change being
refuted is the data-uniform interpretation of its hidden norm constant. -/
theorem not_uniform_targets_atoms_norm :
    ¬ ∃ C Cnorm : ℝ, ∀ (d N : ℕ) (μ₀ : Idx N → Perspective.ProbSphere d)
      (xtarget : Idx N → SSphere d) (T ε : ℝ),
      3 ≤ d → 1 ≤ N → 0 < T → 0 < ε → Function.Injective μ₀ →
      Function.Injective xtarget →
      (∃ w₀ : SSphere d, ∀ i, w₀ ∉ (μ₀ i : Measure (SSphere d)).support) →
      ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → Perspective.ProbSphere d),
        (K : ℝ) ≤ C * (d * N) ∧ PiecewiseConstant d θ T K ∧
        (ε < 1 → ∀ s ∈ Icc (0 : ℝ) T,
          ‖(θ s).V‖ + ‖(θ s).B‖ + ‖(θ s).W‖ + ‖(θ s).U‖ + ‖(θ s).b‖
            ≤ Cnorm * ((d * N : ℝ) / T + log (1 / ε))) ∧
        (∀ i, μ i 0 = μ₀ i ∧ cauchyPB d θ (μ i)) ∧
        ∀ i, W2 d (μ i T : Measure (SSphere d)) (Measure.dirac (xtarget i)) ≤ ε := by
  rintro ⟨C, Cnorm, h⟩
  let M := Cnorm * (6 + log 4)
  obtain ⟨μ₀, xtarget, hinj, htinj, hhole, hnot⟩ := exists_atomic_data_obstructing_norm M
  obtain ⟨θ, K, μ, _, hθ, hnorm, hμ, hacc⟩ := h 3 2 μ₀ xtarget 1 (1 / 4)
    (by decide) (by decide) (by norm_num) (by norm_num) hinj htinj hhole
  apply hnot
  refine ⟨θ, K, μ, hθ, ?_, hμ, hacc⟩
  intro s hs
  have hh := hnorm (by norm_num) s hs
  convert hh using 1
  norm_num [M]

/-- Genuine stationary atomic weak laws under nonzero complete
attention meet every quantitative lower-bound hypothesis jointly.
The actual targets are their antipodal unit atoms. -/
example : ∃ (θ : TimeParams 3) (μ ν : ℝ → Perspective.ProbSphere 3) (p q : SSphere 3),
    cauchyPB 3 θ μ ∧ cauchyPB 3 θ ν ∧
    μ 0 = Perspective.diracProb 3 p ∧ ν 0 = Perspective.diracProb 3 q ∧
    PiecewiseConstant 3 θ 1 1 ∧
    (∀ t ∈ Icc (0 : ℝ) 1,
      ‖(θ t).V‖ + ‖(θ t).B‖ + ‖(θ t).W‖ + ‖(θ t).U‖ + ‖(θ t).b‖ ≤ 2) ∧
    W2 3 (μ 1 : Measure (SSphere 3)) (Measure.dirac p) ≤ 1 / 4 ∧
    W2 3 (ν 1 : Measure (SSphere 3)) (Measure.dirac q) ≤ 1 / 4 ∧
    2 * (1 / 4 : ℝ) ≤ ‖(p : EucSpace 3) - (q : EucSpace 3)‖ := by
  let I := ContinuousLinearMap.id ℝ (EucSpace 3)
  let A : Params 3 := ⟨I, I, 0, 0, 0⟩
  let p : SSphere 3 := atomicCirclePoint 0
  let q := antipode 3 p
  have hθ : pureAttentionParams 3 1 = (fun _ => A) := by
    funext t
    simp only [pureAttentionParams, one_smul]
    rfl
  have hμ (r : SSphere 3) : cauchyPB 3 (fun _ => A) (fun _ => Perspective.diracProb 3 r) := by
    rw [← hθ]
    exact cauchyPB_pureAttention_dirac 1 r
  refine ⟨(fun _ => A), (fun _ => Perspective.diracProb 3 p),
    (fun _ => Perspective.diracProb 3 q), p, q, hμ p, hμ q, rfl, rfl,
    piecewiseConstant_constant A 1 zero_le_one, (fun _ _ => by norm_num [A, I]), ?_, ?_, ?_⟩
  · change W2 3 (Measure.dirac p) (Measure.dirac p) ≤ (1 / 4 : ℝ)
    rw [W2_self]
    norm_num
  · change W2 3 (Measure.dirac q) (Measure.dirac q) ≤ (1 / 4 : ℝ)
    rw [W2_self]
    norm_num
  · change 2 * (1 / 4 : ℝ) ≤ ‖(p : EucSpace 3) - -(p : EucSpace 3)‖
    rw [sub_neg_eq_add, ← two_smul ℝ, norm_smul, mem_sphere_zero_iff_norm.mp p.2]
    norm_num

/-- The obstruction's original hypotheses describe genuine legal
source data: two distinct actual unit Diracs, two distinct unit targets,
a common support hole, a positive horizon and positive accuracy below one. -/
example : ∃ (μ₀ : Idx 2 → Perspective.ProbSphere 3) (xtarget : Idx 2 → SSphere 3),
    3 ≤ 3 ∧ 1 ≤ 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 4 ∧ (1 / 4 : ℝ) < 1 ∧
    Function.Injective μ₀ ∧ Function.Injective xtarget ∧
    ∃ w : SSphere 3, ∀ i, w ∉ (μ₀ i : Measure (SSphere 3)).support := by
  obtain ⟨X, w, hX, hμ, hw, _⟩ := exists_close_injective_atomic_data 1
  exact ⟨(fun i => Perspective.diracProb 3 (X i)), X, le_rfl, by decide, one_pos,
    by norm_num, by norm_num, hμ, hX, w, hw⟩

end Transformer.Interpolation

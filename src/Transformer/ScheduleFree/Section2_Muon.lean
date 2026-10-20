module

public import Transformer.ScheduleFree.Section2_Wrapper
public import Transformer.Muon.Section2_Models

/-!
# Muon in the general Schedule-Free scheme

This is the explicit composition requested here: the arbitrary base
sequence of arXiv:2405.15682v4, §§2–2.1, with the practical Muon step of
arXiv:2502.16982, §§2.1–2.2. It keeps Muon's native Nesterov momentum,
five Newton–Schulz iterations, shape scaling, and decay at the base point.
The outer interpolation beta is independent of Muon's momentum mu.

Equal averaging includes the initial point, as in Section2_Wrapper.
This is a composition of two sources, not a claim that either paper
introduced this optimizer or proved its nonconvex convergence. Auxiliary
Adam blocks in the Python implementation use the same outer wrapper;
the specialization here concerns the selected matrix blocks.
-/

@[expose] public section

open scoped BigOperators Matrix

noncomputable section

namespace Transformer.ScheduleFree

variable {a b : ℕ}

/-- Matrix evaluation/base points and Muon's native momentum.
Sources: arXiv:2405.15682v4, §2; arXiv:2502.16982, §2.1. -/
abbrev MuonState (a b : ℕ) :=
  WrapperState (Matrix (Fin a) (Fin b) ℝ) (Matrix (Fin a) (Fin b) ℝ)

/-- A complete wrapper step with the already-computed query gradient.
Sources: arXiv:2405.15682v4, §2.1; arXiv:2502.16982, §§2.1–2.2. -/
def muonWrapperStep (μ η wd : ℝ) (s : MuonState a b)
    (g : Matrix (Fin a) (Fin b) ℝ) : MuonState a b :=
  wrapperStep (Muon.muonStep μ η wd) s g

/-- The oracle is evaluated at the actual mixed matrix, while Muon acts
on the base matrix. An explicit rate sequence permits optional warmup.
Sources: arXiv:2405.15682v4, §2.1; arXiv:2502.16982, §§2.1–2.2. -/
def muonWrapperRun (β μ wd : ℝ) (rate : ℕ → ℝ)
    (oracle : ℕ → Matrix (Fin a) (Fin b) ℝ → Matrix (Fin a) (Fin b) ℝ)
    (initial : Matrix (Fin a) (Fin b) ℝ) : ℕ → MuonState a b :=
  wrapperRun β (fun t => Muon.muonStep μ (rate t) wd) oracle initial 0

/-- The base matrix takes the complete native Muon direction and decays
its old base, irrespective of the outer averaging beta. Sources:
arXiv:2502.16982, §§2.1–2.2; arXiv:2405.15682v4, §2.1. -/
theorem muonWrapperStep_base (μ η wd : ℝ) (s : MuonState a b)
    (g : Matrix (Fin a) (Fin b) ℝ) :
    (muonWrapperStep μ η wd s g).base = (1 - η * wd) • s.base -
      η • Muon.adjustedUpdate (Muon.approximatePolar (Muon.nesterovInput μ s.inner g)) := by
  change Muon.decayStep η wd s.base _ = _
  exact Muon.decayStep_eq η wd s.base _

/-- Expansion of the native raw momentum, with no extra outer EMA.
Sources: arXiv:2502.16982, §2.1; arXiv:2405.15682v4, §2.1. -/
theorem muonWrapperStep_momentum (μ η wd : ℝ) (s : MuonState a b)
    (g : Matrix (Fin a) (Fin b) ℝ) :
    (muonWrapperStep μ η wd s g).inner = μ • s.inner + g ∧
      Muon.nesterovInput μ s.inner g = μ ^ 2 • s.inner + (1 + μ) • g := by
  constructor
  · change Muon.momentumStep μ s.inner g = _
    unfold Muon.momentumStep
    module
  · exact Muon.nesterovInput_eq μ s.inner g

/-- Evaluation really averages the base iterates of the query-driven
Muon trajectory, not the training queries. Sources: arXiv:2405.15682v4,
§§2–2.1, specialized to arXiv:2502.16982, §§2.1–2.2. -/
theorem muonWrapperRun_evaluation (β μ wd : ℝ) (rate : ℕ → ℝ)
    (oracle : ℕ → Matrix (Fin a) (Fin b) ℝ → Matrix (Fin a) (Fin b) ℝ)
    (initial : Matrix (Fin a) (Fin b) ℝ) (t : ℕ) :
    (muonWrapperRun β μ wd rate oracle initial t).evaluation =
      ((t : ℝ) + 1)⁻¹ • ∑ k ∈ Finset.range (t + 1),
        (muonWrapperRun β μ wd rate oracle initial k).base := by
  exact wrapperRun_evaluation β _ oracle initial 0 t

/-- With zero native momentum and gradient, the full step still applies
base-point decay, then averages with the correct prefix coefficient.
Sources: arXiv:2502.16982, §2.2; arXiv:2405.15682v4, §2.1. -/
theorem muonWrapperStep_zero_gradient (μ η wd : ℝ)
    (x z : Matrix (Fin a) (Fin b) ℝ) (k : ℕ) :
    let s : MuonState a b := ⟨x, z, 0, k⟩
    (muonWrapperStep μ η wd s 0).base = (1 - η * wd) • z ∧
      (muonWrapperStep μ η wd s 0).evaluation =
        (1 - wrapperCoefficient k) • x +
          wrapperCoefficient k • ((1 - η * wd) • z) := by
  dsimp only
  have hz : Muon.approximatePolar (Muon.nesterovInput μ (0 : Matrix (Fin a) (Fin b) ℝ) 0) = 0 := by
    simp [Muon.nesterovInput, Muon.momentumStep, Muon.approximatePolar,
      Muon.schulzIterate_zero]
  have hb := muonWrapperStep_base μ η wd (⟨x, z, 0, k⟩ : MuonState a b) 0
  rw [hz] at hb
  simp only [Muon.adjustedUpdate, smul_zero, sub_zero] at hb
  constructor
  · exact hb
  · change (1 - wrapperCoefficient k) • x +
        wrapperCoefficient k • (muonWrapperStep μ η wd (⟨x, z, 0, k⟩ : MuonState a b) 0).base = _
    rw [hb]

/-- A rational cross-language fixture distinguishes base decay, inclusion
of the initial point, and the training query in a single complete step.
Sources: the composed recurrences above; not a convergence claim. -/
theorem muonWrapper_first_decay_fixture :
    let w : Matrix (Fin 1) (Fin 1) ℝ := Matrix.of (fun _ _ => 1)
    let s := muonWrapperRun (1 / 2) 0 (1 / 2) (fun _ => 1 / 4) (fun _ _ => 0) w 1
    s.base 0 0 = 7 / 8 ∧ s.evaluation 0 0 = 15 / 16 ∧
      wrapperQuery (1 / 2) s 0 0 = 29 / 32 := by
  dsimp only
  have h := muonWrapperStep_zero_gradient (a := 1) (b := 1) 0 (1 / 4) (1 / 2)
    (Matrix.of (fun _ _ => 1)) (Matrix.of (fun _ _ => 1)) 0
  have hb := congrArg (fun m : Matrix (Fin 1) (Fin 1) ℝ => m 0 0) h.1
  have hx := congrArg (fun m : Matrix (Fin 1) (Fin 1) ℝ => m 0 0) h.2
  norm_num [wrapperCoefficient, Matrix.smul_apply, Matrix.add_apply] at hb hx
  change _ ∧ _ ∧ wrapperQuery (1 / 2) (muonWrapperStep 0 (1 / 4) (1 / 2)
    ⟨Matrix.of (fun _ _ => 1), Matrix.of (fun _ _ => 1), 0, 0⟩ 0) 0 0 = _
  exact ⟨hb, hx, by
    change (1 / 2 : ℝ) * _ + (1 - 1 / 2) * _ = _
    rw [hb, hx]
    norm_num⟩

/-- A zero native rate keeps the base but still advances momentum and
equal averaging. This differs from Algorithm 1's squared-rate averaging.
Sources: arXiv:2405.15682v4, §2.1; arXiv:2502.16982, §2.1. -/
theorem muonWrapperStep_zero_rate (μ wd : ℝ) (s : MuonState a b)
    (g : Matrix (Fin a) (Fin b) ℝ) :
    (muonWrapperStep μ 0 wd s g).base = s.base ∧
      (muonWrapperStep μ 0 wd s g).evaluation =
        (1 - wrapperCoefficient s.updates) • s.evaluation +
          wrapperCoefficient s.updates • s.base ∧
      (muonWrapperStep μ 0 wd s g).inner = μ • s.inner + g := by
  have hb := muonWrapperStep_base μ 0 wd s g
  simp only [zero_mul, sub_zero, one_smul, zero_smul] at hb
  refine ⟨hb, ?_, (muonWrapperStep_momentum μ 0 wd s g).1⟩
  change (1 - wrapperCoefficient s.updates) • s.evaluation +
      wrapperCoefficient s.updates • (muonWrapperStep μ 0 wd s g).base = _
  rw [hb]

/-- The next full step uses a coefficient of one third and continues
decaying the base, rather than the previous mixed query. Its exact
arithmetic mean includes all three base matrices. This second checkpoint
detects a wrapper that is correct only on its coincident initial buffers.
Sources: arXiv:2405.15682v4, §2.1, composed with arXiv:2502.16982, §2.2. -/
theorem muonWrapper_second_decay_fixture :
    let w : Matrix (Fin 1) (Fin 1) ℝ := Matrix.of (fun _ _ => 1)
    let s := muonWrapperRun (1 / 2) 0 (1 / 2) (fun _ => 1 / 4) (fun _ _ => 0) w 2
    s.base 0 0 = 49 / 64 ∧ s.evaluation 0 0 = 169 / 192 ∧
      wrapperQuery (1 / 2) s 0 0 = 79 / 96 := by
  norm_num [muonWrapperRun, wrapperRun, wrapperStep, wrapperCoefficient,
    Muon.muonStep, Muon.momentumStep, Muon.nesterovInput, Muon.approximatePolar,
    Muon.schulzIterate_zero, Muon.adjustedUpdate, Muon.decayStep, wrapperQuery,
    Matrix.smul_apply, Matrix.add_apply, Matrix.sub_apply]

end Transformer.ScheduleFree

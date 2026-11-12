module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Optimality
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Consistency
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation.Basic
public import Transformer.Grokking.FlatSpectrum.ConnectedRecovery

/-!+# Graph coverage turns ideal CE into complete probability recovery

Source: `review/flat_spectrum.md`, CM13, applying its existing Theorem E.
If a full separating code bank makes the target moments unit-phase outer
products, a connected observation graph containing a perfect matching and
nuclear budgets of the input dimension identify the entire target table at
ideal observed CE. No rank constraint is imposed on the learned probabilities.
The target's coded structure and graph coverage are explicit hypotheses; this
does not prove that a discovered coding represents unknown ground truth, or
replace exact loss equality with a numerical tolerance.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n] [Nonempty n]

/-- A nuclear upper bound suffices in Theorem E: every interpolating matrix
below the rank-one target's norm is that unique target (CM13). -/
theorem graph_nuclear_budget_recovers {a b : n → ℂ}
    (ha : ∀ i, ‖a i‖ = 1) (hb : ∀ j, ‖b j‖ = 1) {Ω : Set (n × n)}
    (hgraph : (patternGraph Ω).Connected)
    (hmatch : ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Ω)
    {X : Matrix n n ℂ} (hfit : ∀ ij ∈ Ω, X ij.1 ij.2 = vecMulVec a b ij.1 ij.2)
    (hB : nuclearNorm X ≤ Fintype.card n) : X = vecMulVec a b := by
  have hrec := (forall_isMinOn_vecMulVec_iff_connected ha hb).mpr ⟨hgraph, hmatch⟩
  have hk : ∀ ij ∈ Ω, vecMulVec a b ij.1 ij.2 = vecMulVec a b ij.1 ij.2 :=
    fun _ _ => rfl
  have hmin := (hrec (vecMulVec a b) hk).mpr rfl
  apply (hrec X hfit).mp
  intro Q hQ
  calc
    nuclearNorm X ≤ Fintype.card n := hB
    _ = nuclearNorm (vecMulVec a b) := (nuclearNorm_vecMulVec_of_norm_eq_one ha hb).symm
    _ ≤ nuclearNorm Q := hmin hQ

variable {C F S : Type*} [Fintype C] [Fintype F] [Fintype S]
  [DecidableEq C] [Nontrivial C]

omit [Nonempty n] [DecidableEq C] [Nontrivial C] in
/-- The constant Fourier mode automatically has nuclear norm N on probability
rows (CM13), so a native bank need only constrain its nonconstant modes. -/
theorem constant_moment_nuclear {P : Table n n C} (hP : Rows P) :
    nuclearNorm (moment (fun _ : C => 1) P) = Fintype.card n := by
  have he : moment (fun _ : C => 1) P = onesMatrix n := by
    ext i j
    simp [moment, onesMatrix, ← Complex.ofReal_sum, hP.2 i j]
  rw [he, nuclearNorm_onesMatrix]

/-- Actual ideal observed CE, graph coverage and the nuclear budgets recover
every class probability, including all unobserved cells (CM13). Target
separability concerns the given full label table, not merely its observations. -/
theorem graph_probability_recovery (φ : Matrix F C ℂ) {ρ : ℂ} (hρ : ρ ≠ 0)
    (hGram : φᴴ * φ = ρ • 1) (a b : F → n → ℂ)
    (ha : ∀ f i, ‖a f i‖ = 1) (hb : ∀ f j, ‖b f j‖ = 1)
    (labels : n → n → C) (hcode : ∀ f i j, φ f (labels i j) = a f i * b f j)
    (cell : S → n × n) (hgraph : (patternGraph (Set.range cell)).Connected)
    (hmatch : ∃ σ : Equiv.Perm n, ∀ i, (i, σ i) ∈ Set.range cell)
    {P : Table n n C} (hP : Rows P)
    (hB : ∀ f, nuclearNorm (moment (φ f) P) ≤ Fintype.card n)
    (hL : loss cell (fun s => labels (cell s).1 (cell s).2) P ≤
      idealLoss (fun s => labels (cell s).1 (cell s).2)) : P = pointMass labels := by
  apply moment_injective_of_gram φ hρ hGram
  intro f
  have htarget : moment (φ f) (pointMass labels) = vecMulVec (a f) (b f) := by
    rw [moment_pointMass]
    ext i j
    exact hcode f i j
  rw [htarget]
  apply graph_nuclear_budget_recovers (ha f) (hb f) hgraph hmatch ?_ (hB f)
  rintro ij ⟨s, rfl⟩
  have hmass := target_mass_of_loss_le cell _ hP hL s
  have hrow := ConvexChannel.Observation.simplex_eq_target _
    (hP.1 (cell s).1 (cell s).2) (hP.2 (cell s).1 (cell s).2) hmass
  calc
    moment (φ f) P (cell s).1 (cell s).2 =
        moment (φ f) (pointMass labels) (cell s).1 (cell s).2 := by
      simp only [moment, hrow, pointMass, ConvexRecall.targetLogits]
    _ = vecMulVec (a f) (b f) (cell s).1 (cell s).2 := congrFun (congrFun htarget _) _

/-! A nonconstant two-class witness with one genuinely unobserved input. -/

private def observed : Set (Fin 2 × Fin 2) := {ij | ij ≠ (1, 0)}
private noncomputable instance : Fintype observed := Fintype.ofFinite observed
private def cell : observed → Fin 2 × Fin 2 := Subtype.val
private def labels (i j : Fin 2) : Bool := decide (i = j)
private def code : Matrix Bool Bool ℂ := fun f c => if f then (if c then 1 else -1) else 1
private def phase (f : Bool) (i : Fin 2) : ℂ := if f && decide (i = 1) then -1 else 1

private theorem phase_unit (f : Bool) (i : Fin 2) : ‖phase f i‖ = 1 := by
  cases f <;> fin_cases i <;> norm_num [phase]

private theorem code_factors (f : Bool) (i j : Fin 2) :
    code f (labels i j) = phase f i * phase f j := by
  cases f <;> fin_cases i <;> fin_cases j <;> norm_num [code, labels, phase]

private theorem code_gram : codeᴴ * code = (2 : ℂ) • 1 := by
  ext i j
  cases i <;> cases j <;>
    norm_num [Matrix.mul_apply, Matrix.conjTranspose_apply, Matrix.one_apply, code]

private theorem observed_range : Set.range cell = observed := by
  ext ij
  constructor
  · rintro ⟨s, rfl⟩
    exact s.property
  · intro h
    exact ⟨⟨ij, h⟩, rfl⟩

private theorem observed_connected : (patternGraph (Set.range cell)).Connected := by
  rw [observed_range]
  exact (rigid_iff_connected observed).mp rigid_ne_one_zero

private theorem observed_matching :
    ∃ σ : Equiv.Perm (Fin 2), ∀ i, (i, σ i) ∈ Set.range cell := by
  refine ⟨Equiv.refl _, ?_⟩
  intro i
  rw [observed_range]
  fin_cases i <;> simp [observed]

private theorem witness_budget (f : Bool) :
    nuclearNorm (moment (code f) (pointMass labels)) ≤ Fintype.card (Fin 2) := by
  have he : moment (code f) (pointMass labels) = vecMulVec (phase f) (phase f) := by
    rw [moment_pointMass]
    ext i j
    exact code_factors f i j
  rw [he, nuclearNorm_vecMulVec_of_norm_eq_one (phase_unit f) (phase_unit f)]

private theorem witness_loss : loss cell (fun s => labels (cell s).1 (cell s).2)
    (pointMass labels) = idealLoss (fun s => labels (cell s).1 (cell s).2) := by
  apply loss_eq_ideal_of_fit
  intro s
  rfl

example := graph_nuclear_budget_recovers (phase_unit true) (phase_unit true)
  observed_connected observed_matching
  (X := vecMulVec (phase true) (phase true)) (fun _ _ => rfl)
  (nuclearNorm_vecMulVec_of_norm_eq_one (phase_unit true) (phase_unit true)).le

example := graph_probability_recovery code (by norm_num : (2 : ℂ) ≠ 0) code_gram
  phase phase phase_unit phase_unit labels code_factors cell
  observed_connected observed_matching (rows_pointMass labels) witness_budget witness_loss.le

example := constant_moment_nuclear (rows_pointMass labels)

example : (1, 0) ∉ Set.range cell := by rw [observed_range]; simp [observed]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

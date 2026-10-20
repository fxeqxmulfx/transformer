module

public import Transformer.Basic
public import Mathlib.Analysis.Calculus.FDeriv.CompCLM
public import Mathlib.Analysis.Calculus.FDeriv.Prod
public import Mathlib.Analysis.Calculus.FDeriv.Add

/-!
# §3 — The coupled angular and radial Jacobian

Karagodin, Polyanskiy, Rigollet, arXiv:2510.22026v2, supplement,
Appendix D, proof of Theorem 3.1, Step 4.

In local angular coordinates `x` and radial coordinates `q`, the source has
`x' = W(x,q) G(x)`, `q' = f(x)-q`, with `G` the negative energy gradient.
At `G(x)=0`, differentiating the actual field cancels the derivative of `W`
and gives the blocks `(W A,0;B,-I)`. Every positive angular eigenvalue is an
eigenvalue of that full Jacobian, irrespective of the radial coupling `B`.

These are the differential and linear-algebra implications used by Step 4.
They do not assume that the full Jacobian is symmetric and do not assert
that the basin of an unstable equilibrium has measure zero. That measure
statement and the angular instability of each nonsynchronized attention
equilibrium are separate inputs to the remaining synchronization proof.
-/

@[expose] public section

namespace Transformer.Normalization

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- The autonomous system of Appendix D, Step 4, written with angles first
and radii second. `W` is the angular modulator; `G` is the angular field. -/
def coupledField (G : E → E) (f : E → F) (W : E × F → E →L[ℝ] E) (p : E × F) : E × F :=
  (W p (G p.1), f p.1 - p.2)

/-- The block linearization `(v,u) ↦ (A v, B v-u)` in Appendix D, Step 4.
The angular scaling is already included in `A`. -/
noncomputable def coupledJacobian (A : E →L[ℝ] E) (B : E →L[ℝ] F) :
    E × F →L[ℝ] E × F :=
  (A.comp (ContinuousLinearMap.fst ℝ E F)).prod
    (B.comp (ContinuousLinearMap.fst ℝ E F) - ContinuousLinearMap.snd ℝ E F)

/-- Evaluation of the block operator used in Step 4. -/
theorem coupledJacobian_apply (A : E →L[ℝ] E) (B : E →L[ℝ] F) (v : E) (u : F) :
    coupledJacobian A B (v, u) = (A v, B v - u) := rfl

omit [NormedSpace ℝ F] in
/-- With injective angular modulation, equilibria are exactly angular
critical points with `q=f(x)`. Source: Appendix D, Steps 3–4. -/
theorem coupledField_eq_zero_iff {G : E → E} {f : E → F}
    {W : E × F → E →L[ℝ] E} {p : E × F} (hW : Function.Injective (W p)) :
    coupledField G f W p = 0 ↔ G p.1 = 0 ∧ f p.1 = p.2 := by
  constructor
  · intro h
    have hfirst := congrArg Prod.fst h
    have hsecond := congrArg Prod.snd h
    exact ⟨hW (by simpa [coupledField] using hfirst),
      sub_eq_zero.mp (by simpa [coupledField] using hsecond)⟩
  · rintro ⟨hG, hf⟩
    simp [coupledField, hG, hf]

example : Function.Injective ((fun _ : ℝ × ℝ => ContinuousLinearMap.id ℝ ℝ) (0, 1)) :=
  fun _ _ h => h

/-- Differentiating the coupled field at an angular critical point gives
`J=(W A,0;B,-I)`: the derivative of `W` multiplies `G(x)=0` and vanishes.
Source: arXiv:2510.22026v2, Appendix D, Step 4, displayed block Jacobian. -/
theorem hasFDerivAt_coupledField_of_critical {G : E → E} {f : E → F}
    {W : E × F → E →L[ℝ] E} {p : E × F} {A : E →L[ℝ] E} {B : E →L[ℝ] F}
    (hG : HasFDerivAt G A p.1) (hf : HasFDerivAt f B p.1)
    (hW : DifferentiableAt ℝ W p) (hcrit : G p.1 = 0) :
    HasFDerivAt (coupledField G f W) (coupledJacobian ((W p).comp A) B) p := by
  have hangle := hW.hasFDerivAt.clm_apply (hG.comp p hasFDerivAt_fst)
  have hradius := (hf.comp p hasFDerivAt_fst).sub hasFDerivAt_snd
  simpa [coupledField, coupledJacobian, Function.comp_def, hcrit, ContinuousLinearMap.comp_assoc]
    using! hangle.prodMk hradius

/-- A one-dimensional angular field and radial forcing realize the full
differential hypotheses at the equilibrium `(x,q)=(0,1)`. -/
example : HasFDerivAt (id : ℝ → ℝ) (ContinuousLinearMap.id ℝ ℝ) 0 ∧
    HasFDerivAt (fun x : ℝ => 1 + x) (ContinuousLinearMap.id ℝ ℝ) 0 ∧
    DifferentiableAt ℝ (fun _ : ℝ × ℝ => ContinuousLinearMap.id ℝ ℝ) (0, 1) ∧
    (id : ℝ → ℝ) 0 = 0 :=
  ⟨hasFDerivAt_id 0, (hasFDerivAt_id 0).const_add 1, differentiableAt_const _, rfl⟩

/-- An angular eigenvector lifts explicitly to the full Jacobian:
`v ↦ (v, (λ+1)⁻¹ Bv)`. Only the radial eigenvalue `-1` is excluded.
Source: arXiv:2510.22026v2, Appendix D, Step 4, block triangularity. -/
theorem coupledJacobian_lift_eigenvector (A : E →L[ℝ] E) (B : E →L[ℝ] F)
    {lam : ℝ} (hlam : lam ≠ -1) (v : E) (hv : A v = lam • v) :
    coupledJacobian A B (v, (lam + 1)⁻¹ • B v) = lam • (v, (lam + 1)⁻¹ • B v) := by
  have hlam' : lam + 1 ≠ 0 := by intro h; apply hlam; linarith
  apply Prod.ext
  · exact hv
  · change B v - (lam + 1)⁻¹ • B v = lam • ((lam + 1)⁻¹ • B v)
    calc B v - (lam + 1)⁻¹ • B v = (1 - (lam + 1)⁻¹) • B v := by rw [sub_smul, one_smul]
      _ = (lam * (lam + 1)⁻¹) • B v := by congr 1; field_simp; ring
      _ = lam • ((lam + 1)⁻¹ • B v) := (smul_smul _ _ _).symm

example : (1 : ℝ) ≠ -1 ∧ (ContinuousLinearMap.id ℝ ℝ) 1 = (1 : ℝ) • 1 := by norm_num

/-- Apart from the radial eigenvalue `-1`, the real eigenvalues of the
coupled Jacobian are exactly the angular ones. If an eigenvector had zero
angular component, its radial component would have eigenvalue `-1`.
Source: arXiv:2510.22026v2, Appendix D, Step 4, block triangularity. -/
theorem coupledJacobian_eigen_iff (A : E →L[ℝ] E) (B : E →L[ℝ] F)
    {lam : ℝ} (hlam : lam ≠ -1) :
    (∃ z : E × F, z ≠ 0 ∧ coupledJacobian A B z = lam • z) ↔
      ∃ v : E, v ≠ 0 ∧ A v = lam • v := by
  constructor
  · rintro ⟨z, hz, heig⟩
    have hangle : A z.1 = lam • z.1 := congrArg Prod.fst heig
    refine ⟨z.1, ?_, hangle⟩
    intro hzero
    have hradial : -z.2 = lam • z.2 := by
      simpa [coupledJacobian, hzero] using congrArg Prod.snd heig
    have hmul : (lam + 1) • z.2 = 0 := by rw [add_smul, one_smul, ← hradial]; simp
    have hu : z.2 = 0 := (smul_eq_zero.mp hmul).resolve_left (by intro h; apply hlam; linarith)
    exact hz (Prod.ext hzero hu)
  · rintro ⟨v, hv, heig⟩
    refine ⟨(v, (lam + 1)⁻¹ • B v), ?_, coupledJacobian_lift_eigenvector A B hlam v heig⟩
    intro hz
    exact hv (congrArg Prod.fst hz)

example : (1 : ℝ) ≠ -1 := by norm_num

/-- A positive angular eigenvalue remains a positive real eigenvalue of
the full coupled Jacobian, with a nonzero lifted eigenvector.
Source: arXiv:2510.22026v2, Appendix D, Step 4. -/
theorem coupledJacobian_unstable (A : E →L[ℝ] E) (B : E →L[ℝ] F)
    (hA : ∃ lam : ℝ, 0 < lam ∧ ∃ v : E, v ≠ 0 ∧ A v = lam • v) :
    ∃ lam : ℝ, 0 < lam ∧ ∃ z : E × F, z ≠ 0 ∧ coupledJacobian A B z = lam • z := by
  obtain ⟨lam, hlam, v, hv, hAv⟩ := hA
  refine ⟨lam, hlam, (v, (lam + 1)⁻¹ • B v), ?_,
    coupledJacobian_lift_eigenvector A B (by linarith) v hAv⟩
  intro hz
  exact hv (congrArg Prod.fst hz)

example : ∃ lam : ℝ, 0 < lam ∧ ∃ v : ℝ, v ≠ 0 ∧ (ContinuousLinearMap.id ℝ ℝ) v = lam • v :=
  ⟨1, one_pos, 1, one_ne_zero, by simp⟩

/-- The derivative of the actual coupled field is unstable whenever its
angular linearization has a positive real eigenvalue. This combines the
computed derivative with the explicit lift, without postulating a Jacobian.
Source: arXiv:2510.22026v2, Appendix D, Step 4. -/
theorem fderiv_coupledField_unstable {G : E → E} {f : E → F}
    {W : E × F → E →L[ℝ] E} {p : E × F} {A : E →L[ℝ] E} {B : E →L[ℝ] F}
    (hG : HasFDerivAt G A p.1) (hf : HasFDerivAt f B p.1)
    (hW : DifferentiableAt ℝ W p) (hcrit : G p.1 = 0)
    (hA : ∃ lam : ℝ, 0 < lam ∧ ∃ v : E, v ≠ 0 ∧ W p (A v) = lam • v) :
    ∃ lam : ℝ, 0 < lam ∧ ∃ z : E × F, z ≠ 0 ∧
      fderiv ℝ (coupledField G f W) p z = lam • z := by
  rw [(hasFDerivAt_coupledField_of_critical hG hf hW hcrit).fderiv]
  exact coupledJacobian_unstable ((W p).comp A) B hA

/-- The field `(x,1+x-q)` has an unstable equilibrium at `(0,1)` and
realizes every hypothesis of the derivative-instability theorem. -/
example : ∃ lam : ℝ, 0 < lam ∧ ∃ z : ℝ × ℝ, z ≠ 0 ∧
    fderiv ℝ (coupledField (id : ℝ → ℝ) (fun x => 1 + x)
      (fun _ => ContinuousLinearMap.id ℝ ℝ)) (0, 1) z = lam • z := by
  exact fderiv_coupledField_unstable (hasFDerivAt_id 0) ((hasFDerivAt_id 0).const_add 1)
    (differentiableAt_const _) rfl ⟨1, one_pos, 1, one_ne_zero, by simp⟩

end Transformer.Normalization

module

public import Transformer.PhasorV2.PeriNorm
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan

/-!
# Phasor v2: the phase-conjugated skip of mHC

The fifth element of Phasor v2 is Phasor's residual: the four streams of mHC (Xie et al.,
arXiv:2512.24880v2, §4.1–4.2), mixed with phases. Source: DiffGPT (`~/python_projects/gpt`,
`fe1d528`), `research/main_model/README.txt`, *Architecture*:

    h'_{i,s} = ∑_t P_{i,s,t} r_s conj(r_t) h_{i,t} + write_{i,s} α R_D(…),

"`r_0=1` and `r_s=(1+i*b_s)/sqrt(1+b_s^2)` for `s=1,2,3`", and "the skip map is `D P D*`, where
`D=diag(r)`, with unchanged real read/write gates". This file is the skip;
`Transformer.PhasorV2.MHCRouter` gives `P` and `r`, and `Transformer.PhasorV2.MHC` is the
sublayer.

* `phase b = (1 + i b)/√(1 + b²)` is `e^{i arctan b}` (`phase_eq_exp`). It lies on the unit
  circle (`norm_phase`) and is one at `b = 0` (`phase_zero`).
* The source's sum `phaseSkip r P h` is `D P D* h` (`phaseSkip_eq`): turn every stream back by its
  phase, mix the streams by the real matrix `P` as the paper's skip `mix` does, and turn them
  forward again. Unit phases keep the paper's bounds:
  * if `P ≥ 0` has unit row sums, a skip of streams of length at most `B` has length at most
    `B` (`norm_phaseSkip_le`);
  * every energy bound of `mix P` holds for the phase skip (`energy_phaseSkip_le`);
  * if the rows of `P` sum to one, the streams aligned with the phases, `h_t = r_t v`, are kept
    (`phaseSkip_aligned`). All phases one gives the paper's skip (`phaseSkip_one`), which keeps
    equal streams.
-/

@[expose] public section

open scoped ComplexConjugate

namespace Transformer.PhasorV2

variable {c : ℕ}

/-- The four streams of a position. -/
abbrev State (c : ℕ) := Fin 4 → CSpace c

/-- **A stream phase** (DiffGPT's main model, *Architecture*): `(1 + i b)/√(1 + b²)`. -/
noncomputable def phase (b : ℝ) : ℂ :=
  (1 + (b : ℂ) * Complex.I) / ((Real.sqrt (1 + b ^ 2) : ℝ) : ℂ)

/-- **The phase is `e^{i arctan b}`.** -/
theorem phase_eq_exp (b : ℝ) : phase b = Complex.exp (Real.arctan b * Complex.I) := by
  rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_arctan,
    Real.sin_arctan, phase]
  push_cast
  ring

/-- **Every phase has modulus one.** -/
theorem norm_phase (b : ℝ) : ‖phase b‖ = 1 := by
  rw [phase_eq_exp]
  exact Complex.norm_exp_ofReal_mul_I _

/-- **The phase of `b = 0`**, the initialization, is one. -/
theorem phase_zero : phase 0 = 1 := by
  simp [phase]

/-- **The paper's skip** (arXiv:2512.24880v2, Eq. (3)): stream `s` of `P h` is `∑_t P_{s,t} h_t`
for a real matrix `P`. -/
noncomputable def mix (P : Matrix (Fin 4) (Fin 4) ℝ) (h : State c) : State c :=
  fun s => ∑ t, (P s t : ℂ) • h t

/-- `D = diag(r)`: stream `s` turned by `r_s`. -/
noncomputable def turnStreams (r : Fin 4 → ℂ) (h : State c) : State c := fun s => r s • h s

/-- **The phase-conjugated skip** (DiffGPT's main model, *Architecture*): stream `s` is
`∑_t P_{s,t} r_s conj(r_t) h_t`. -/
noncomputable def phaseSkip (r : Fin 4 → ℂ) (P : Matrix (Fin 4) (Fin 4) ℝ) (h : State c) :
    State c :=
  fun s => ∑ t, ((P s t : ℂ) * r s * conj (r t)) • h t

/-- **The skip map is `D P D*`**: turn back, mix by `P`, turn forward. -/
theorem phaseSkip_eq (r : Fin 4 → ℂ) (P : Matrix (Fin 4) (Fin 4) ℝ) (h : State c) :
    phaseSkip r P h = turnStreams r (mix P (turnStreams (fun t => conj (r t)) h)) := by
  funext s
  simp only [phaseSkip, turnStreams, mix, Finset.smul_sum, smul_smul]
  refine Finset.sum_congr rfl fun t _ => ?_
  congr 1
  ring

/-- **All phases one give the paper's skip.** -/
theorem phaseSkip_one (P : Matrix (Fin 4) (Fin 4) ℝ) (h : State c) :
    phaseSkip (fun _ => 1) P h = mix P h := by
  funext s
  simp [phaseSkip, mix]

/-- **Unit phases keep the length of every stream.** -/
theorem norm_turnStreams {r : Fin 4 → ℂ} (hr : ∀ s, ‖r s‖ = 1) (h : State c) (s : Fin 4) :
    ‖turnStreams r h s‖ = ‖h s‖ := by
  rw [turnStreams, norm_smul, hr s, one_mul]

/-- The phases of three coordinates `b`, stream `0` at `b = 0`. -/
example (b : Fin 4 → ℝ) (h : State 64) :=
  norm_turnStreams (r := fun s => phase (b s)) (fun s => norm_phase (b s)) h 2

/-- **The paper's skip keeps a bound of the streams** when `P ≥ 0` has unit row sums. -/
theorem norm_mix_le {P : Matrix (Fin 4) (Fin 4) ℝ} (hP : ∀ s t, 0 ≤ P s t)
    (hrow : ∀ s, ∑ t, P s t = 1) {h : State c} {B : ℝ} (hB : ∀ t, ‖h t‖ ≤ B) (s : Fin 4) :
    ‖mix P h s‖ ≤ B := by
  calc ‖mix P h s‖ ≤ ∑ t, ‖(P s t : ℂ) • h t‖ := norm_sum_le _ _
    _ = ∑ t, P s t * ‖h t‖ := Finset.sum_congr rfl fun t _ => by
        rw [norm_smul, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (hP s t)]
    _ ≤ ∑ t, P s t * B :=
        Finset.sum_le_sum fun t _ => mul_le_mul_of_nonneg_left (hB t) (hP s t)
    _ = B := by rw [← Finset.sum_mul, hrow s, one_mul]

/-- The uniform mixture of four normalized streams. -/
example (e : Fin 4 → CSpace 64) :=
  norm_mix_le (P := Matrix.of fun _ _ => 1 / 4) (fun _ _ => by norm_num)
    (fun _ => by norm_num [Fin.sum_univ_four]) (h := fun t => rms (1 / 2 ^ 23) (e t))
    (fun t => norm_rms_le (by norm_num) (e t)) 0

/-- **The phase skip keeps a bound of the streams** when `P ≥ 0` has unit row sums. -/
theorem norm_phaseSkip_le {r : Fin 4 → ℂ} (hr : ∀ s, ‖r s‖ = 1)
    {P : Matrix (Fin 4) (Fin 4) ℝ} (hP : ∀ s t, 0 ≤ P s t) (hrow : ∀ s, ∑ t, P s t = 1)
    {h : State c} {B : ℝ} (hB : ∀ t, ‖h t‖ ≤ B) (s : Fin 4) : ‖phaseSkip r P h s‖ ≤ B := by
  have hr' : ∀ t, ‖conj (r t)‖ = 1 := fun t => by rw [Complex.norm_conj, hr t]
  rw [phaseSkip_eq, norm_turnStreams hr]
  exact norm_mix_le hP hrow (fun t => (norm_turnStreams hr' h t).trans_le (hB t)) s

/-- Phases of any coordinates, the uniform mixture, normalized streams. -/
example (b : Fin 4 → ℝ) (e : Fin 4 → CSpace 64) :=
  norm_phaseSkip_le (r := fun s => phase (b s)) (fun s => norm_phase (b s))
    (P := Matrix.of fun _ _ => 1 / 4) (fun _ _ => by norm_num)
    (fun _ => by norm_num [Fin.sum_univ_four]) (h := fun t => rms (1 / 2 ^ 23) (e t))
    (fun t => norm_rms_le (by norm_num) (e t)) 3

/-- The squared length `∑_s ‖h_s‖²` of a state: that of its `8c = 4D` real coordinates. -/
noncomputable def energy (h : State c) : ℝ := ∑ s, ‖h s‖ ^ 2

/-- **The identity skip keeps every stream.** -/
theorem mix_one (h : State c) : mix 1 h = h := by
  funext s
  simp [mix, Matrix.one_apply]

/-- **Every energy bound of the paper's skip holds for the phase skip**, with unit phases: the
source's "unitary conjugation preserves … its energy bounds". -/
theorem energy_phaseSkip_le {r : Fin 4 → ℂ} (hr : ∀ s, ‖r s‖ = 1)
    {P : Matrix (Fin 4) (Fin 4) ℝ} {K : ℝ} (hP : ∀ g : State c, energy (mix P g) ≤ K * energy g)
    (h : State c) : energy (phaseSkip r P h) ≤ K * energy h := by
  have hr' : ∀ t, ‖conj (r t)‖ = 1 := fun t => by rw [Complex.norm_conj, hr t]
  have hturn : ∀ (q : Fin 4 → ℂ), (∀ s, ‖q s‖ = 1) → ∀ g : State c,
      energy (turnStreams q g) = energy g := fun q hq g => by
    simp only [energy, norm_turnStreams hq]
  rw [phaseSkip_eq, hturn r hr, ← hturn _ hr' h]
  exact hP _

/-- The identity skip, energy gain one. -/
example (b : Fin 4 → ℝ) (h : State 64) :=
  energy_phaseSkip_le (r := fun s => phase (b s)) (fun s => norm_phase (b s)) (P := 1) (K := 1)
    (fun g => by rw [mix_one, one_mul]) h

/-- **The streams aligned with the phases are kept**, `h_t = r_t v`, when the rows of `P` sum to
one: the source's "phase-aligned conserved mode". -/
theorem phaseSkip_aligned {r : Fin 4 → ℂ} (hr : ∀ s, ‖r s‖ = 1) {P : Matrix (Fin 4) (Fin 4) ℝ}
    (hrow : ∀ s, ∑ t, P s t = 1) (v : CSpace c) :
    phaseSkip r P (fun t => r t • v) = fun s => r s • v := by
  have hback : turnStreams (fun t => conj (r t)) (fun t => r t • v) = fun _ => v := by
    funext t
    simp only [turnStreams, smul_smul, Complex.conj_mul', hr t]
    simp
  have hmix : mix P (fun _ => v) = fun _ => v := by
    funext s
    simp only [mix]
    rw [← Finset.sum_smul, ← Complex.ofReal_sum, hrow s, Complex.ofReal_one, one_smul]
  rw [phaseSkip_eq, hback, hmix]
  rfl

/-- The uniform mixture keeps the aligned streams. -/
example (b : Fin 4 → ℝ) (v : CSpace 64) :=
  phaseSkip_aligned (r := fun s => phase (b s)) (fun s => norm_phase (b s))
    (P := Matrix.of fun _ _ => 1 / 4) (fun _ => by norm_num [Fin.sum_univ_four]) v

end Transformer.PhasorV2

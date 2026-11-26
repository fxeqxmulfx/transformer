module

public import Transformer.PhasorV2.MHCPhase
public import Transformer.MHC.Section4_Parameterization

/-!
# Phasor v2: the router of an mHC sublayer

Every attention and feed-forward sublayer has its own router. Source: DiffGPT
(`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`, *Architecture*, and
`mhc.json`: the parameter-free RMS of the flattened four-stream state, a real projection to 24
logits, three learned scales and 24 biases; sigmoid read gates "without sum normalization",
twice-sigmoid write gates, and "the original 20 cycle real Sinkhorn P", columns then rows. This
is the parameterization of Xie et al. (arXiv:2512.24880v2, §4.2), whose gates and finite Sinkhorn
map are `MHC.gate`, `MHC.writeGate` and `MHC.finiteSinkhorn`. Three real coordinates `b_s` give
the phases `r_0 = 1` and `r_s = phase b_s`, shared across tokens (`Router.rot`). The source's
FP16/TF32 router arithmetic, its log-space first Sinkhorn cycle and the fixed rescaling of the
router RMS are numerics; in exact reals they change nothing.

* The router input has energy at most `4D` (`energy_flatRMS_le`).
* Read gates lie in `(0, 1)`, write gates in `(0, 2)` (`Router.read_mem`, `Router.write_mem`).
  The skip matrix is positive with unit row sums (`Router.skip_pos`, `Router.skip_row_sum`); its
  columns need not sum to one (`MHC.one_cycle_not_doubly_stochastic`). Every phase has modulus
  one (`Router.norm_rot`).
* The source's initialization (`Router.initial`: scales `0.01`, read and write biases `0`, skip
  bias `2I`, phase coordinates `0`) has all phases one (`Router.rot_initial`), so its skip is the
  paper's (`Router.phaseSkip_eq_mix`).
-/

@[expose] public section

namespace Transformer.PhasorV2

variable {c : ℕ}

/-- **The router input** `R_{4D}(vec h)` (`mhc.json`, `router_input`): the parameter-free RMS of
the flattened state, over its `4D = 8c` real coordinates. -/
noncomputable def flatRMS (eps : ℝ) (h : State c) : State c :=
  ((Real.sqrt (8 * c) / Real.sqrt (energy h + 8 * c * eps) : ℝ) : ℂ) • h

/-- **The router input has energy at most `4D`**: RMS at most one, for every `ε ≥ 0`. -/
theorem energy_flatRMS_le {eps : ℝ} (heps : 0 ≤ eps) (h : State c) :
    energy (flatRMS eps h) ≤ 8 * c := by
  have hE : 0 ≤ energy h := Finset.sum_nonneg fun s _ => sq_nonneg _
  have hε : 0 ≤ 8 * (c : ℝ) * eps := mul_nonneg (by positivity) heps
  have hden : 0 ≤ energy h + 8 * c * eps := add_nonneg hE hε
  have hk : energy (flatRMS eps h) =
      (Real.sqrt (8 * c) / Real.sqrt (energy h + 8 * c * eps)) ^ 2 * energy h := by
    simp only [energy, flatRMS, Pi.smul_apply, norm_smul, Complex.norm_real, Real.norm_eq_abs,
      mul_pow, sq_abs, Finset.mul_sum]
  rw [hk, div_pow, Real.sq_sqrt (by positivity), Real.sq_sqrt hden, div_mul_eq_mul_div,
    mul_div_assoc]
  exact mul_le_of_le_one_right (by positivity)
    (div_le_one_of_le₀ (le_add_of_nonneg_right hε) hden)

/-- The main model: `D = 128`, `ε = 2^{−23}`. -/
example (h : State 64) := energy_flatRMS_le (eps := 1 / 2 ^ 23) (by norm_num) h

/-- **The router of a sublayer** (`mhc.json`, `router_projection`): the real projections of the
normalized state to the 4 read, 4 write and 16 skip logits, one scale for each group, the 24
biases, and the phase coordinates `b_1, b_2, b_3`. -/
structure Router (c : ℕ) where
  readProj : State c →ₗ[ℝ] (Fin 4 → ℝ)
  writeProj : State c →ₗ[ℝ] (Fin 4 → ℝ)
  skipProj : State c →ₗ[ℝ] Matrix (Fin 4) (Fin 4) ℝ
  scale : Fin 3 → ℝ
  readBias : Fin 4 → ℝ
  writeBias : Fin 4 → ℝ
  skipBias : Matrix (Fin 4) (Fin 4) ℝ
  b : Fin 3 → ℝ

/-- **The read gates** at the state `h`: `σ(scale₀ · readProj(R_{4D} h) + readBias)`. -/
noncomputable def Router.read (R : Router c) (eps : ℝ) (h : State c) (s : Fin 4) : ℝ :=
  MHC.gate (R.scale 0 * R.readProj (flatRMS eps h) s + R.readBias s)

/-- **The write gates**: `2σ(scale₁ · writeProj(R_{4D} h) + writeBias)`. -/
noncomputable def Router.write (R : Router c) (eps : ℝ) (h : State c) (s : Fin 4) : ℝ :=
  MHC.writeGate (R.scale 1 * R.writeProj (flatRMS eps h) s + R.writeBias s)

/-- **The skip matrix**: twenty Sinkhorn cycles from `exp(scale₂ · skipProj(R_{4D} h) + skipBias)`. -/
noncomputable def Router.skip (R : Router c) (eps : ℝ) (h : State c) :
    Matrix (Fin 4) (Fin 4) ℝ :=
  MHC.finiteSinkhorn
    (Matrix.of fun s t => R.scale 2 * R.skipProj (flatRMS eps h) s t + R.skipBias s t) 20

/-- **The phases**: `r_0 = 1` and `r_{s+1} = phase b_{s+1}`. -/
noncomputable def Router.rot (R : Router c) : Fin 4 → ℂ :=
  Fin.cons 1 fun s => phase (R.b s)

theorem Router.read_mem (R : Router c) (eps : ℝ) (h : State c) (s : Fin 4) :
    0 < R.read eps h s ∧ R.read eps h s < 1 :=
  ⟨MHC.gate_pos _, MHC.gate_lt_one _⟩

theorem Router.write_mem (R : Router c) (eps : ℝ) (h : State c) (s : Fin 4) :
    0 < R.write eps h s ∧ R.write eps h s < 2 :=
  MHC.writeGate_bounds _

theorem Router.skip_pos (R : Router c) (eps : ℝ) (h : State c) (s t : Fin 4) :
    0 < R.skip eps h s t :=
  MHC.sinkhornIterate_pos _ (fun _ _ => Real.exp_pos _) 20 s t

/-- **The rows of the skip matrix sum to one**, after the final row normalization. -/
theorem Router.skip_row_sum (R : Router c) (eps : ℝ) (h : State c) (s : Fin 4) :
    ∑ t, R.skip eps h s t = 1 :=
  MHC.twenty_iterations_row_sum _ s

theorem Router.rot_zero (R : Router c) : R.rot 0 = 1 := by
  simp [Router.rot]

theorem Router.rot_succ (R : Router c) (s : Fin 3) : R.rot s.succ = phase (R.b s) := by
  simp [Router.rot]

theorem Router.norm_rot (R : Router c) (s : Fin 4) : ‖R.rot s‖ = 1 := by
  refine Fin.cases ?_ (fun t => ?_) s
  · rw [R.rot_zero, norm_one]
  · rw [R.rot_succ, norm_phase]

/-- **The router keeps the streams aligned with its phases**: `h_t = r_t v` passes the skip of any
state `g` unchanged. -/
example (R : Router 64) (g : State 64) (v : CSpace 64) :=
  phaseSkip_aligned R.norm_rot (R.skip_row_sum (1 / 2 ^ 23) g) v

/-- **Zero phase coordinates give all phases one.** -/
theorem Router.rot_eq_one (R : Router c) (hb : R.b = 0) : R.rot = fun _ => 1 := by
  funext s
  refine Fin.cases ?_ (fun t => ?_) s
  · exact R.rot_zero
  · rw [R.rot_succ, hb, Pi.zero_apply, phase_zero]

/-- **With zero phase coordinates the skip is the paper's.** -/
theorem Router.phaseSkip_eq_mix (R : Router c) (hb : R.b = 0) (P : Matrix (Fin 4) (Fin 4) ℝ)
    (h : State c) : phaseSkip R.rot P h = mix P h := by
  rw [R.rot_eq_one hb, phaseSkip_one]

/-- **The source's initialization** (`mhc.json`, `initialization`) with given projections, drawn
there from `N(0, 0.02)`: scales `0.01`, read and write biases `0`, skip bias `2I`, phase
coordinates `0`. -/
noncomputable def Router.initial (readProj writeProj : State c →ₗ[ℝ] (Fin 4 → ℝ))
    (skipProj : State c →ₗ[ℝ] Matrix (Fin 4) (Fin 4) ℝ) : Router c where
  readProj := readProj
  writeProj := writeProj
  skipProj := skipProj
  scale := fun _ => 1 / 100
  readBias := 0
  writeBias := 0
  b := 0
  skipBias := 2 • (1 : Matrix (Fin 4) (Fin 4) ℝ)

/-- **At initialization every phase is one**, whatever the projections. -/
theorem Router.rot_initial (readProj writeProj : State c →ₗ[ℝ] (Fin 4 → ℝ))
    (skipProj : State c →ₗ[ℝ] Matrix (Fin 4) (Fin 4) ℝ) :
    (Router.initial readProj writeProj skipProj).rot = fun _ => 1 :=
  Router.rot_eq_one _ rfl

/-- At initialization the skip of any state is the paper's. -/
example (readProj writeProj : State 64 →ₗ[ℝ] (Fin 4 → ℝ))
    (skipProj : State 64 →ₗ[ℝ] Matrix (Fin 4) (Fin 4) ℝ) (g h : State 64) :=
  (Router.initial readProj writeProj skipProj).phaseSkip_eq_mix rfl
    ((Router.initial readProj writeProj skipProj).skip (1 / 2 ^ 23) g) h

end Transformer.PhasorV2

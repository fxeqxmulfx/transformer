module

public import Transformer.PhasorV2.MHCRouter

/-!
# Phasor v2: the mHC sublayer with phases

Source: DiffGPT (`~/python_projects/gpt`, `fe1d528`), `research/main_model/README.txt`,
*Architecture*, and `mhc.json`. This is the update of Xie et al. (arXiv:2512.24880v2, Eq. (3),
constrained as in §4.2) with a Peri-norm branch and phases. At every position `i`,

    z_i = R_D(∑_s read_{i,s} h_{i,s}),
    h'_{i,s} = ∑_t P_{i,s,t} r_s conj(r_t) h_{i,t} + write_{i,s} α R_D(F(z)_i),

where the sublayer's router reads the state `h_i` of the position (`Router.step`). The embedding
enters as "four identical copies after embedding RMS", `h⁰_{i,s} = R_D(E[token_i])`. `stack` runs
`n` sublayers, each with its router and raw branch; in the source attention and feed-forward
alternate, `n = 2L`.

* A sublayer moves a stream by the skip, which keeps a bound of the streams, plus the write,
  which has length at most `2 |α| √D` (`Router.norm_step_le`). So after `n` sublayers every
  stream has grown by at most `2 n |α| √D` (`norm_stack_le`).
* From the normalized embedding, after `n` sublayers every stream has RMS at most `1 + 2 n |α|`
  (`rms_stack_le`), for all weights, routers and branches. At the source's `α = 1/√(2L)` after
  `2L` sublayers that is `1 + 2√(2L)` (`rms_stack_depth_le`).
* Causal branches give causal sublayers and stacks (`Router.isCausal_step`, `isCausal_stack`).
-/

@[expose] public section

open Transformer.Sandwich

namespace Transformer.PhasorV2

variable {c : ℕ}

/-- **What the router reads** at the state `h`: `∑_s read_s h_s`, before the branch's `R_D`. -/
noncomputable def Router.readMix (R : Router c) (eps : ℝ) (h : State c) : CSpace c :=
  ∑ s, (R.read eps h s : ℂ) • h s

/-- **An mHC sublayer with phases** (DiffGPT's main model, *Architecture*): at the position `i`,
stream `s` is the phase skip of the four streams of `h_i` plus the write gate times `α` times the
Peri-norm branch of what the router reads at every position. -/
noncomputable def Router.step (R : Router c) (eps α : ℝ) (F : (ℕ → CSpace c) → ℕ → CSpace c)
    (h : ℕ → State c) (i : ℕ) : State c := fun s =>
  phaseSkip R.rot (R.skip eps (h i)) (h i) s +
    ((R.write eps (h i) s * α : ℝ) : ℂ) • periBranch eps F (fun j => R.readMix eps (h j)) i

/-- **`n` sublayers**, the sublayer `l` with the router `R l` and the raw branch `F l`. -/
noncomputable def stack (eps α : ℝ) (R : ℕ → Router c)
    (F : ℕ → (ℕ → CSpace c) → ℕ → CSpace c) : ℕ → (ℕ → State c) → ℕ → State c
  | 0, h => h
  | n + 1, h => (R n).step eps α (F n) (stack eps α R F n h)

/-- **A sublayer moves a stream by at most `2 |α| √D` beyond a bound of the streams**: if every
stream of `h_i` has length at most `B`, every new stream has length at most `B + 2 |α| √D`. -/
theorem Router.norm_step_le (R : Router c) {eps : ℝ} (heps : 0 ≤ eps) (α : ℝ)
    (F : (ℕ → CSpace c) → ℕ → CSpace c) (h : ℕ → State c) (i : ℕ) {B : ℝ}
    (hB : ∀ t, ‖h i t‖ ≤ B) (s : Fin 4) :
    ‖R.step eps α F h i s‖ ≤ B + 2 * |α| * Real.sqrt (2 * c) := by
  have hskip := norm_phaseSkip_le R.norm_rot (fun s t => (R.skip_pos eps (h i) s t).le)
    (R.skip_row_sum eps (h i)) hB s
  have hw := R.write_mem eps (h i) s
  have hbr := norm_periBranch_le heps F (fun j => R.readMix eps (h j)) i
  have hwrite : ‖((R.write eps (h i) s * α : ℝ) : ℂ) •
      periBranch eps F (fun j => R.readMix eps (h j)) i‖ ≤ 2 * |α| * Real.sqrt (2 * c) := by
    rw [norm_smul, Complex.norm_real, Real.norm_eq_abs, abs_mul, abs_of_pos hw.1]
    exact mul_le_mul (mul_le_mul_of_nonneg_right hw.2.le (abs_nonneg α)) hbr (norm_nonneg _)
      (by positivity)
  exact (norm_add_le _ _).trans (add_le_add hskip hwrite)

/-- A sublayer of the main model on the normalized embedding, `α = 1/√8`. -/
example (R : Router 64) (F : (ℕ → CSpace 64) → ℕ → CSpace 64) (e : ℕ → CSpace 64) :=
  R.norm_step_le (eps := 1 / 2 ^ 23) (by norm_num) (1 / Real.sqrt 8) F
    (fun j _ => rms (1 / 2 ^ 23) (e j)) 7 (fun _ => norm_rms_le (by norm_num) (e 7)) 0

/-- **After `n` sublayers every stream has grown by at most `2 n |α| √D`**, whatever the
routers and branches. -/
theorem norm_stack_le {eps : ℝ} (heps : 0 ≤ eps) (α : ℝ) (R : ℕ → Router c)
    (F : ℕ → (ℕ → CSpace c) → ℕ → CSpace c) (n : ℕ) (h : ℕ → State c) (i : ℕ) {B : ℝ}
    (hB : ∀ t, ‖h i t‖ ≤ B) (s : Fin 4) :
    ‖stack eps α R F n h i s‖ ≤ B + n * (2 * |α| * Real.sqrt (2 * c)) := by
  induction n generalizing s with
  | zero => simpa [stack] using hB s
  | succ n ih =>
    rw [stack]
    refine ((R n).norm_step_le heps α (F n) _ i ih s).trans (le_of_eq ?_)
    push_cast
    ring

/-- Eight sublayers of the main model on the normalized embedding. -/
example (R : ℕ → Router 64) (F : ℕ → (ℕ → CSpace 64) → ℕ → CSpace 64) (e : ℕ → CSpace 64) :=
  norm_stack_le (eps := 1 / 2 ^ 23) (by norm_num) (1 / Real.sqrt 8) R F 8
    (fun j _ => rms (1 / 2 ^ 23) (e j)) 7 (fun _ => norm_rms_le (by norm_num) (e 7)) 0

/-- **From the normalized embedding every stream has RMS at most `1 + 2 n |α|`** after `n`
sublayers: the RMS `‖h‖/√D`, for all weights, routers and branches. -/
theorem rms_stack_le {eps : ℝ} (heps : 0 ≤ eps) (hc : 0 < c) (α : ℝ) (R : ℕ → Router c)
    (F : ℕ → (ℕ → CSpace c) → ℕ → CSpace c) (n : ℕ) (e : ℕ → CSpace c) (i : ℕ) (s : Fin 4) :
    ‖stack eps α R F n (fun j _ => rms eps (e j)) i s‖ / Real.sqrt (2 * c) ≤ 1 + 2 * n * |α| := by
  have hs : 0 < Real.sqrt (2 * c) := Real.sqrt_pos.2 (mul_pos two_pos (Nat.cast_pos.2 hc))
  rw [div_le_iff₀ hs]
  calc ‖stack eps α R F n (fun j _ => rms eps (e j)) i s‖
      ≤ Real.sqrt (2 * c) + n * (2 * |α| * Real.sqrt (2 * c)) :=
        norm_stack_le heps α R F n (fun j _ => rms eps (e j)) i
          (fun _ => norm_rms_le heps (e i)) s
    _ = (1 + 2 * n * |α|) * Real.sqrt (2 * c) := by ring

/-- Eight sublayers of the main model. -/
example (R : ℕ → Router 64) (F : ℕ → (ℕ → CSpace 64) → ℕ → CSpace 64) (e : ℕ → CSpace 64) :=
  rms_stack_le (eps := 1 / 2 ^ 23) (by norm_num) (c := 64) (by norm_num) (1 / Real.sqrt 8) R F 8
    e 7 0

/-- **`L` layers at the source's `α = 1/√(2L)`**: every stream has RMS at most `1 + 2√(2L)`. -/
theorem rms_stack_depth_le {eps : ℝ} (heps : 0 ≤ eps) (hc : 0 < c) {L : ℕ} (hL : 0 < L)
    (R : ℕ → Router c) (F : ℕ → (ℕ → CSpace c) → ℕ → CSpace c) (e : ℕ → CSpace c) (i : ℕ)
    (s : Fin 4) :
    ‖stack eps (1 / Real.sqrt (2 * L)) R F (2 * L) (fun j _ => rms eps (e j)) i s‖ /
      Real.sqrt (2 * c) ≤ 1 + 2 * Real.sqrt (2 * L) := by
  have hpos : (0 : ℝ) < 2 * L := mul_pos two_pos (Nat.cast_pos.2 hL)
  refine (rms_stack_le heps hc _ R F (2 * L) e i s).trans (le_of_eq ?_)
  rw [abs_of_pos (one_div_pos.2 (Real.sqrt_pos.2 hpos))]
  push_cast
  rw [mul_one_div, mul_div_assoc, Real.div_sqrt]

/-- Four layers: RMS at most `1 + 2√8`. -/
example (R : ℕ → Router 64) (F : ℕ → (ℕ → CSpace 64) → ℕ → CSpace 64) (e : ℕ → CSpace 64) :=
  rms_stack_depth_le (eps := 1 / 2 ^ 23) (by norm_num) (c := 64) (by norm_num) (L := 4)
    (by norm_num) R F e 7 0

/-- **A causal raw branch gives a causal sublayer**: the router and the skip read only the
position's own state. -/
theorem Router.isCausal_step (R : Router c) (eps α : ℝ) {F : (ℕ → CSpace c) → ℕ → CSpace c}
    (hF : IsCausal F) : IsCausal (R.step eps α F) := by
  intro x y i hxy
  have hb := isCausal_periBranch eps hF (fun j => R.readMix eps (x j))
    (fun j => R.readMix eps (y j)) i fun j hj => by rw [hxy j hj]
  funext s
  simp only [Router.step]
  rw [hxy i le_rfl, hb]

/-- A feed-forward branch. -/
example (R : Router 4) (f : CSpace 4 → CSpace 4) :=
  R.isCausal_step 1 1 (F := fun y i => f (y i)) fun _ _ i h => congrArg f (h i le_rfl)

/-- **Causal branches give a causal stack.** -/
theorem isCausal_stack (eps α : ℝ) (R : ℕ → Router c) {F : ℕ → (ℕ → CSpace c) → ℕ → CSpace c}
    (hF : ∀ l, IsCausal (F l)) (n : ℕ) : IsCausal (stack eps α R F n) := by
  induction n with
  | zero => exact fun x y i h => h i le_rfl
  | succ n ih => exact ((R n).isCausal_step eps α (hF n)).comp ih

/-- A stack of feed-forward branches. -/
example (R : ℕ → Router 4) (f : ℕ → CSpace 4 → CSpace 4) :=
  isCausal_stack 1 1 R (F := fun l y i => f l (y i))
    (fun l _ _ i h => congrArg (f l) (h i le_rfl)) 3

end Transformer.PhasorV2

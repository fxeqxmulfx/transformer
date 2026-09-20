/-
# The emergence of clusters in self-attention dynamics — the symmetric triple

A construction, not a statement of arXiv:2305.05465v6.

§7 argues about solutions of `e:Idnonresca` in `d = 1` with `n` pairwise
distinct tokens, and its lemmas separate the tokens that stay bounded from the
ones that do not.  The source exhibits no solution, and the repository has
none either: the stationary configurations that witness the hypotheses
elsewhere in this paper all have coinciding tokens, so none of them is ordered
in the sense of §7.

This file supplies one.  Put three tokens at `-u`, `0`, `u` on the line.  The
middle token sees every other one at inner product `0`, so it is pulled by the
uniform average `(-u + 0 + u)/3 = 0` and does not move; the outer two are
pulled symmetrically, and the whole system collapses to the scalar equation

  `u̇ = u (e^{u²} - e^{-u²}) / (e^{u²} + 1 + e^{-u²})`,

which is `symDrift`.  Everything below is proved from a solution `u` of that
scalar equation, taken as a hypothesis: the reduction is exact, and it is the
only nontrivial solution family of `e:Idnonresca` available in closed form.

Its use is to witness §7's hypotheses.  It is a configuration of three ordered
tokens, with exactly one of them — the interior one — identically `0`, hence
bounded; that is the case `i₀ ∉ {1, n}` of `l:boundedother`, which no
configuration with fewer than three tokens can exhibit.
-/

import Transformer.Clusters.Section7_Unbounded

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

/-! ### The line `ℝ^1` -/

/-- The unit vector of `ℝ^1`. -/
noncomputable def unit1 : EucSpace 1 := EuclideanSpace.single 0 (1 : ℝ)

@[simp] theorem norm_unit1 : ‖unit1‖ = 1 := by
  simp [unit1]

@[simp] theorem coord_smul_unit1 (a : ℝ) : (a • unit1) 0 = a := by
  simp [unit1]

/-- On the line, the inner product of two multiples of the unit vector is the
product of the multipliers. -/
@[simp] theorem inner_smul_unit1 (a b : ℝ) :
    inner (𝕜 := ℝ) (a • unit1) (b • unit1) = a * b := by
  rw [real_inner_smul_left, real_inner_smul_right, real_inner_self_eq_norm_sq, norm_unit1]
  ring

/-! ### The symmetric triple -/

/-- The three signs `(-1, 0, 1)` of the symmetric triple. -/
def symSign : Idx 3 → ℝ := ![-1, 0, 1]

@[simp] theorem symSign_zero : symSign 0 = -1 := rfl

@[simp] theorem symSign_one : symSign 1 = 0 := rfl

@[simp] theorem symSign_two : symSign 2 = 1 := rfl

/-- The symmetric triple of tokens `(-u, 0, u)` on the line. -/
noncomputable def symTriple (u : ℝ) (i : Idx 3) : EucSpace 1 := (symSign i * u) • unit1

/-- The scalar equation the symmetric triple reduces `e:Idnonresca` to:

  `u̇ = u (e^{u²} - e^{-u²}) / (e^{u²} + 1 + e^{-u²})`. -/
noncomputable def symDrift (u : ℝ) : ℝ :=
  u * (Real.exp (u ^ 2) - Real.exp (-u ^ 2)) / (Real.exp (u ^ 2) + 1 + Real.exp (-u ^ 2))

theorem symDrift_zero : symDrift 0 = 0 := by
  simp [symDrift]

/-- The self-attention matrix of the symmetric triple: the scores are
`s_i s_j u²`. -/
theorem attentionMatrix_symTriple (u : ℝ) (i j : Idx 3) :
    attentionMatrix (1 : ParamMatrix 1) 1 (symTriple u) i j
      = Real.exp (symSign i * symSign j * u ^ 2)
          / ∑ l : Idx 3, Real.exp (symSign i * symSign l * u ^ 2) := by
  unfold attentionMatrix Perspective.softmaxWeight symTriple
  simp only [one_apply_eq_self, inner_smul_unit1]
  rw [show symSign i * u * (symSign j * u) = symSign i * symSign j * u ^ 2 by ring]
  exact congrArg _ (Finset.sum_congr rfl fun l _ => by
    rw [show symSign i * u * (symSign l * u) = symSign i * symSign l * u ^ 2 by ring])

/-- The partition function of the symmetric triple is positive. -/
theorem sum_exp_symSign_pos (u : ℝ) (i : Idx 3) :
    0 < ∑ l : Idx 3, Real.exp (symSign i * symSign l * u ^ 2) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

/-- **The scalar reduction.**  The attention-weighted average of the three
coordinates, seen from the `i`-th token, is `s_i` times `symDrift u`. -/
theorem sum_weighted_symSign (u : ℝ) (i : Idx 3) :
    ∑ j : Idx 3, (Real.exp (symSign i * symSign j * u ^ 2)
        / ∑ l : Idx 3, Real.exp (symSign i * symSign l * u ^ 2)) * (symSign j * u)
      = symSign i * symDrift u := by
  have hi : ∀ k : Idx 3, k = 0 ∨ k = 1 ∨ k = 2 := by decide
  rcases hi i with rfl | rfl | rfl <;>
    · have h : Real.exp (u ^ 2) + 1 + Real.exp (-u ^ 2) ≠ 0 := by positivity
      simp only [Fin.sum_univ_three, symSign_zero, symSign_one, symSign_two, symDrift]
      norm_num
      all_goals field_simp
      all_goals ring

/-- **The reduction.**  The drift `e:Idnonresca` puts on the `i`-th token of
the symmetric triple is `s_i` times `symDrift u`: the middle token does not
move, and the outer two move apart at equal speed. -/
theorem drift_symTriple (u : ℝ) (i : Idx 3) :
    ∑ j : Idx 3, attentionMatrix (1 : ParamMatrix 1) 1 (symTriple u) i j • symTriple u j
      = (symSign i * symDrift u) • unit1 := by
  have hstep : ∀ j : Idx 3,
      attentionMatrix (1 : ParamMatrix 1) 1 (symTriple u) i j • symTriple u j
        = (attentionMatrix (1 : ParamMatrix 1) 1 (symTriple u) i j * (symSign j * u)) • unit1 := by
    intro j
    rw [symTriple, smul_smul]
  rw [Finset.sum_congr rfl fun j _ => hstep j, ← Finset.sum_smul]
  congr 1
  simp only [attentionMatrix_symTriple]
  exact sum_weighted_symSign u i

/-- **The symmetric triple solves `e:Idnonresca`.**  Given a solution `u` of
the scalar equation, `(-u, 0, u)` is a solution of the full system. -/
theorem idNonrescaledDynamics_symTriple (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (symDrift (u t)) t) :
    IdNonrescaledDynamics (fun t => symTriple (u t)) := by
  intro t i
  rw [drift_symTriple]
  exact ((hu t).const_mul (symSign i)).smul_const unit1

/-- The symmetric triple is ordered as soon as `u > 0`. -/
theorem isOrderedConfig_symTriple {u : ℝ} (hu : 0 < u) : IsOrderedConfig (symTriple u) := by
  intro i j hij
  simp only [symTriple, coord_smul_unit1]
  fin_cases i <;> fin_cases j <;>
    simp_all [symSign, Fin.lt_def]

/-- The interior token of the symmetric triple sits at the origin, so it is
uniformly bounded. -/
theorem isBoundedToken_symTriple (u : ℝ → ℝ) :
    IsBoundedToken (fun t => symTriple (u t)) 1 := by
  refine ⟨0, fun t _ => ?_⟩
  simp [symTriple, symSign]

/-- The interior index of a triple is neither the first nor the last. -/
theorem symInterior_ne_first : (1 : Idx 3) ≠ 0 := by decide

theorem symInterior_ne_last : (1 : Idx 3) ≠ Fin.last 2 := by decide

/-- The scalar equation is satisfiable: `u ≡ 0` solves it, since
`symDrift 0 = 0`.  It is the ordered solution — one with `u 0 > 0` — that §7
needs, and that is the content of `wellposed_particles`. -/
example : ∀ t : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (symDrift ((fun _ : ℝ => (0 : ℝ)) t)) t := by
  intro t
  simpa [symDrift_zero] using hasDerivAt_const t (0 : ℝ)

end Clusters
end Transformer

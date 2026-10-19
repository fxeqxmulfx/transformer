module

public import Transformer.Kinetic.Section3_ThirdCumulant

/-!
# The third cumulant with its diagonal split off

ArXiv:2605.09213v1, §3, the proof of `lem:correl`, Step 4, `eq:est-kap3-th`: the
second differences `s²_{l,m}` of `abs_thirdCumulant_le` are `s_l` on the diagonal and
envelopes `w_{l,m}` off it. The diagonal gives `(3/2) Σ_l s_l(F) s_l(G) s_l(H)`
(`abs_thirdCumulant_le_split`), the source's term with a single token.
-/

@[expose] public section

open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Kinetic

variable {N : ℕ} (μs : Idx N → Measure ℝ) [∀ i, IsProbabilityMeasure (μs i)]

/-- **The third-order inequality, split.** -/
theorem abs_thirdCumulant_le_split {F G H : (Idx N → ℝ) → ℝ} (hF : Measurable F)
    (hG : Measurable G) (hH : Measurable H) {CF CG CH : ℝ} (hFb : ∀ x, |F x| ≤ CF)
    (hGb : ∀ x, |G x| ≤ CG) (hHb : ∀ x, |H x| ≤ CH) {sF sG sH : Idx N → ℝ}
    {wF wG wH : Idx N → Idx N → ℝ}
    (hsF : ∀ x l z, |F x - F (Function.update x l z)| ≤ sF l)
    (hsG : ∀ x l z, |G x - G (Function.update x l z)| ≤ sG l)
    (hsH : ∀ x l z, |H x - H (Function.update x l z)| ≤ sH l)
    (hwF : ∀ x l m z z', m ≠ l → |F x - F (Function.update x l z) - F (Function.update x m z') +
      F (Function.update (Function.update x l z) m z')| ≤ wF l m)
    (hwG : ∀ x l m z z', m ≠ l → |G x - G (Function.update x l z) - G (Function.update x m z') +
      G (Function.update (Function.update x l z) m z')| ≤ wG l m)
    (hwH : ∀ x l m z z', m ≠ l → |H x - H (Function.update x l z) - H (Function.update x m z') +
      H (Function.update (Function.update x l z) m z')| ≤ wH l m)
    (hwF0 : ∀ l, 0 ≤ wF l l) (hwG0 : ∀ l, 0 ≤ wG l l) (hwH0 : ∀ l, 0 ≤ wH l l) :
    |∫ x, (F x - ∫ y, F y ∂(Measure.pi μs)) * (G x - ∫ y, G y ∂(Measure.pi μs)) *
        (H x - ∫ y, H y ∂(Measure.pi μs)) ∂(Measure.pi μs)| ≤
      3 / 2 * ∑ l, sF l * sG l * sH l +
        1 / 4 * ∑ l, ∑ m, (sF m * (wG l m * sH l + sG l * wH l m) +
          sG m * (wF l m * sH l + sF l * wH l m)) := by
  refine (abs_thirdCumulant_le μs hF hG hH hFb hGb hHb hsF hsG hsH
    (s2F := fun l m => (if l = m then sF l else 0) + wF l m)
    (s2G := fun l m => (if l = m then sG l else 0) + wG l m)
    (s2H := fun l m => (if l = m then sH l else 0) + wH l m)
    (fun x l m z z' hml => (hwF x l m z z' hml).trans (by simp [hml.symm]))
    (fun x l m z z' hml => (hwG x l m z z' hml).trans (by simp [hml.symm]))
    (fun x l m z z' hml => (hwH x l m z z' hml).trans (by simp [hml.symm]))
    (fun l => by simpa using hwF0 l) (fun l => by simpa using hwG0 l)
    (fun l => by simpa using hwH0 l)).trans (le_of_eq ?_)
  have key : ∀ l m : Idx N,
      sF m * (((if l = m then sG l else 0) + wG l m) * sH l +
          sG l * ((if l = m then sH l else 0) + wH l m)) +
        sG m * (((if l = m then sF l else 0) + wF l m) * sH l +
          sF l * ((if l = m then sH l else 0) + wH l m)) =
      (if l = m then 4 * (sF l * sG l * sH l) else 0) +
        (sF m * (wG l m * sH l + sG l * wH l m) + sG m * (wF l m * sH l + sF l * wH l m)) := by
    intro l m
    split_ifs with h
    · subst h; ring
    · ring
  simp only [key, Finset.sum_add_distrib, Finset.sum_ite_eq, Finset.mem_univ, ↓reduceIte]
  rw [← Finset.mul_sum]
  ring

/-- The hypotheses of `abs_thirdCumulant_le_split` hold for constant observables under
Dirac masses, with vanishing envelopes. -/
example := abs_thirdCumulant_le_split (fun _ : Idx 2 => Measure.dirac (0 : ℝ))
  (F := fun _ => 0) (G := fun _ => 0) (H := fun _ => 0) measurable_const measurable_const
  measurable_const (CF := 0) (CG := 0) (CH := 0) (fun _ => by simp) (fun _ => by simp)
  (fun _ => by simp) (sF := fun _ => 0) (sG := fun _ => 0) (sH := fun _ => 0)
  (wF := fun _ _ => 0) (wG := fun _ _ => 0) (wH := fun _ _ => 0) (fun _ _ _ => by simp)
  (fun _ _ _ => by simp) (fun _ _ _ => by simp) (fun _ _ _ _ _ _ => by simp)
  (fun _ _ _ _ _ _ => by simp) (fun _ _ _ _ _ _ => by simp) (fun _ => le_rfl)
  (fun _ => le_rfl) (fun _ => le_rfl)

end Kinetic
end Transformer

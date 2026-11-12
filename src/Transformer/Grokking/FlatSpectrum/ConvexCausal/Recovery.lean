module

public import Transformer.Grokking.FlatSpectrum.ConvexCausal.Learner

/-!
# Every convex optimum has the identified causal rule

Source: `review/convex_causal.md`, CCa5. This theorem concerns the entire
simplex of joint mixtures of a fixed rule bank, including nondeterministic
mixtures. The feasible set is not silently replaced by its deterministic
vertices. Observed CE equality eliminates every contradicted positive weight;
input-signature separation then identifies every unseen prediction.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexCausal

open scoped BigOperators

variable {H X C S : Type*} [Fintype H] [DecidableEq H] [Fintype C] [Nonempty C]
  [DecidableEq C] [Fintype S]

omit [DecidableEq H] [Fintype C] [Nonempty C] in
/-- A contradicted rule has zero weight whenever the observed target has full mass (CCa5). -/
theorem contradicted_weight_zero (rule : H → X → C) {w : H → ℝ} (hw : Simplex w)
    (x : X) (c : C) (hmass : mix (fun h => point (rule h)) w x c = 1)
    (h : H) (hbad : rule h x ≠ c) : w h = 0 := by
  have hle (k : H) : w k * point (rule k) x c ≤ w k := by
    by_cases he : c = rule k x
    · simp [point, he]
    · simpa [point, he] using hw.1 k
  have he : (∑ k, w k * point (rule k) x c) = ∑ k, w k := hmass.trans hw.2.symm
  have hk := (Finset.sum_eq_sum_iff_of_le (fun k _ => hle k)).mp he h (Finset.mem_univ h)
  simpa [point, Ne.symm hbad] using hk.symm

/-- Every optimum on the full convex mixture domain equals the identified vertex (CCa5). -/
theorem optimizer_vertex (rule : H → X → C) (input : S → X) (label : S → C)
    (g : H) (hfit : ∀ s, rule g (input s) = label s) (hsep : Separates rule input)
    {ε : ℝ} (hε : 0 < ε) (hε1 : ε < 1) {w : H → ℝ} (hw : Simplex w)
    (hmin : IsMinOn (fun v => loss ε input label (mix (fun h => point (rule h)) v))
      {v | Simplex v} w) : w = vertex g := by
  have hr := mix_rows (fun h => rows_point (rule h)) hw
  have hbound : loss ε input label (mix (fun h => point (rule h)) w) ≤
      loss ε input label (mix (fun h => point (rule h)) (vertex g)) := hmin (simplex_vertex g)
  rw [mix_vertex] at hbound
  have hgfit : ∀ s, point (rule g) (input s) (label s) = 1 := by
    intro s
    simp [point, hfit s]
  rw [loss_eq_ideal (rows_point _) hε hε1.le input label hgfit] at hbound
  have hz (h : H) (hne : h ≠ g) : w h = 0 := by
    obtain ⟨s, hs⟩ := hsep h g hne
    apply contradicted_weight_zero rule hw (input s) (label s)
      (observed_mass hr hε hε1 input label hbound s) h
    simpa [hfit s] using hs
  have hg : w g = 1 := by
    rw [← hw.2]
    symm
    apply Finset.sum_eq_single g
    · intro h _ hne
      exact hz h hne
    · simp
  funext h
  by_cases he : h = g
  · subst h
    simpa [vertex] using hg
  · simpa [vertex, he] using hz h he

/-- The convex optimizer's complete probabilities are identified, not just observed argmaxes (CCa5). -/
theorem optimizer_prediction (rule : H → X → C) (input : S → X) (label : S → C)
    (g : H) (hfit : ∀ s, rule g (input s) = label s) (hsep : Separates rule input)
    {ε : ℝ} (hε : 0 < ε) (hε1 : ε < 1) {w : H → ℝ} (hw : Simplex w)
    (hmin : IsMinOn (fun v => loss ε input label (mix (fun h => point (rule h)) v))
      {v | Simplex v} w) : mix (fun h => point (rule h)) w = point (rule g) := by
  rw [optimizer_vertex rule input label g hfit hsep hε hε1 hw hmin, mix_vertex]

/-- Inhabited optimizer hypotheses: a concrete fitted vertex is globally optimal (CCa5 witness). -/
theorem parity_vertex_minimum :
    IsMinOn (fun w => loss (1 / 4) (fun _ : Unit => (false, false)) (fun _ => false)
      (mix (fun h => point (parityRule h)) w)) {w | Simplex w} (vertex false) := by
  intro w hw
  change loss (1 / 4) (fun _ : Unit => (false, false)) (fun _ => false)
    (mix (fun h => point (parityRule h)) (vertex false)) ≤ _
  rw [mix_vertex]
  exact fit_minimum (rows_point (parityRule false)) (by norm_num) (by norm_num)
    (fun _ : Unit => (false, false)) (fun _ => false) (by intro s; rfl)
    (mix_rows (fun h => rows_point (parityRule h)) hw)

example := contradicted_weight_zero parityRule (simplex_vertex false) (false, false) false
  (by simp [mix_vertex, point, parityRule]) true (by decide)
example := optimizer_vertex parityRule (fun _ : Unit => (false, false)) (fun _ => false)
  false (by intro s; rfl) parity_separates (by norm_num) (by norm_num)
  (simplex_vertex false) parity_vertex_minimum
example := optimizer_prediction parityRule (fun _ : Unit => (false, false)) (fun _ => false)
  false (by intro s; rfl) parity_separates (by norm_num) (by norm_num)
  (simplex_vertex false) parity_vertex_minimum

end Transformer.Grokking.FlatSpectrum.ConvexCausal

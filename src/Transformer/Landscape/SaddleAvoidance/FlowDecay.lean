module

public import Transformer.Landscape.SaddleAvoidance.Decay
public import Mathlib.Analysis.SpecialFunctions.Pow.Deriv
public import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Saddle avoidance — functions that decrease at a power of themselves

The continuous-time counterparts of `Landscape.SaddleAvoidance.Decay`. If
`r' ≤ -λ r` on `[T, ∞)`, then `e^{λ t} r t` decreases, so
`r t ≤ r T e^{λ T} e^{-λ t}` (`le_mul_exp_of_hasDerivAt`). If `r > 0` and
`r' ≤ -c r ^ (1 + β)` with `β > 0`, then `r ^ (-β)` grows at least at the rate
`β c` (`rpow_neg_add_mul_le_of_hasDerivAt`), so
`r t ≤ C (t + 1) ^ (-1/β)` for `t ≥ 2T + 1` when `c > 0`
(`le_mul_rpow_of_hasDerivAt`). A bound from some time on holds
from time `0` on, with a larger constant, when the bounded function is
bounded on every `[0, T]` and the bounding one is positive and antitone
(`exists_forall_le_mul_of_eventually_atTop`), and so does a bound by a power
of a bounded function (`exists_le_mul_and_le_mul_rpow_of_eventually`).

They give the rates of descent curves (`Landscape.SaddleAvoidance.FlowRates`), the
analogue for flows of `prop:rates` of arXiv:1602.04915.
-/

@[expose] public section

open Filter Set Real

namespace Transformer.Landscape.SaddleAvoidance

/-- **A bound from some time on holds from time `0` on**, with a larger
constant, when the bounded function is bounded on every `[0, T]` and the
bounding one is positive and antitone on `[0, ∞)`. -/
theorem exists_forall_le_mul_of_eventually_atTop {u g : ℝ → ℝ} (hg : ∀ t, 0 ≤ t → 0 < g t)
    (hga : AntitoneOn g (Ici 0)) (hb : ∀ T, ∃ M, ∀ t ∈ Icc 0 T, u t ≤ M) {C : ℝ}
    (h : ∀ᶠ t in atTop, u t ≤ C * g t) : ∃ C', ∀ t, 0 ≤ t → u t ≤ C' * g t := by
  obtain ⟨T₁, hT₁⟩ := eventually_atTop.1 h
  set T := max T₁ 0 with hT
  obtain ⟨M, hM⟩ := hb T
  have hT0 : 0 ≤ T := le_max_right _ _
  have hgT := hg T hT0
  refine ⟨max C (max M 0 / g T), fun t ht => ?_⟩
  rcases le_total t T with htT | hTt
  · have h2 : g T ≤ g t := hga ht hT0 htT
    calc u t ≤ max M 0 := (hM t ⟨ht, htT⟩).trans (le_max_left _ _)
      _ = max M 0 / g T * g T := by field_simp
      _ ≤ max M 0 / g T * g t :=
        mul_le_mul_of_nonneg_left h2 (div_nonneg (le_max_right _ _) hgT.le)
      _ ≤ max C (max M 0 / g T) * g t :=
        mul_le_mul_of_nonneg_right (le_max_right _ _) (hg t ht).le
  · exact (hT₁ t ((le_max_left _ _).trans hTt)).trans
      (mul_le_mul_of_nonneg_right (le_max_left _ _) (hg t ht).le)

/-- **Bounds from time `0` on, from eventual bounds.**  Let `r ≥ 0` and `d` be
bounded on every `[0, T]`, and `g > 0` antitone on `[0, ∞)`. If from some
time on `r t ≤ C₁ g t` and `d t ≤ K r t ^ p`, with `K, p ≥ 0`, then for some
`C ≥ 0`, `r t ≤ C g t` and `d t ≤ C g t ^ p` for every `t ≥ 0`. -/
theorem exists_le_mul_and_le_mul_rpow_of_eventually {r d g : ℝ → ℝ} {C₁ K p : ℝ}
    (hr : ∀ t, 0 ≤ t → 0 ≤ r t) (hg : ∀ t, 0 ≤ t → 0 < g t) (hga : AntitoneOn g (Ici 0))
    (hK : 0 ≤ K) (hp : 0 ≤ p) (hrb : ∀ T, ∃ M, ∀ t ∈ Icc 0 T, r t ≤ M)
    (hdb : ∀ T, ∃ M, ∀ t ∈ Icc 0 T, d t ≤ M) (hC₁ : ∀ᶠ t in atTop, r t ≤ C₁ * g t)
    (h : ∀ᶠ t in atTop, d t ≤ K * r t ^ p) :
    ∃ C, 0 ≤ C ∧ ∀ t, 0 ≤ t → r t ≤ C * g t ∧ d t ≤ C * g t ^ p := by
  obtain ⟨C₂, hC₂⟩ := exists_forall_le_mul_of_eventually_atTop hg hga hrb hC₁
  have hC₂' : ∀ t, 0 ≤ t → r t ≤ max C₂ 0 * g t := fun t ht =>
    (hC₂ t ht).trans (mul_le_mul_of_nonneg_right (le_max_left _ _) (hg t ht).le)
  have hgp : ∀ t, 0 ≤ t → 0 < g t ^ p := fun t ht => rpow_pos_of_pos (hg t ht) p
  obtain ⟨C₃, hC₃⟩ := exists_forall_le_mul_of_eventually_atTop hgp
    (fun s hs t ht hst => rpow_le_rpow (hg t ht).le (hga hs ht hst) hp) hdb
    (C := K * max C₂ 0 ^ p) ((h.and (eventually_ge_atTop 0)).mono fun t ht => by
      calc d t ≤ K * r t ^ p := ht.1
        _ ≤ K * (max C₂ 0 * g t) ^ p :=
          mul_le_mul_of_nonneg_left (rpow_le_rpow (hr t ht.2) (hC₂' t ht.2) hp) hK
        _ = K * max C₂ 0 ^ p * g t ^ p := by
          rw [mul_rpow (le_max_right _ _) (hg t ht.2).le, mul_assoc])
  refine ⟨max (max C₂ 0) C₃, le_max_of_le_left (le_max_right _ _), fun t ht => ⟨?_, ?_⟩⟩
  · exact (hC₂' t ht).trans (mul_le_mul_of_nonneg_right (le_max_left _ _) (hg t ht).le)
  · exact (hC₃ t ht).trans (mul_le_mul_of_nonneg_right (le_max_right _ _) (hgp t ht).le)

/-- **Exponential decay.**  If `r' ≤ -λ r` on `[T, ∞)`, then
`r t ≤ r T e^{λ T} e^{-λ t}` there: `e^{λ t} r t` decreases. -/
theorem le_mul_exp_of_hasDerivAt {r r' : ℝ → ℝ} {lam T : ℝ}
    (hr : ∀ t, T ≤ t → HasDerivAt r (r' t) t) (h : ∀ t, T ≤ t → r' t ≤ -(lam * r t))
    {t : ℝ} (ht : T ≤ t) : r t ≤ r T * exp (lam * T) * exp (-(lam * t)) := by
  have hd : ∀ s, T ≤ s → HasDerivAt (fun s => exp (lam * s) * r s)
      (exp (lam * s) * (lam * 1) * r s + exp (lam * s) * r' s) s := fun s hs =>
    ((hasDerivAt_id' s).const_mul lam).exp.mul (hr s hs)
  have hanti : AntitoneOn (fun s => exp (lam * s) * r s) (Ici T) :=
    antitoneOn_of_hasDerivWithinAt_nonpos (convex_Ici T)
      (fun s hs => (hd s hs).continuousAt.continuousWithinAt)
      (fun s hs => (hd s (interior_subset hs)).hasDerivWithinAt) fun s hs => by
        nlinarith [mul_le_mul_of_nonneg_left (h s (interior_subset hs)) (exp_pos (lam * s)).le]
  have h1 : exp (lam * t) * r t ≤ exp (lam * T) * r T := hanti self_mem_Ici ht ht
  calc r t = exp (-(lam * t)) * (exp (lam * t) * r t) := by
        rw [← mul_assoc, ← exp_add, neg_add_cancel, exp_zero, one_mul]
    _ ≤ exp (-(lam * t)) * (exp (lam * T) * r T) := mul_le_mul_of_nonneg_left h1 (exp_pos _).le
    _ = r T * exp (lam * T) * exp (-(lam * t)) := by ring

/-- **Polynomial decay.**  If `r > 0` and `r' ≤ -c r ^ (1 + β)` on `[T, ∞)`,
with `β > 0`, then `r T ^ (-β) + β c (t - T) ≤ r t ^ (-β)` there: the
derivative of `r ^ (-β)` is `-β r' r ^ (-β - 1) ≥ β c`. -/
theorem rpow_neg_add_mul_le_of_hasDerivAt {r r' : ℝ → ℝ} {β c T : ℝ} (hβ : 0 < β)
    (hpos : ∀ t, T ≤ t → 0 < r t) (hr : ∀ t, T ≤ t → HasDerivAt r (r' t) t)
    (h : ∀ t, T ≤ t → r' t ≤ -(c * r t ^ (1 + β))) {t : ℝ} (ht : T ≤ t) :
    r T ^ (-β) + β * c * (t - T) ≤ r t ^ (-β) := by
  have hd : ∀ s, T ≤ s → HasDerivAt (fun s => r s ^ (-β) - β * c * s)
      (r' s * (-β) * r s ^ (-β - 1) - β * c * 1) s := fun s hs =>
    ((hr s hs).rpow_const (Or.inl (hpos s hs).ne')).sub ((hasDerivAt_id' s).const_mul (β * c))
  have hmono : MonotoneOn (fun s => r s ^ (-β) - β * c * s) (Ici T) :=
    monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici T)
      (fun s hs => (hd s hs).continuousAt.continuousWithinAt)
      (fun s hs => (hd s (interior_subset hs)).hasDerivWithinAt) fun s hs => by
        have hs' : T ≤ s := interior_subset hs
        have hq : 0 < r s ^ (-β - 1) := rpow_pos_of_pos (hpos s hs') _
        have heq : -(c * r s ^ (1 + β)) * (β * r s ^ (-β - 1)) = -(β * c) := by
          calc _ = -(β * c) * (r s ^ (1 + β) * r s ^ (-β - 1)) := by ring
            _ = -(β * c) := by
              rw [← rpow_add (hpos s hs'), show 1 + β + (-β - 1) = 0 by ring, rpow_zero,
                mul_one]
        nlinarith [mul_le_mul_of_nonneg_right (h s hs') (mul_nonneg hβ.le hq.le)]
  have h1 : r T ^ (-β) - β * c * T ≤ r t ^ (-β) - β * c * t := hmono self_mem_Ici ht ht
  linarith

/-- **Polynomial decay, explicitly.**  If `r > 0` and `r' ≤ -c r ^ (1 + β)` on
`[T, ∞)`, with `T ≥ 0` and `β, c > 0`, then
`r t ≤ (β c / 2) ^ (-1/β) (t + 1) ^ (-1/β)` for `t ≥ 2T + 1`, where
`t - T ≥ (t + 1) / 2`. -/
theorem le_mul_rpow_of_hasDerivAt {r r' : ℝ → ℝ} {β c T : ℝ} (hβ : 0 < β) (hc : 0 < c)
    (hT0 : 0 ≤ T) (hpos : ∀ t, T ≤ t → 0 < r t) (hr : ∀ t, T ≤ t → HasDerivAt r (r' t) t)
    (h : ∀ t, T ≤ t → r' t ≤ -(c * r t ^ (1 + β))) {t : ℝ} (ht : 2 * T + 1 ≤ t) :
    r t ≤ (β * c / 2) ^ (-(1 / β)) * (t + 1) ^ (-(1 / β)) := by
  have hT : T ≤ t := by linarith
  have hgrow := rpow_neg_add_mul_le_of_hasDerivAt hβ hpos hr h hT
  have hpT : 0 < r T ^ (-β) := rpow_pos_of_pos (hpos T le_rfl) _
  have h2 : (t + 1) / 2 ≤ t - T := by linarith
  have hlow : β * c / 2 * (t + 1) ≤ r t ^ (-β) := by
    nlinarith [mul_le_mul_of_nonneg_left h2 (mul_pos hβ hc).le]
  have hc2 : 0 < β * c / 2 := half_pos (mul_pos hβ hc)
  calc r t ≤ (β * c / 2 * (t + 1)) ^ (-(1 / β)) :=
        le_rpow_neg_one_div_of_le_rpow_neg hβ (hpos t hT) (mul_pos hc2 (by linarith)) hlow
    _ = _ := mul_rpow hc2.le (by linarith)

/-- The hypotheses of the theorems above hold for `u = r = d = g = 1/(t + 1)`
with `C = C₁ = K = p = 1`; for `r t = e^{-t}` with `λ = 1` and `T = 0`; and for
`r t = 1/(t + 1)` with `β = 1`, `c = 1/2` and `T = 0`, whose derivative
`-1/(t + 1)²` is at most `-r ^ 2 / 2`, and `0 < 1/2`. -/
example : (∀ t : ℝ, 0 ≤ t → 0 < 1 / (t + 1)) ∧ AntitoneOn (fun t : ℝ => 1 / (t + 1)) (Ici 0) ∧
    (∀ T : ℝ, ∃ M, ∀ t ∈ Icc 0 T, 1 / (t + 1) ≤ M) ∧
    (∀ᶠ t : ℝ in atTop, 1 / (t + 1) ≤ 1 * (1 / (t + 1)) ∧
      1 / (t + 1) ≤ 1 * (1 / (t + 1)) ^ (1 : ℝ)) ∧
    (∀ t : ℝ, 0 ≤ t → HasDerivAt (fun s => exp (-s)) (-exp (-t)) t ∧
      -exp (-t) ≤ -(1 * exp (-t))) ∧
    (0 : ℝ) < 1 / 2 ∧ ∀ t : ℝ, 0 ≤ t → HasDerivAt (fun s : ℝ => 1 / (s + 1)) (-1 / (t + 1) ^ 2) t ∧
      -1 / (t + 1) ^ 2 ≤ -(1 / 2 * (1 / (t + 1)) ^ (1 + 1 : ℝ)) := by
  refine ⟨fun t ht => by positivity, fun a ha b _ hab => ?_, fun T => ⟨1, fun t ht => ?_⟩,
    Eventually.of_forall fun t => ⟨(one_mul _).ge, by rw [rpow_one, one_mul]⟩,
    fun t _ => ⟨by simpa using (hasDerivAt_neg t).exp, by rw [one_mul]⟩, by norm_num,
    fun t ht => ⟨?_, ?_⟩⟩
  · exact one_div_le_one_div_of_le (by linarith [mem_Ici.1 ha]) (by linarith)
  · exact (div_le_one₀ (by linarith [ht.1])).2 (by linarith [ht.1])
  · have h := ((hasDerivAt_id' t).add_const 1).inv (by positivity)
    simp only [one_div]
    convert h using 1
  · have hX : 0 < 1 / (t + 1) ^ 2 := by positivity
    rw [show (1 + 1 : ℝ) = (2 : ℕ) by norm_num, rpow_natCast, div_pow, one_pow, neg_div]
    linarith

end Transformer.Landscape.SaddleAvoidance

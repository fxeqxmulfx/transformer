module

public import Mathlib.Analysis.Calculus.Deriv.Slope
public import Mathlib.Analysis.Calculus.Deriv.Pow
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Topology.Order.Compact
public import Mathlib.Tactic

/-!
# A finite first-crossing argument for the causal gap barrier

ArXiv:2411.04990v2, §5, proof of `lemma:meta`, the comparison after
`-Δ̇ ≤ 2s_k h(Δ)`. A closest predecessor can change with time, so the
argument below keeps every predecessor gap and chooses the first time
any of them meets its barrier. No derivative of a selected minimum is assumed.

The scalar endpoint slope calculation uses the same Mathlib slope API
as `Transformer.Modes.deriv_nonneg_of_isMaxOn_Icc_right`, with the order
reversed for a minimum. The finite comparison itself is proved here.
-/

@[expose] public section

open Filter Set Topology

namespace Transformer.Causal

/-- At a right-endpoint minimum the derivative is nonpositive.
This is the endpoint step in the first-crossing comparison of
arXiv:2411.04990v2, §5, proof of `lemma:meta`. -/
theorem deriv_nonpos_of_isMinOn_Icc_right {f : ℝ → ℝ} {a b d : ℝ}
    (hab : a < b) (hmin : IsMinOn f (Icc a b) b) (hd : HasDerivAt f d b) : d ≤ 0 := by
  have hlim := hd.tendsto_slope.mono_left (nhdsLT_le_nhdsNE b)
  have hineq : ∀ᶠ x in 𝓝[<] b, slope f b x ≤ 0 := by
    filter_upwards [Icc_mem_nhdsLT hab, self_mem_nhdsWithin] with x hx hx'
    rw [slope_def_field]
    exact div_nonpos_of_nonneg_of_nonpos (sub_nonneg.mpr (hmin hx)) (sub_nonpos.mpr hx'.le)
  exact le_of_tendsto hlim hineq

/-- All endpoint inputs are satisfied by `f(t)=-t` on `[0,1]`. -/
example : (0 : ℝ) < 1 ∧ IsMinOn (fun t : ℝ => -t) (Icc 0 1) 1 ∧
    HasDerivAt (fun t : ℝ => -t) (-1) 1 := by
  refine ⟨zero_lt_one, fun t ht => ?_, ?_⟩
  · change -(1 : ℝ) ≤ -t
    linarith [ht.2]
  · exact hasDerivAt_neg 1

/-- **A finite family cannot first cross a boundary with positive derivative.**
Each scalar path starts positive. When all paths are nonnegative and one
is zero, its derivative is positive. Then every path remains positive on
the whole finite horizon.

Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`, the first-crossing
comparison for the moving lower bound on the center's predecessor gaps. -/
theorem finite_gap_barrier {ι : Type*} (I : Finset ι) (f df : ι → ℝ → ℝ) (T : ℝ)
    (hf : ∀ i ∈ I, ∀ t : ℝ, HasDerivAt (f i) (df i t) t)
    (hinit : ∀ i ∈ I, 0 < f i 0)
    (hboundary : ∀ t ∈ Icc (0 : ℝ) T, (∀ i ∈ I, 0 ≤ f i t) →
      ∀ i ∈ I, f i t = 0 → 0 < df i t) :
    ∀ t ∈ Icc (0 : ℝ) T, ∀ i ∈ I, 0 < f i t := by
  intro t ht i hi
  by_contra hbad
  let S : Set ℝ := Icc 0 T ∩ ⋃ j ∈ I, {s : ℝ | f j s ≤ 0}
  have hclosed : IsClosed (⋃ j ∈ I, {s : ℝ | f j s ≤ 0}) := by
    apply isClosed_biUnion_finset
    intro j hj
    exact isClosed_le (continuous_iff_continuousAt.mpr fun s => (hf j hj s).continuousAt)
      continuous_const
  have hcompact : IsCompact S := isCompact_Icc.inter_right hclosed
  have hnonempty : S.Nonempty := ⟨t, ht, mem_iUnion₂.mpr ⟨i, hi, not_lt.mp hbad⟩⟩
  obtain ⟨t₀, hfirst⟩ := hcompact.exists_isLeast hnonempty
  have ht₀ : t₀ ∈ Icc (0 : ℝ) T := hfirst.1.1
  obtain ⟨i₀, hi₀, hbad₀⟩ := mem_iUnion₂.mp hfirst.1.2
  have ht₀pos : 0 < t₀ := by
    by_contra h
    have hz : t₀ = 0 := by linarith [ht₀.1]
    rw [hz] at hbad₀
    exact not_le_of_gt (hinit i₀ hi₀) hbad₀
  have hprior : ∀ s : ℝ, 0 ≤ s → s < t₀ → ∀ j ∈ I, 0 < f j s := by
    intro s hs0 hst j hj
    by_contra hsbad
    have hs : s ∈ S := ⟨⟨hs0, hst.le.trans ht₀.2⟩,
      mem_iUnion₂.mpr ⟨j, hj, not_lt.mp hsbad⟩⟩
    exact not_le_of_gt hst (hfirst.2 hs)
  have hnonneg : ∀ j ∈ I, 0 ≤ f j t₀ := by
    intro j hj
    have hlim := (hf j hj t₀).continuousAt.tendsto.mono_left
      (nhdsWithin_le_nhds : 𝓝[<] t₀ ≤ 𝓝 t₀)
    apply ge_of_tendsto hlim
    filter_upwards [Icc_mem_nhdsLT ht₀pos, self_mem_nhdsWithin] with s hs hst
    exact (hprior s hs.1 hst j hj).le
  have hzero : f i₀ t₀ = 0 := le_antisymm hbad₀ (hnonneg i₀ hi₀)
  have hmin : IsMinOn (f i₀) (Icc 0 t₀) t₀ := by
    intro s hs
    change f i₀ t₀ ≤ f i₀ s
    rw [hzero]
    rcases lt_or_eq_of_le hs.2 with hst | hst
    · exact (hprior s hs.1 hst i₀ hi₀).le
    · rw [hst, hzero]
  have hdf := deriv_nonpos_of_isMinOn_Icc_right ht₀pos hmin (hf i₀ hi₀ t₀)
  exact not_lt_of_ge hdf (hboundary t₀ ht₀ hnonneg i₀ hi₀ hzero)

/-- The complete finite-barrier hypotheses hold for a nonempty family of
increasing affine paths. Their derivative is positive at every time. -/
example : (∀ i ∈ ({0} : Finset ℕ), ∀ t : ℝ,
      HasDerivAt ((fun (_ : ℕ) (s : ℝ) => s + 1) i) ((fun (_ : ℕ) (_ : ℝ) => 1) i t) t) ∧
    (∀ i ∈ ({0} : Finset ℕ), 0 < (fun (_ : ℕ) (s : ℝ) => s + 1) i 0) ∧
    (∀ t ∈ Icc (0 : ℝ) 1,
      (∀ i ∈ ({0} : Finset ℕ), 0 ≤ (fun (_ : ℕ) (s : ℝ) => s + 1) i t) →
      ∀ i ∈ ({0} : Finset ℕ), (fun (_ : ℕ) (s : ℝ) => s + 1) i t = 0 →
        0 < (fun (_ : ℕ) (_ : ℝ) => (1 : ℝ)) i t) := by
  refine ⟨fun _ _ t => (hasDerivAt_id t).add_const 1, fun _ _ => by norm_num,
    fun _ _ _ _ _ _ => zero_lt_one⟩

/-- **An affine moving barrier bounds every squared gap below.** The
boundary derivative is compared with the derivative of
`(r+L(T-t))²`. Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`,
the moving comparison following `-Δ̇ ≤ 2s_k h(Δ)`. -/
theorem finite_affine_gap_barrier {ι : Type*} (I : Finset ι)
    (g dg : ι → ℝ → ℝ) (T r L : ℝ) (hr : 0 ≤ r) (hL : 0 ≤ L)
    (hg : ∀ i ∈ I, ∀ t : ℝ, HasDerivAt (g i) (dg i t) t)
    (hinit : ∀ i ∈ I, (r + L * T) ^ 2 < g i 0)
    (hboundary : ∀ t ∈ Icc (0 : ℝ) T,
      (∀ i ∈ I, (r + L * (T - t)) ^ 2 ≤ g i t) →
      ∀ i ∈ I, g i t = (r + L * (T - t)) ^ 2 →
        -2 * L * (r + L * (T - t)) < dg i t) :
    ∀ t ∈ Icc (0 : ℝ) T, ∀ i ∈ I, r ^ 2 < g i t := by
  let b (t : ℝ) := r + L * (T - t)
  have hb (t : ℝ) : HasDerivAt b (-L) t := by
    simpa only [b, id_eq, mul_neg_one] using
      ((hasDerivAt_id t).const_sub T).const_mul L |>.const_add r
  have hf (i : ι) (hi : i ∈ I) (t : ℝ) :
      HasDerivAt (fun s => g i s - b s ^ 2) (dg i t + 2 * L * b t) t := by
    convert (hg i hi t).sub ((hb t).pow 2) using 1
    ring
  have hpos := finite_gap_barrier I (fun i t => g i t - b t ^ 2)
    (fun i t => dg i t + 2 * L * b t) T hf (fun i hi => by
      simpa only [b, sub_zero, sub_pos] using hinit i hi) ?_
  · intro t ht i hi
    have h := hpos t ht i hi
    have hbt : r ≤ b t := by dsimp [b]; nlinarith [mul_nonneg hL (sub_nonneg.mpr ht.2)]
    have hs : r ^ 2 ≤ b t ^ 2 := by nlinarith
    linarith
  · intro t ht hall i hi heq
    have hbd := hboundary t ht (fun j hj => by
      simpa only [b] using sub_nonneg.mp (hall j hj)) i hi (by
        simpa only [b] using sub_eq_zero.mp heq)
    change 0 < dg i t + 2 * L * (r + L * (T - t))
    linarith

/-- The affine-barrier hypotheses are realized by the increasing scalar
path `g(t)=t+2`, with `r=1`, `L=0`, and a nonempty singleton index set. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 0 ∧
    (∀ i ∈ ({0} : Finset ℕ), ∀ t : ℝ,
      HasDerivAt ((fun (_ : ℕ) (s : ℝ) => s + 2) i) ((fun (_ : ℕ) (_ : ℝ) => 1) i t) t) ∧
    (∀ i ∈ ({0} : Finset ℕ), (1 + 0 * 1 : ℝ) ^ 2 <
      (fun (_ : ℕ) (s : ℝ) => s + 2) i 0) ∧
    (∀ t ∈ Icc (0 : ℝ) 1,
      (∀ i ∈ ({0} : Finset ℕ), (1 + 0 * (1-t) : ℝ) ^ 2 ≤
        (fun (_ : ℕ) (s : ℝ) => s + 2) i t) →
      ∀ i ∈ ({0} : Finset ℕ), (fun (_ : ℕ) (s : ℝ) => s + 2) i t = (1 + 0 * (1-t) : ℝ) ^ 2 →
        -2 * 0 * (1 + 0 * (1-t) : ℝ) < (fun (_ : ℕ) (_ : ℝ) => (1 : ℝ)) i t) := by
  refine ⟨by norm_num, le_rfl, fun _ _ t => (hasDerivAt_id t).add_const 2,
    fun _ _ => by norm_num, fun _ _ _ _ _ _ => by norm_num⟩

end Transformer.Causal

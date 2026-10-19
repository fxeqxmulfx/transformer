module

public import Transformer.Interpolation.Section3_BallParameters
public import Transformer.Interpolation.Section3_ChainMass
public import Transformer.Interpolation.Section1_NeuralPhaseLaw

/-!
# Actual original neural controls for a finite ball chain

ArXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`.

An initial zero phase supplies the base case with one constant piece.
Each induction step joins the prefix to one actual two-neuron ball
transport, at half the requested horizon. The piece count increases by
one, the original integrated particle equation joins exactly, and the
fixed complement contains the complement of the enlarged ball union.

The construction retains the true constant final phase and global weak
existence and transport for every probability. Concentration parameters
are chosen for the specified initial law. The mass induction uses the
fresh portion's unchanged mass and the transported overlap estimate.

The retained fraction is `ofReal (1 - ε)` before taking its power. This
agrees with the source for `ε ≤ 1` and gives zero for larger positive
errors when the chain has at least one step. The source's unclamped real
power is false for `ε = 3`, `K = 2`.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The original controls, actual flow, all-law weak transport and chain
mass bound are constructed jointly with at most `K+1` pieces. Auxiliary
for arXiv:2411.04551v3, Appendix, `lem: tubular.mass.movement`; the
nonnegative retained fraction is clamped before taking its power. -/
theorem exists_chain_neural_controls (hd : 2 ≤ d) (i j : Fin d) (hij : i ≠ j)
    (K : ℕ) (z : ℕ → SSphere d) (R : ℕ → ℝ)
    (hadj : ∀ k : ℕ, 1 ≤ k → k ≤ K →
      (Metric.ball (z k) (R k) ∩ Metric.ball (z (k - 1)) (R (k - 1))).Nonempty)
    (hdisj : ∀ k k' : ℕ, k ≤ K → k' ≤ K → k + 2 ≤ k' →
      Disjoint (Metric.ball (z k) (R k)) (Metric.ball (z k') (R k')))
    (ε T : ℝ) (hε : 0 < ε) (hT : 0 < T) (ν : Perspective.ProbSphere d) :
    ∃ (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
      (W U : ℝ → ParamMatrix d) (a : ℝ → EucSpace d) (φ : ℝ → SSphere d → SSphere d),
      (∀ s : ℝ, T ≤ s → neuralParams d W U a s = twoRidgeParams i j p u b q v c) ∧
      PiecewiseConstant d (neuralParams d W U a) T (K + 1) ∧
      IsNeuralFlow d W U a φ ∧
      (∀ t, (∃ L : NNReal, LipschitzWith L (φ t)) ∧ Function.Bijective (φ t)) ∧
      IsTransportOutside d φ (⋃ k ∈ Iic K, Metric.ball (z k) (R k)) T ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d (neuralParams d W U a) μ) ∧
      (∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d (neuralParams d W U a) μ →
        ∀ t, (μ t : Measure (SSphere d)) = Measure.map (φ t) (μ 0 : Measure (SSphere d))) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ν → cauchyPB d (neuralParams d W U a) μ →
        ENNReal.ofReal (1 - ε) ^ K * (ν : Measure (SSphere d))
          (⋃ k ∈ Iic K, Metric.ball (z k) (R k)) ≤ (μ T : Measure (SSphere d)) (Metric.ball (z K) (R K)) := by
  induction K generalizing T with
  | zero =>
    let θ := twoRidgeParams i j (0 : EucSpace d) 0 0 0 0 0
    let W : ℝ → ParamMatrix d := fun _ => θ.W
    let U : ℝ → ParamMatrix d := fun _ => θ.U
    let a : ℝ → EucSpace d := fun _ => θ.b
    let φ (t : ℝ) (x : SSphere d) := twoRidgeFlow (0 : EucSpace d) 0 0 0 0 0 x t
    have hfix (x : SSphere d) (t : ℝ) : φ t x = x := twoRidgeFlow_zero_data x t
    refine ⟨0, 0, 0, 0, 0, 0, W, U, a, φ, fun _ _ => rfl,
      piecewiseConstant_constant θ T hT.le, isNeuralFlow_twoRidgeFlow i j hij _ _ _ _ _ _,
      fun t => ⟨exists_lipschitzWith_twoRidgeFlow _ _ _ _ _ _ t,
        bijective_twoRidgeFlow _ _ _ _ _ _ t⟩,
      twoRidgeFlow_isTransportOutside _ _ _ _ _ _ _ (fun x _ t => hfix x t) T, ?_, ?_, ?_⟩
    · intro ρ
      exact ⟨twoRidgeLaw 0 0 0 0 0 0 ρ, twoRidgeLaw_zero _ _ _ _ _ _ _,
        cauchyPB_twoRidgeLaw i j hij _ _ _ _ _ _ _⟩
    · intro μ hμ t
      rw [cauchyPB_twoRidge_zero_data i j hij μ hμ t]
      have he : φ t = id := funext fun x => hfix x t
      rw [he, Measure.map_id]
    · intro μ h0 hμ
      rw [cauchyPB_twoRidge_zero_data i j hij μ hμ T, h0]
      simp only [pow_zero, one_mul]
      apply measure_mono
      intro x hx
      obtain ⟨k, hx⟩ := mem_iUnion.mp hx
      obtain ⟨hk, hx⟩ := mem_iUnion.mp hx
      have he : k = 0 := by change k ≤ 0 at hk; omega
      exact he ▸ hx
  | succ K ih =>
    let S := T / 2
    have hS : 0 < S := by dsimp [S]; positivity
    have hST : S ≤ T := by dsimp [S]; linarith
    have hTS : 0 < T - S := by dsimp [S]; linarith
    obtain ⟨p, u, b, q, v, c, W, U, a, φ, htail, hpc, hflow, hreg, hout, hex, hmap, hmass⟩ :=
      ih (fun k hk hkK => hadj k hk (by omega))
        (fun k k' hk hk' hkk' => hdisj k k' (by omega) (by omega) hkk') S hS
    obtain ⟨η, hη0, hη⟩ := hex ν
    have hmeet : (Metric.ball (z K) (R K) ∩ Metric.ball (z (K + 1)) (R (K + 1))).Nonempty := by
      obtain ⟨x, hx, hy⟩ := hadj (K + 1) (by omega) le_rfl
      exact ⟨x, by simpa using hy, hx⟩
    obtain ⟨p', u', b', q', v', c', hfix', hmove⟩ := exists_twoRidge_overlap_mass_parameters hd
      (z K) (z (K + 1)) (R K) (R (K + 1)) hmeet ε (T - S) hε hTS.ne' (η S)
    let θ' := twoRidgeParams i j p' u' b' q' v' c'
    let W' : ℝ → ParamMatrix d := concatenate S W (fun _ => θ'.W)
    let U' : ℝ → ParamMatrix d := concatenate S U (fun _ => θ'.U)
    let a' : ℝ → EucSpace d := concatenate S a (fun _ => θ'.b)
    let ψ (t : ℝ) (x : SSphere d) := twoRidgeFlow p' u' b' q' v' c' x t
    let Φ := concatenateFlow S φ ψ
    have hθ : neuralParams d W' U' a' = concatenate S (neuralParams d W U a) (fun _ => θ') :=
      neuralParams_concatenate S _ _ _ _ _ _
    have hu := cauchyPB_neural_unique_of_map W U a φ hmap
    have hnewex (ρ : Perspective.ProbSphere d) := exists_unique_cauchyPB_concatenate_twoRidge
      i j hij p u b q v c S hS (neuralParams d W U a) htail hex hu p' u' b' q' v' c' ρ
    have hmS : Measurable (φ S) := (hreg S).1.choose_spec.continuous.measurable
    have hnewmap (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (neuralParams d W' U' a') μ)
        (t : ℝ) : (μ t : Measure (SSphere d)) = Measure.map (Φ t) (μ 0 : Measure (SSphere d)) := by
      rw [hθ] at hμ
      exact cauchyPB_neural_concatenate_twoRidge_map i j hij p u b q v c S hS W U a φ
        htail hex hu hmap hmS p' u' b' q' v' c' μ hμ t
    refine ⟨p', u', b', q', v', c', W', U', a', Φ, ?_, ?_, ?_, ?_, ?_, ?_, hnewmap, ?_⟩
    · intro s hs
      rw [hθ, concatenate_of_ge S _ _ s (hST.trans hs)]
    · exact piecewiseConstant_neural_concatenate_constant W U a θ'.W θ'.U θ'.b S T (K + 1) hST hpc
    · exact isNeuralFlow_concatenate S hS W U _ _ a _ φ ψ hflow
        (isNeuralFlow_twoRidgeFlow i j hij p' u' b' q' v' c')
    · exact concatenateFlow_lipschitz_bijective S φ ψ hreg
        (fun t => ⟨exists_lipschitzWith_twoRidgeFlow _ _ _ _ _ _ t,
          bijective_twoRidgeFlow _ _ _ _ _ _ t⟩)
    · have ho := isTransportOutside_concatenate S T hS.le φ ψ
        (⋃ k ∈ Iic K, Metric.ball (z k) (R k)) (Metric.ball (z K) (R K)) hout
        (twoRidgeFlow_isTransportOutside p' u' b' q' v' c' _ hfix' (T - S))
      intro t ht
      obtain ⟨hL, hb, hf⟩ := ho t ht
      refine ⟨hL, hb, fun x hx => hf x ?_⟩
      intro hh
      apply hx
      rcases hh with hh | hh
      · obtain ⟨k, hh⟩ := mem_iUnion.mp hh
        obtain ⟨hk, hh⟩ := mem_iUnion.mp hh
        exact mem_iUnion.mpr ⟨k, mem_iUnion.mpr ⟨by change k ≤ K + 1; change k ≤ K at hk; omega, hh⟩⟩
      · exact mem_iUnion.mpr ⟨K, mem_iUnion.mpr ⟨by change K ≤ K + 1; omega, hh⟩⟩
    · intro ρ
      obtain ⟨μ, hμ, _⟩ := hnewex ρ
      exact ⟨μ, hμ.1, by rw [hθ]; exact hμ.2⟩
    · intro μ h0 hμ
      have heT : (μ T : Measure (SSphere d)) = Measure.map (ψ (T - S)) (η S : Measure (SSphere d)) := by
        rw [hnewmap μ hμ T, h0]
        have heΦ : Φ T = ψ (T - S) ∘ φ S := by
          funext x
          exact concatenateFlow_of_ge S φ ψ T hST x
        rw [heΦ, ← Measure.map_map (measurable_twoRidgeFlow_initial _ _ _ _ _ _ _) hmS,
          ← hη0, ← hmap η hη S]
      rw [heT]
      apply chain_mass_step (ν : Measure (SSphere d)) (η S : Measure (SSphere d)) (ψ (T - S))
        (bijective_twoRidgeFlow p' u' b' q' v' c' _).injective
        (measurable_twoRidgeFlow_initial _ _ _ _ _ _ _) (fun k => Metric.ball (z k) (R k)) K
        (fun _ => Metric.isOpen_ball.measurableSet)
        (fun k hk => hdisj k (K + 1) (by omega) le_rfl hk)
        (fun x hx => hfix' x hx _) (ENNReal.ofReal (1 - ε))
        (by exact (ENNReal.ofReal_le_one).mpr (by linarith))
        (hmass η hη0 hη) ?_ (by simpa only [coe_twoRidgeLaw] using hmove)
      rw [hmap η hη S, hη0]
      exact map_chain_fresh_mass (ν : Measure (SSphere d)) (φ S) (hreg S).2.injective hmS
        (fun k => Metric.ball (z k) (R k)) K (fun _ => Metric.isOpen_ball.measurableSet)
        (fun k hk => hdisj k (K + 1) (by omega) le_rfl hk) (hout S ⟨hS.le, le_rfl⟩).2.2

/-- A genuine two-ball circle chain, distinct hidden coordinates,
positive accuracy and time, and an atomic initial law jointly satisfy
all construction hypotheses. -/
example : ∃ (z : ℕ → SSphere 2) (R : ℕ → ℝ) (ν : Perspective.ProbSphere 2),
    2 ≤ 2 ∧ (0 : Fin 2) ≠ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (∀ k : ℕ, 1 ≤ k → k ≤ 1 →
      (Metric.ball (z k) (R k) ∩ Metric.ball (z (k - 1)) (R (k - 1))).Nonempty) ∧
    (∀ k k' : ℕ, k ≤ 1 → k' ≤ 1 → k + 2 ≤ k' →
      Disjoint (Metric.ball (z k) (R k)) (Metric.ball (z k') (R k'))) ∧
    (ν : Measure (SSphere 2)) univ = 1 := by
  let p : SSphere 2 := basePoint 1
  refine ⟨(fun _ => p), (fun _ => 1), Perspective.diracProb 2 p,
    le_rfl, by norm_num, zero_lt_one, zero_lt_one,
    fun _ _ _ => ⟨p, by simp, by simp⟩, ?_, IsProbabilityMeasure.measure_univ⟩
  intro k k' hk hk' hkk'
  omega

end Transformer.Interpolation

module

public import Transformer.ALM.SoftmaxValue

/-! A real scalar memory address with a learned log-write weight lifts to
the planar ALM maximum-inner-product query. Soft value error is controlled
by the measured gap. This adapts the “2D attention” and “LookUp gates”
sections of Percepta's Append-only Lookup Machine to learned real addresses.
-/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.AddressGap
open scoped BigOperators

def distanceScore (q a bias : ℝ) := bias - (q-a)^2
def point (a bias : ℝ) : ℝ × ℝ := (a,bias-a^2)
def planar (q : ℝ) (p : ℝ × ℝ) := 2*q*p.1+p.2

/-- The query-only offset cancels in every comparison, preserving score
gaps and the latest-insertion tie rule exactly in real arithmetic. -/
theorem same_gap (q a b u v : ℝ) :
    planar q (point a u) - planar q (point b v) = distanceScore q a u - distanceScore q b v := by
  unfold planar point distanceScore
  ring

/-- A score gap observed in the planar index controls the distance from
the soft distance-based vector read to the hard indexed payload. Each
write's log strength may vary independently of its address. -/
theorem value_error {n : ℕ} [Nonempty (Fin n)] {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (q β δ C : ℝ) (nonneg : 0≤β) (a b : Fin n → ℝ) (values : Fin n → E) (best : Fin n)
    (bounded : ∀ i, ‖values i-values best‖ ≤ C)
    (gap : ∀ i, i≠best → planar q (point (a i) (b i)) + δ ≤ planar q (point (a best) (b best))) :
    let score := fun i => distanceScore q (a i) (b i)
    ‖(∑ i, (Real.exp (β*score i) / ∑ k, Real.exp (β*score k)) • values i) - values best‖ ≤
      (((n:ℝ)-1)*Real.exp (-(β*δ)))*C := by
  dsimp only
  apply ALM.softmax_output_close β _ values best C _ bounded
  apply ALM.softmax_winner_ge β nonneg (fun i : Fin n => distanceScore q (a i) (b i)) best δ
  intro i other
  have eq := same_gap q (a best) (a i) (b best) (b i)
  linarith [gap i other]

example : let a : Fin 2 → ℝ := ![0,2]
    let v : Fin 2 → ℝ := ![1,4]
    let score := fun i => distanceScore 0 (a i) 0
    ‖(∑ i, (Real.exp (1*score i) / ∑ k, Real.exp (1*score k)) • v i) - v 0‖ ≤
      (((2:ℝ)-1)*Real.exp (-(1*4)))*3 := by
  apply value_error 0 1 4 3 (by norm_num) ![0,2] (fun _=>0) ![1,4] 0
  · intro i
    fin_cases i <;> norm_num
  · intro i h
    fin_cases i <;> norm_num [point,planar] at *

end Transformer.Phasor.TripleTransport.AddressGap

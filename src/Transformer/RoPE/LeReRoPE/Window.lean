module

public import Transformer.RoPE.ReRoPE.Section2_Fusion
public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Mul

/-!
# LeReRoPE: a learned window and slope

In no paper. ReRoPE clamps a pair's distance `d` at the window `w`; Leaky ReRoPE lets it grow with
the slope `1/k` beyond it (Su, kexue.fm/archives/9708, 融合 (Fusion);
`Transformer.RoPE.ReRoPE.rectifiedPosition`, `Transformer.RoPE.ReRoPE.leakyPosition`). Here a head
learns both: the distance `d` sits at `p(d) = min(d, w) + s·max(d − w, 0)`.

* `windowPosition_zero`, `windowPosition_inv`, `windowPosition_one`: the slope `s = 0` is ReRoPE,
  `s = 1/k` Leaky ReRoPE and `s = 1` RoPE;
* `windowPosition_of_le`, `windowPosition_of_ge`: `p(d) = d` inside the window and
  `w + s(d − w)` beyond it;
* `windowPosition_sub_of_ge`: two distances beyond the window are `s(d − d')` apart, whatever `w`;
  only the slope separates far pairs. `windowPosition_strictMono`: a slope `s > 0` keeps every
  two distances apart, while `s = 0` puts every far distance at `w`;
* `hasDerivAt_windowPosition_slope`: `∂p/∂s = max(d − w, 0)`, a far distance's excess over the
  window;
* `hasDerivAt_windowPosition_window_of_lt`, `hasDerivAt_windowPosition_window_of_gt`:
  `∂p/∂w = 1 − s` at every distance beyond the window and `0` inside it. The window moves all far
  pairs alike; `hasDerivAt_windowPosition_gap`: the gap of two far distances has the window
  derivative `0` and the slope derivative `d − d'`;
* `route_eq_windowPosition`: one pass computes the map, as it computes Leaky ReRoPE's: a pair
  `t − j < w` takes the query at `t` and the key at `j`, a far pair the query at `w + s(t − w)`
  and the key at `s j`.
-/

@[expose] public section

open Filter Topology

namespace Transformer.RoPE.LeReRoPE

open Transformer.RoPE.ReRoPE

/-- A head's position at the distance `d` under its window `w` and slope `s`:
`min(d, w) + s·max(d − w, 0)`. -/
noncomputable def windowPosition (w s d : ℝ) : ℝ :=
  min d w + s * max (d - w) 0

/-- Inside the window the position is the distance. -/
theorem windowPosition_of_le {w d : ℝ} (s : ℝ) (hd : d ≤ w) : windowPosition w s d = d := by
  rw [windowPosition, min_eq_left hd, max_eq_right (sub_nonpos.2 hd), mul_zero, add_zero]

example : windowPosition 16 (1 / 16) 10 = 10 :=
  windowPosition_of_le _ (by norm_num)

/-- Beyond the window the position grows with the slope `s`: `w + s(d − w)`. -/
theorem windowPosition_of_ge {w d : ℝ} (s : ℝ) (hd : w ≤ d) :
    windowPosition w s d = w + s * (d - w) := by
  rw [windowPosition, min_eq_right hd, max_eq_left (sub_nonneg.2 hd)]

example : windowPosition 16 (1 / 16) 32 = 16 + 1 / 16 * (32 - 16) :=
  windowPosition_of_ge _ (by norm_num)

/-- The slope `0` is ReRoPE's clamp `min(d, w)`. -/
theorem windowPosition_zero (w d : ℝ) : windowPosition w 0 d = rectifiedPosition w d := by
  rw [windowPosition, zero_mul, add_zero, rectifiedPosition]

/-- The slope `1/k` is Leaky ReRoPE's position, for every `k`. -/
theorem windowPosition_inv (w k d : ℝ) : windowPosition w k⁻¹ d = leakyPosition w k d := by
  rw [leakyPosition]
  split_ifs with hd
  · exact windowPosition_of_le _ hd.le
  · rw [windowPosition_of_ge _ (not_lt.1 hd), div_eq_inv_mul]

/-- The slope `1` is RoPE: every distance is its own position, whatever the window. -/
theorem windowPosition_one (w d : ℝ) : windowPosition w 1 d = d := by
  rcases le_total d w with hd | hd
  · exact windowPosition_of_le 1 hd
  · rw [windowPosition_of_ge 1 hd]
    ring

/-- **Only the slope separates far pairs**: two distances beyond the window are `s(d − d')`
apart, whatever the window. -/
theorem windowPosition_sub_of_ge {w d d' : ℝ} (s : ℝ) (hd : w ≤ d) (hd' : w ≤ d') :
    windowPosition w s d - windowPosition w s d' = s * (d - d') := by
  rw [windowPosition_of_ge s hd, windowPosition_of_ge s hd']
  ring

/-- The two inputs of branch amplitudes, `64` and `32` back, under the window `16`. -/
example (s : ℝ) : windowPosition 16 s 64 - windowPosition 16 s 32 = s * (64 - 32) :=
  windowPosition_sub_of_ge s (by norm_num) (by norm_num)

/-- A slope `s > 0` keeps every two distances apart. -/
theorem windowPosition_strictMono {w s : ℝ} (hs : 0 < s) : StrictMono (windowPosition w s) := by
  intro a b hab
  rcases le_or_gt b w with hb | hb
  · rw [windowPosition_of_le s (hab.le.trans hb), windowPosition_of_le s hb]
    exact hab
  · rw [windowPosition_of_ge s hb.le]
    rcases le_or_gt a w with ha | ha
    · rw [windowPosition_of_le s ha]
      nlinarith [mul_pos hs (sub_pos.2 hb)]
    · rw [windowPosition_of_ge s ha.le]
      nlinarith [mul_lt_mul_of_pos_left (sub_lt_sub_right hab w) hs]

example : StrictMono (windowPosition 16 (1 / 16)) :=
  windowPosition_strictMono (by norm_num)

/-- **The slope's derivative is the excess over the window**: `∂p/∂s = max(d − w, 0)`. -/
theorem hasDerivAt_windowPosition_slope (w d s : ℝ) :
    HasDerivAt (fun s => windowPosition w s d) (max (d - w) 0) s := by
  have h := ((hasDerivAt_id' (x := s)).mul_const (max (d - w) 0)).const_add (min d w)
  rw [one_mul] at h
  exact h

/-- **The window moves every far pair alike**: `∂p/∂w = 1 − s` at a distance beyond it. -/
theorem hasDerivAt_windowPosition_window_of_lt {w d : ℝ} (s : ℝ) (hd : w < d) :
    HasDerivAt (fun w => windowPosition w s d) (1 - s) w := by
  have h : HasDerivAt (fun w => w + s * (d - w)) (1 - s) w := by
    convert (hasDerivAt_id' (x := w)).add (((hasDerivAt_id' (x := w)).const_sub d).const_mul s)
      using 1
    ring
  refine h.congr_of_eventuallyEq ?_
  filter_upwards [eventually_lt_nhds hd] with ω hω using windowPosition_of_ge s hω.le

example : HasDerivAt (fun w => windowPosition w (1 / 16) 64) (1 - 1 / 16) 16 :=
  hasDerivAt_windowPosition_window_of_lt _ (by norm_num)

/-- Inside the window the position does not depend on it: `∂p/∂w = 0`. -/
theorem hasDerivAt_windowPosition_window_of_gt {w d : ℝ} (s : ℝ) (hd : d < w) :
    HasDerivAt (fun w => windowPosition w s d) 0 w := by
  refine (hasDerivAt_const w d).congr_of_eventuallyEq ?_
  filter_upwards [eventually_gt_nhds hd] with ω hω using windowPosition_of_le s hω.le

example : HasDerivAt (fun w => windowPosition w (1 / 16) 10) 0 16 :=
  hasDerivAt_windowPosition_window_of_gt _ (by norm_num)

/-- **The window does not see the gap**: for two distances beyond the window, the gap
`p(d) − p(d')` has the window derivative `0` and the slope derivative `d − d'`. -/
theorem hasDerivAt_windowPosition_gap {w s d d' : ℝ} (hd : w < d) (hd' : w < d') :
    HasDerivAt (fun ω => windowPosition ω s d - windowPosition ω s d') 0 w ∧
      HasDerivAt (fun σ => windowPosition w σ d - windowPosition w σ d') (d - d') s := by
  refine ⟨?_, ?_⟩
  · convert (hasDerivAt_windowPosition_window_of_lt s hd).sub
      (hasDerivAt_windowPosition_window_of_lt s hd') using 1
    ring
  · convert (hasDerivAt_windowPosition_slope w d s).sub
      (hasDerivAt_windowPosition_slope w d' s) using 1
    rw [max_eq_left (sub_nonneg.2 hd.le), max_eq_left (sub_nonneg.2 hd'.le)]
    ring

/-- Branch amplitudes' inputs `32` and `64` back, under the window `16`: the window's derivative
of their gap is `0`, the slope's `32 − 64`. -/
example (s : ℝ) :
    HasDerivAt (fun ω => windowPosition ω s 32 - windowPosition ω s 64) 0 16 ∧
      HasDerivAt (fun σ => windowPosition 16 σ 32 - windowPosition 16 σ 64) (32 - 64) s :=
  hasDerivAt_windowPosition_gap (by norm_num) (by norm_num)

/-- **One pass computes the map**: a pair `t − j < w` takes the query rotated to `t` and the key
to `j`, a far pair the query rotated to `w + s(t − w)` and the key to `s j`; the difference is the
pair's position. -/
theorem route_eq_windowPosition (w s t j : ℝ) :
    (if t - j < w then t - j else (w + s * (t - w)) - s * j) = windowPosition w s (t - j) := by
  split_ifs with h
  · exact (windowPosition_of_le s h.le).symm
  · rw [windowPosition_of_ge s (not_lt.1 h)]
    ring

end Transformer.RoPE.LeReRoPE

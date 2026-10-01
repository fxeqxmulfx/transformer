/-
# IC-EoT: the fixed time-of-use tariffs

arXiv:2603.22095v2, §4.4.1. These are the tariffs adopted by the
experiment, not a statement about current market tariffs.
-/

import Transformer.ICEoT.Section4_Performance

namespace Transformer.ICEoT

/-- The adopted apartment weekday tariff, with weekends/public holidays
off-peak all day; §4.4.1. Hour intervals are left-closed/right-open. -/
def apartmentTariff (hour : Fin 24) (weekend holiday : Bool) : ℚ :=
  if weekend || holiday || hour.val < 8 then 0.089
  else if (10 ≤ hour.val ∧ hour.val < 14) ∨ (18 ≤ hour.val ∧ hour.val < 22) then 0.210
  else 0.135

/-- The adopted office summer dual-zone tariff; §4.4.1. -/
def officeTariff (hour : Fin 24) : ℚ :=
  if (2 ≤ hour.val ∧ hour.val < 4) ∨ (11 ≤ hour.val ∧ hour.val < 15) then 0.129
  else 0.209

/-- The tariffs used by the experiment satisfy Corollary 2's price-sign
hypothesis for every hour, weekday, weekend and public holiday; §4.4.1. -/
theorem adopted_tariffs_positive (hour : Fin 24) (weekend holiday : Bool) :
    0 < apartmentTariff hour weekend holiday ∧ 0 < officeTariff hour := by
  constructor
  · unfold apartmentTariff
    split_ifs <;> norm_num
  · unfold officeTariff
    split_ifs <;> norm_num

/-- Representative exact boundary values, §4.4.1: apartment weekday
08/10/14/18/22 and office 02/04/11/15, including weekend/holiday handling. -/
theorem adopted_tariff_boundaries :
    apartmentTariff 8 false false = 0.135 ∧ apartmentTariff 10 false false = 0.210 ∧
    apartmentTariff 14 false false = 0.135 ∧ apartmentTariff 18 false false = 0.210 ∧
    apartmentTariff 22 false false = 0.135 ∧ apartmentTariff 10 true false = 0.089 ∧
    apartmentTariff 10 false true = 0.089 ∧ officeTariff 2 = 0.129 ∧ officeTariff 4 = 0.209 ∧
    officeTariff 11 = 0.129 ∧ officeTariff 15 = 0.209 := by
  norm_num [apartmentTariff, officeTariff]

end Transformer.ICEoT

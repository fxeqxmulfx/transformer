//! Canonical source snapshots versus independent moment storage. The first
//! moment has signed aliases; the second moment has unsigned aliases.
use crate::domain::model::{self, Arm};
pub const COUNT: usize = 143360;
pub fn pack(arm: Arm, expanded: &[f32]) -> Vec<f32> {
    assert_eq!(expanded.len(), arm.count());
    if arm == Arm::Swiglu {
        return expanded.to_vec();
    }
    model::pairs()
        .into_iter()
        .flat_map(|[ar, bi, _, _, _]| [expanded[ar as usize], expanded[bi as usize]])
        .collect()
}
pub fn expand(arm: Arm, owners: &[f32], second: bool) -> Vec<f32> {
    assert_eq!(owners.len(), COUNT);
    if arm == Arm::Swiglu {
        return owners.to_vec();
    }
    let mut values = vec![0.; arm.count()];
    for ([ar, bi, br, ai, kind], pair) in model::pairs().into_iter().zip(owners.as_chunks::<2>().0)
    {
        values[ar as usize] = pair[0];
        values[bi as usize] = pair[1];
        if kind == 0 {
            values[ai as usize] = pair[0];
            values[br as usize] = if second { pair[1] } else { -pair[1] };
        }
    }
    values
}
pub fn bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_and_unsigned_aliases_round_trip_including_zero_signs() {
        for arm in Arm::ALL {
            let mut owners: Vec<_> = (0..COUNT).map(|i| (i % 17) as f32 - 8.).collect();
            owners[COUNT - 1] = -0.;
            for second in [false, true] {
                let expanded = expand(arm, &owners, second);
                assert!(bits(&pack(arm, &expanded), &owners));
                assert!(bits(&expand(arm, &pack(arm, &expanded), second), &expanded));
            }
        }
    }
}

//! A fixed one-hot gate is an instance of CM46 weights_mem/ce_convex.
//! Unique source-state counts define profiles; query exclusion removes one
//! state even when the original diagnostic contains repeated records.
use crate::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub counts: Vec<(usize, usize)>,
}

pub(crate) fn profiles(source: &kernel::Model) -> Vec<Profile> {
    source
        .blocks
        .iter()
        .map(|block| {
            let mut counts = BTreeMap::new();
            for &label in &block.observed_labels {
                *counts.entry(label).or_insert(0) += 1;
            }
            Profile {
                counts: counts.into_iter().collect(),
            }
        })
        .collect()
}

pub(crate) fn bucket(profile: &Profile, excluded: Option<usize>) -> usize {
    let (mut total, mut top) = (0, 0);
    for &(label, count) in &profile.counts {
        let remaining = count - usize::from(excluded == Some(label));
        total += remaining;
        top = top.max(remaining);
    }
    let minority = total - top;
    if minority == 0 {
        0
    } else if 8 * minority <= total {
        1
    } else if 4 * minority <= total {
        2
    } else if 8 * minority <= 3 * total {
        3
    } else {
        4
    }
}

pub(crate) fn build(source: &kernel::Model, profiles: &[Profile]) -> Vec<usize> {
    (0..source.n * source.n)
        .map(|cell| {
            let (block, state) = source.location(cell);
            bucket(
                &profiles[block],
                source.blocks[block].observed_labels.get(state).copied(),
            )
        })
        .collect()
}

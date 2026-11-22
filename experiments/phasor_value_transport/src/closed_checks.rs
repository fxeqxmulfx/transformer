//! Independent source-prefix oracle for a closed relation address.
use crate::{engine::Result, relation_support::ClosedRead};
use std::collections::{BTreeMap, BTreeSet};

pub fn reference(prefix: &[u32], source: u32, vocab: u32) -> Option<u32> {
    let mut proposals = Vec::new();
    for relation in 0..vocab {
        let edges: BTreeMap<_, _> = prefix
            .windows(3)
            .filter(|triple| triple[1] == relation)
            .map(|triple| (triple[0], triple[2]))
            .collect();
        let Some(middle) = edges.get(&source) else {
            continue;
        };
        let domain: BTreeSet<_> = edges.keys().copied().collect();
        let image: BTreeSet<_> = edges.values().copied().collect();
        if image == domain && edges[middle] != source {
            proposals.push((domain.len(), edges[middle]));
        }
    }
    let largest = proposals.iter().map(|&(size, _)| size).max()?;
    let ends: BTreeSet<_> = proposals
        .into_iter()
        .filter_map(|(size, end)| (size == largest).then_some(end))
        .collect();
    (ends.len() == 1).then(|| *ends.first().unwrap())
}

pub fn run() -> Result<()> {
    let mut state = ClosedRead::new(11);
    for mut code in 0..3u32.pow(8) {
        state.clear();
        let mut prefix = Vec::new();
        for _ in 0..8 {
            let query = code % 3;
            code /= 3;
            if state.address(query) != reference(&prefix, query, 3) {
                return Err(
                    "closed address differs from the independently reconstructed relation".into(),
                );
            }
            prefix.push(query);
            if let [.., key, relation, value] = &prefix[..] {
                state.write(*key, *relation, *value);
            }
        }
    }
    state.clear();
    let cycles: &[(&[u32], u32, Option<u32>)] = &[
        (&[0, 1, 2], 8, Some(2)),
        (&[0, 2, 1], 9, None),
        (&[0, 1, 2], 9, Some(2)),
        (&[0, 2, 3, 1], 10, Some(3)),
    ];
    for &(cycle, relation, expected) in cycles {
        for (&a, &b) in cycle.iter().zip(cycle.iter().cycle().skip(1)) {
            state.write(a, relation, b);
        }
        if state.address(0) != expected {
            return Err(
                "closed relations disagree with the expected consensus or size ranking".into(),
            );
        }
    }
    state.write(5, 10, 2);
    if state.address(0) != Some(2) {
        return Err("a relation with a duplicate destination remained eligible".into());
    }
    Ok(())
}

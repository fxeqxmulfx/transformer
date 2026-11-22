//! Relate ordered output pairs witnessed in two learned contexts at shared keys.
//! All addresses are internal prototype slots. See aligned_read_sound.
use super::{ABSENT, Components};
use std::collections::HashMap;

struct PeerRows {
    first: usize,
    second: usize,
    shared_keys: usize,
    source: usize,
    target: usize,
    anchor: usize,
}

/// Each candidate needs two other supporting keys in its row pair and two
/// agreeing peer rows. Any conflicting vote suppresses the completion.
pub fn fill_from_peers(slots: usize, bank: &[u32], output: &mut [u32]) {
    let contexts = bank.len() / slots;
    assert_eq!(output.len(), bank.len());
    if contexts < 3 || slots < 3 || !output.contains(&ABSENT) {
        return;
    }
    let available: Vec<_> = bank
        .chunks_exact(slots)
        .enumerate()
        .filter(|(_, row)| row.iter().filter(|&&v| v != ABSENT).take(2).count() == 2)
        .collect();
    if available.len() < 3 {
        return;
    }
    let pair = |a: u32, b: u32| (a as usize) * slots + b as usize;
    let mut graph = Components::new(slots * slots);
    let mut pairs = Vec::new();
    for (index, &(i, left)) in available.iter().enumerate() {
        for &(j, right) in &available[index + 1..] {
            let mut observed = left
                .iter()
                .zip(right)
                .enumerate()
                .filter(|(_, (a, b))| **a != ABSENT && **b != ABSENT);
            let Some((first, (&a, &b))) = observed.next() else {
                continue;
            };
            let Some((second, (&c, &d))) = observed.next() else {
                continue;
            };
            let ab = pair(a, b);
            let ba = pair(b, a);
            graph.join(ab, pair(c, d));
            graph.join(ba, pair(d, c));
            let mut shared_keys = 2;
            for (_, (&u, &v)) in observed {
                graph.join(ab, pair(u, v));
                graph.join(ba, pair(v, u));
                shared_keys = 3; // Only the distinction between two and three is needed below.
            }
            for (source, target, anchor) in [(i, j, ab), (j, i, ba)] {
                pairs.push(PeerRows {
                    first,
                    second,
                    shared_keys,
                    source,
                    target,
                    anchor,
                });
            }
        }
    }
    graph.flatten();
    let mut decoders = vec![HashMap::new(); slots];
    for (source, decoder) in decoders.iter_mut().enumerate() {
        for image in 0..slots {
            let component = graph.parent[source * slots + image];
            decoder
                .entry(component)
                .and_modify(|v| *v = ABSENT)
                .or_insert(image as u32);
        }
    }
    let mut votes = vec![(ABSENT, 0_u8); bank.len()];
    for p in pairs {
        let component = graph.parent[p.anchor];
        let source = &bank[p.source * slots..(p.source + 1) * slots];
        for (key, &value) in source.iter().enumerate() {
            let at = p.target * slots + key;
            if value == ABSENT
                || output[at] != ABSENT
                || p.shared_keys == 2 && (key == p.first || key == p.second)
            {
                continue;
            }
            let Some(&image) = decoders[value as usize].get(&component) else {
                continue;
            };
            if image == ABSENT {
                continue;
            }
            let (winner, count) = &mut votes[at];
            if *count == 0 {
                *winner = image;
                *count = 1;
            } else if *winner != image {
                *winner = ABSENT;
            } else {
                *count = 2;
            }
        }
    }
    for (read, (candidate, count)) in output.iter_mut().zip(votes) {
        if count == 2 && candidate != ABSENT {
            *read = candidate;
        }
    }
}

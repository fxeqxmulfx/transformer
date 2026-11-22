//! Infer row adjacency from agreement on common hidden keys, then read a
//! consistent output for each finite shortest-path length in that graph.
use super::ABSENT;
use std::collections::VecDeque;

fn neighbours(slots: usize, evidence: &[u32]) -> Vec<Vec<usize>> {
    let rows: Vec<_> = evidence.chunks_exact(slots).collect();
    let mut edges = vec![Vec::new(); slots];
    let mut left_counts = vec![0_u32; slots];
    let mut right_counts = vec![0_u32; slots];
    for (a, &left) in rows.iter().enumerate() {
        for (b, &right) in rows[..a].iter().enumerate() {
            left_counts.fill(0);
            right_counts.fill(0);
            let mut common = 0_u64;
            let mut matching = 0_u64;
            for (&u, &v) in left.iter().zip(right) {
                if u != ABSENT && v != ABSENT {
                    common += 1;
                    matching += u64::from(u == v);
                    left_counts[u as usize] += 1;
                    right_counts[v as usize] += 1;
                }
            }
            if matching < 6 {
                continue;
            }
            let independent: u64 = left_counts
                .iter()
                .zip(&right_counts)
                .map(|(&u, &v)| u64::from(u) * u64::from(v))
                .sum();
            if matching * common > 2 * independent {
                edges[a].push(b);
                edges[b].push(a);
            }
        }
    }
    edges
}

fn distances(edges: &[Vec<usize>]) -> Vec<usize> {
    let slots = edges.len();
    let mut distance = vec![usize::MAX; slots * slots];
    let mut queue = VecDeque::new();
    for (start, row) in distance.chunks_exact_mut(slots).enumerate() {
        row[start] = 0;
        queue.push_back(start);
        while let Some(a) = queue.pop_front() {
            for &b in &edges[a] {
                if row[b] == usize::MAX {
                    row[b] = row[a] + 1;
                    queue.push_back(b);
                }
            }
        }
    }
    distance
}

#[derive(Default, Clone, Copy)]
struct Layer {
    value: Option<u32>,
    votes: usize,
    conflict: bool,
}

fn read_layers(edges: &[Vec<usize>], bank: &[u32]) -> Vec<u32> {
    let distance = distances(edges);
    let mut layers = vec![Layer::default(); edges.len()];
    for (&d, &y) in distance.iter().zip(bank) {
        if d != usize::MAX && y != ABSENT {
            let layer = &mut layers[d];
            layer.conflict |= layer.value.is_some_and(|v| v != y);
            layer.value = Some(y);
            layer.votes += 1;
        }
    }
    distance
        .iter()
        .zip(bank)
        .map(|(&d, &old)| {
            let Some(layer) = layers.get(d) else {
                return ABSENT;
            };
            if !layer.conflict && layer.votes >= 3 + usize::from(old != ABSENT) {
                layer.value.unwrap_or(ABSENT)
            } else {
                ABSENT
            }
        })
        .collect()
}

/// Edges come from the current hidden evidence. Every output witness for
/// a length class must come from the original learned bank.
pub(super) fn proposals(slots: usize, evidence: &[u32], bank: &[u32]) -> Option<Vec<u32>> {
    let edges = neighbours(slots, evidence);
    edges
        .iter()
        .any(|row| !row.is_empty())
        .then(|| read_layers(&edges, bank))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_star_outputs_require_other_cells_and_agreement() {
        let edges = vec![vec![1, 2, 3], vec![0], vec![0], vec![0], vec![]];
        let mut bank = vec![ABSENT; 25];
        for (a, b) in [(1, 2), (2, 1), (3, 1)] {
            bank[5 * a + b] = 4;
        }
        let reads = read_layers(&edges, &bank);
        assert_eq!(reads[5 * 2 + 3], 4);
        assert_eq!(reads[7], ABSENT);
        assert_eq!(reads[5 * 3 + 4], ABSENT);
        bank[5 + 3] = 2;
        assert_eq!(read_layers(&edges, &bank)[5 * 2 + 3], ABSENT);
    }

    #[test]
    fn learned_adjacency_and_outputs_move_with_the_hidden_dictionary() {
        let n = 41;
        let rename = |v: usize| (7 * v + 9) % n;
        let mut bank = vec![ABSENT; n * n];
        let mut renamed = bank.clone();
        for a in 0..n {
            for b in 0..n {
                let y = (5 * (a.abs_diff(b) / 2) + 3) % n;
                if (a, b) != (11, 24) && (a, b) != (24, 11) {
                    bank[a * n + b] = y as u32;
                    renamed[rename(a) * n + rename(b)] = rename(y) as u32;
                }
            }
        }
        let a = proposals(n, &bank, &bank).unwrap();
        let b = proposals(n, &renamed, &renamed).unwrap();
        assert_eq!(a[n * 11 + 24], 33);
        for (i, y) in a.into_iter().enumerate() {
            let expected = if y == ABSENT {
                ABSENT
            } else {
                rename(y as usize) as u32
            };
            assert_eq!(b[rename(i / n) * n + rename(i % n)], expected);
        }
        use super::super::{RoleEvidence, extend};
        let empty = vec![ABSENT; n * n];
        for model in [RoleEvidence::OBSERVED_ROLES, RoleEvidence::WITH_GRAPH] {
            let mut evidence = bank.clone();
            let mut read = empty.clone();
            let roles = RoleEvidence {
                role_proposals: false,
                ..model
            };
            assert!(extend(n, &bank, &mut evidence, &empty, &mut read, roles));
            assert_eq!(
                read[n * 11 + 24],
                if model.graph_layers { 33 } else { ABSENT }
            );
        }
    }
}

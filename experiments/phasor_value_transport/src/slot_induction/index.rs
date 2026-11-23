//! Exact grouping of repeated floating-point vectors. IndexedInduction
//! proves that the latest stamp over all angular neighbors equals the
//! backward scan, including when another sequence shares the geometry.
use super::{accepts, dot, latest};
use std::collections::HashMap;

#[derive(Default)]
struct Class {
    representative: usize,
    neighbors: Vec<usize>,
    last_successor: Option<usize>,
    latest_match: Option<usize>,
}

pub(super) fn donors(
    input: &[f32],
    energy: &[f32],
    width: usize,
    length: usize,
) -> Vec<Option<usize>> {
    let share_geometry = length >= 256 && energy.len() > length;
    let edge_limit = if share_geometry {
        8 * length
    } else {
        4 * length
    };
    let mut ids = HashMap::<Vec<u32>, usize>::new();
    let mut classes = Vec::<Class>::new();
    let mut bits = Vec::with_capacity(width);
    let mut prefix = Vec::<usize>::with_capacity(length);
    let mut edges = 0;
    let mut result = Vec::with_capacity(energy.len());

    for start in (0..energy.len()).step_by(length) {
        if share_geometry && classes.len() <= 4 * length && edges <= edge_limit {
            for class in &mut classes {
                class.last_successor = None;
                class.latest_match = None;
            }
        } else {
            ids.clear();
            classes.clear();
            edges = 0;
        }
        prefix.clear();

        for position in 0..length {
            let row = start + position;
            // Dense angular neighborhoods can require quadratic space.
            // Finish this sequence with the bounded-memory backwards scan.
            if classes.len() > 4 * length || edges > edge_limit {
                result
                    .extend((row..start + length).map(|r| latest(input, energy, width, start, r)));
                break;
            }
            if position >= 2 {
                let completed = prefix[position - 2];
                classes[completed].last_successor = Some(row - 1);
                for i in 0..classes[completed].neighbors.len() {
                    let neighbor = classes[completed].neighbors[i];
                    classes[neighbor].latest_match = Some(row - 1);
                }
            }
            let query = &input[row * width..][..width];
            bits.clear();
            bits.extend(query.iter().map(|v| v.to_bits()));
            let class = if let Some(&id) = ids.get(bits.as_slice()) {
                id
            } else {
                let id = classes.len();
                let mut fresh = Class {
                    representative: row,
                    ..Class::default()
                };
                for (i, other) in classes.iter_mut().enumerate() {
                    let j = other.representative;
                    let score = dot(query, &input[j * width..][..width]);
                    if accepts(score, energy[row], energy[j]) {
                        fresh.neighbors.push(i);
                        other.neighbors.push(id);
                        fresh.latest_match = fresh.latest_match.max(other.last_successor);
                    }
                }
                if accepts(energy[row], energy[row], energy[row]) {
                    fresh.neighbors.push(id);
                }
                edges += 2 * fresh.neighbors.len();
                classes.push(fresh);
                ids.insert(bits.clone(), id);
                id
            };
            prefix.push(class);
            result.push(classes[class].latest_match);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot_induction::Prefix;

    #[test]
    fn all_donors_match_a_backward_scan_on_three_independent_sequences() {
        let mut rng = crate::random::Random::new(413);
        let width = 16;
        for length in [63, 96, 127, 260] {
            for crowded in [false, true] {
                let table: Vec<Vec<f32>> = (0..85)
                    .map(|i| {
                        (0..width)
                            .map(|j| {
                                if i == 0 {
                                    return if j % 2 == 0 { 0. } else { -0. };
                                }
                                if crowded {
                                    rng.gauss(1., 0.01) as f32
                                } else {
                                    rng.gauss(0., 0.6) as f32
                                }
                            })
                            .collect()
                    })
                    .collect();
                let mut input = Vec::new();
                for row in 0..3 * length {
                    let id = if row % 7 == 0 {
                        0
                    } else {
                        rng.below(table.len() as u32) as usize
                    };
                    let mut v = table[id].clone();
                    if row % 17 == 0 {
                        v[0] += 0.0002;
                    }
                    input.extend(v);
                }
                let rows: Vec<_> = input.chunks_exact(width).collect();
                let prefix = Prefix::new(input.clone(), width, length);
                for (r, query) in rows.iter().enumerate() {
                    let end = r.saturating_sub(1);
                    let expected = (r - r % length..end)
                        .rev()
                        .find(|&j| {
                            let s = dot(query, rows[j]);
                            s > 0. && 2. * s * s > dot(query, query) * dot(rows[j], rows[j])
                        })
                        .map(|j| (j + 1, rows[j + 1]));
                    assert_eq!(prefix.read(r), expected);
                }
            }
        }
    }
}

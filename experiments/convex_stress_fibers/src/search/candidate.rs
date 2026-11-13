use super::*;
use groups::Axis;

pub(super) fn point(space: &finite_system::Space, coefficients: &[u16], p: usize) -> Vec<u16> {
    space
        .particular
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            ((v as usize
                + coefficients
                    .iter()
                    .zip(&space.directions)
                    .map(|(&c, d)| c as usize * d[i] as usize)
                    .sum::<usize>())
                % p) as u16
        })
        .collect()
}
pub(super) fn classify(
    input: &Input,
    partition: &groups::Partition,
    size: usize,
    x: &[u16],
    work: &mut Work,
) -> std::result::Result<Vec<usize>, Decision> {
    let n = input.n;
    for (axis, phases) in [(Axis::Row, &x[..n]), (Axis::Column, &x[n..2 * n])] {
        let mut seen = vec![None; n];
        for (second, &phase) in phases.iter().enumerate() {
            work.permutation_entries += 1;
            if let Some(first) = seen[phase as usize] {
                return Err(Decision::Collision {
                    axis,
                    first,
                    second,
                    phase,
                });
            }
            seen[phase as usize] = Some(second);
        }
    }
    let mut seen = vec![None; n];
    for (second, &[cell, label]) in input.observations.iter().enumerate() {
        work.decoder_observations += 1;
        let z = (x[cell % n] as usize + x[n + cell / n] as usize) % n;
        if let Some((previous, first)) = seen[z] {
            if previous != label {
                return Err(Decision::Conflict {
                    first,
                    second,
                    latent: z,
                });
            }
        } else {
            seen[z] = Some((label, second));
        }
    }
    if let Some(latent) = seen.iter().position(Option::is_none) {
        return Err(Decision::Uncovered { latent });
    }
    let decoder: Vec<_> = seen.into_iter().map(|s| s.unwrap().0).collect();
    let mut cardinalities = vec![0; input.vocab];
    let mut masses = vec![0; input.vocab];
    for (z, &label) in decoder.iter().enumerate() {
        work.fiber_entries += 1;
        cardinalities[label] += 1;
        masses[label] = (masses[label] + z) % n;
    }
    // Every size-k group is checked, including those beyond the equation cap.
    for (group, g) in partition
        .groups
        .iter()
        .enumerate()
        .filter(|(_, g)| g.members.len() == size)
    {
        work.fiber_entries += 1;
        let (cardinality, mass) = (cardinalities[g.label], masses[g.label]);
        if cardinality != size || mass != x[2 * n] as usize {
            return Err(Decision::FiberMismatch {
                group,
                cardinality,
                mass,
            });
        }
    }
    Ok(decoder)
}

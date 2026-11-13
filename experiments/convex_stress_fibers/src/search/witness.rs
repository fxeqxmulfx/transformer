//! Independent integer checks use original observations and actual full fibers.
//! They do not call the producer's collision/decoder/fiber classifier.
use super::*;
use groups::Axis;

fn latent(n: usize, x: &[u16], cell: usize) -> usize {
    (x[cell % n] as usize + x[n + cell / n] as usize) % n
}
fn decoder(input: &Input, x: &[u16]) -> Result<Vec<usize>> {
    let mut result = vec![];
    for z in 0..input.n {
        let mut labels: Vec<_> = input
            .observations
            .iter()
            .filter(|s| latent(input.n, x, s[0]) == z)
            .map(|s| s[1])
            .collect();
        labels.sort_unstable();
        labels.dedup();
        if labels.len() != 1 {
            return Err("independent decoder has a conflict or an uncovered latent state".into());
        }
        result.push(labels[0]);
    }
    Ok(result)
}
fn endpoint(
    input: &Input,
    partition: &groups::Partition,
    size: usize,
    x: &[u16],
) -> Result<Vec<usize>> {
    let n = input.n;
    for offset in [0, n] {
        for i in 0..n {
            for j in 0..i {
                if x[offset + i] == x[offset + j] {
                    return Err("claimed candidate phase map is not injective".into());
                }
            }
        }
    }
    let d = decoder(input, x)?;
    for g in partition.groups.iter().filter(|g| g.members.len() == size) {
        let mut image: Vec<_> = g
            .members
            .iter()
            .map(|&i| match g.axis {
                Axis::Row => (x[g.fixed] as usize + x[n + i] as usize) % n,
                Axis::Column => (x[i] as usize + x[n + g.fixed] as usize) % n,
            })
            .collect();
        image.sort_unstable();
        let actual: Vec<_> = (0..n).filter(|&z| d[z] == g.label).collect();
        if image != actual
            || actual.len() != size
            || actual.iter().sum::<usize>() % n != x[2 * n] as usize
        {
            return Err("actual complete-fiber image/cardinality/mass premise fails".into());
        }
        let (offset, fixed) = match g.axis {
            Axis::Row => (n, g.fixed),
            Axis::Column => (0, n + g.fixed),
        };
        let sum = g
            .members
            .iter()
            .map(|&i| x[offset + i] as usize)
            .sum::<usize>()
            + size * x[fixed] as usize;
        if sum % n != x[2 * n] as usize {
            return Err("actual CM26 phase-sum equation fails".into());
        }
    }
    Ok(d)
}
pub(super) fn check(
    input: &Input,
    partition: &groups::Partition,
    size: usize,
    x: &[u16],
    trial: &Trial,
    candidates: &[Candidate],
    accepted_so_far: usize,
) -> Result<()> {
    let n = input.n;
    match trial.decision {
        Decision::Collision {
            axis,
            first,
            second,
            phase,
        } => {
            let offset = if axis == Axis::Row { 0 } else { n };
            if first >= n
                || second >= n
                || first >= second
                || x[offset + first] != phase
                || x[offset + second] != phase
            {
                return Err("invalid phase-collision witness".into());
            }
        }
        Decision::Conflict {
            first,
            second,
            latent: z,
        } => {
            if first >= input.observations.len()
                || second >= input.observations.len()
                || first >= second
            {
                return Err("invalid decoder conflict indices".into());
            }
            let (a, b) = (input.observations[first], input.observations[second]);
            if a[1] == b[1] || latent(n, x, a[0]) != z || latent(n, x, b[0]) != z {
                return Err("false original-observation decoder conflict".into());
            }
        }
        Decision::Uncovered { latent: z } => {
            if z >= n || input.observations.iter().any(|s| latent(n, x, s[0]) == z) {
                return Err("declared uncovered latent state is observed".into());
            }
        }
        Decision::FiberMismatch {
            group,
            cardinality,
            mass,
        } => {
            let g = partition
                .groups
                .get(group)
                .ok_or("invalid rejected fiber index")?;
            let d = decoder(input, x)?;
            let actual: Vec<_> = (0..n).filter(|&z| d[z] == g.label).collect();
            if g.members.len() != size
                || actual.len() != cardinality
                || actual.iter().sum::<usize>() % n != mass
                || (cardinality == size && mass == x[2 * n] as usize)
            {
                return Err("invalid actual-fiber rejection witness".into());
            }
        }
        Decision::Duplicate { candidate } | Decision::Accepted { candidate } => {
            let d = endpoint(input, partition, size, x)?;
            let stored = candidates
                .get(candidate)
                .ok_or("missing admitted candidate")?;
            let model = &stored.model;
            model.validate()?;
            let output: Vec<_> = (0..n * n).map(|cell| d[latent(n, x, cell)]).collect();
            if model.n != n
                || model.q != n
                || model.vocab != input.vocab
                || model.output_labels != output
            {
                return Err("stored candidate differs from the independently decoded table".into());
            }
            if matches!(trial.decision, Decision::Duplicate { .. }) {
                if candidate >= accepted_so_far {
                    return Err("duplicate refers to a later candidate".into());
                }
            } else {
                if candidate != accepted_so_far
                    || stored.common_mass != x[2 * n] as usize
                    || model
                        .rows
                        .iter()
                        .copied()
                        .ne(x[..n].iter().map(|&z| z as usize))
                    || model
                        .columns
                        .iter()
                        .copied()
                        .ne(x[n..2 * n].iter().map(|&z| z as usize))
                    || model.decoder != d
                {
                    return Err("admitted latent coordinates or mass differ".into());
                }
                if stored.audit != model.audit(&input.samples())?
                    || stored.audit["training_errors"] != 0
                {
                    return Err("actual endpoint raw spectrum/CE audit differs".into());
                }
            }
        }
    }
    Ok(())
}

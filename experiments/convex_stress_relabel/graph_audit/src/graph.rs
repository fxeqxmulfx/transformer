//! Training-only integer checks for the graph and phase conditions in CM13.
//! Class phases are categorical coordinates; input token values enter as indices.
use convex_stress_budgets::Result;
use serde::Serialize;
use std::collections::VecDeque;

#[derive(Clone, Copy)]
pub struct Pair {
    pub row: usize,
    pub column: usize,
    pub label: usize,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Structure {
    pub vocabulary: usize,
    pub observed_cells: usize,
    pub connected_components: usize,
    pub connected: bool,
    pub perfect_matching: Option<Vec<usize>>,
    pub observed_phase_consistent: bool,
    pub row_phases: Option<Vec<usize>>,
    pub column_phases: Option<Vec<usize>>,
    pub completion_labels: Option<Vec<usize>>,
}

fn augment(
    row: usize,
    edges: &[Vec<usize>],
    seen: &mut [bool],
    owner: &mut [Option<usize>],
) -> bool {
    for &column in &edges[row] {
        if seen[column] {
            continue;
        }
        seen[column] = true;
        if owner[column].is_none_or(|old| augment(old, edges, seen, owner)) {
            owner[column] = Some(row);
            return true;
        }
    }
    false
}

pub fn analyze(n: usize, training: &[Pair], coding: &[usize]) -> Result<Structure> {
    if !(2..=193).contains(&n) || training.is_empty() {
        return Err("invalid graph dimensions".into());
    }
    let mut sorted = coding.to_vec();
    sorted.sort_unstable();
    if sorted != (0..n).collect::<Vec<_>>() {
        return Err("graph audit requires a class permutation".into());
    }
    let mut labels = vec![None; n * n];
    let mut neighbors = vec![vec![]; 2 * n];
    let mut rows = vec![vec![]; n];
    for &Pair { row, column, label } in training {
        if row >= n || column >= n || label >= n || labels[row + n * column].is_some() {
            return Err("invalid or duplicated observed cell".into());
        }
        labels[row + n * column] = Some(label);
        neighbors[row].push((n + column, coding[label]));
        neighbors[n + column].push((row, coding[label]));
        rows[row].push(column);
    }
    for edges in &mut neighbors {
        edges.sort_unstable();
    }
    for edges in &mut rows {
        edges.sort_unstable();
    }
    let mut potentials = vec![None; 2 * n];
    let mut components = 0;
    let mut consistent = true;
    for start in 0..2 * n {
        if potentials[start].is_some() {
            continue;
        }
        components += 1;
        potentials[start] = Some(0);
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            for &(other, phase) in &neighbors[node] {
                let proposed = (phase + n - potentials[node].unwrap()) % n;
                if let Some(prior) = potentials[other] {
                    consistent &= prior == proposed;
                } else {
                    potentials[other] = Some(proposed);
                    queue.push_back(other);
                }
            }
        }
    }
    let mut owner = vec![None; n];
    let mut cardinality = 0;
    for row in 0..n {
        cardinality += usize::from(augment(row, &rows, &mut vec![false; n], &mut owner));
    }
    let matching = if cardinality == n {
        let mut matched = vec![0; n];
        for (column, row) in owner.into_iter().enumerate() {
            matched[row.unwrap()] = column;
        }
        let mut sorted = matched.clone();
        sorted.sort_unstable();
        if sorted != (0..n).collect::<Vec<_>>()
            || matched
                .iter()
                .enumerate()
                .any(|(row, &column)| labels[row + n * column].is_none())
        {
            return Err("constructed matching failed independent edge/permutation checks".into());
        }
        Some(matched)
    } else {
        None
    };
    let row_phases = consistent.then(|| {
        potentials[..n]
            .iter()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
    });
    let column_phases = consistent.then(|| {
        potentials[n..]
            .iter()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
    });
    let completion = if components == 1 && consistent {
        let mut inverse = vec![0; n];
        for (label, &phase) in coding.iter().enumerate() {
            inverse[phase] = label;
        }
        let values: Vec<_> = (0..n * n)
            .map(|cell| {
                inverse[(row_phases.as_ref().unwrap()[cell % n]
                    + column_phases.as_ref().unwrap()[cell / n])
                    % n]
            })
            .collect();
        if training
            .iter()
            .any(|p| values[p.row + n * p.column] != p.label)
        {
            return Err("constructed completion does not fit observations".into());
        }
        Some(values)
    } else {
        None
    };
    Ok(Structure {
        vocabulary: n,
        observed_cells: training.len(),
        connected_components: components,
        connected: components == 1,
        perfect_matching: matching,
        observed_phase_consistent: consistent,
        row_phases,
        column_phases,
        completion_labels: completion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cells(xs: &[(usize, usize, usize)]) -> Vec<Pair> {
        xs.iter()
            .map(|&(row, column, label)| Pair { row, column, label })
            .collect()
    }

    #[test]
    fn nonconstant_missing_cell_has_coverage_and_unique_phase_completion() -> Result<()> {
        let observed = cells(&[(0, 0, 0), (0, 1, 1), (1, 1, 0)]);
        let check = analyze(2, &observed, &[0, 1])?;
        assert!(
            check.connected && check.observed_phase_consistent && check.perfect_matching.is_some()
        );
        assert_eq!(check.completion_labels, Some(vec![0, 1, 1, 0]));
        let renamed = cells(&[(0, 0, 1), (0, 1, 0), (1, 1, 1)]);
        let other = analyze(2, &renamed, &[1, 0])?;
        assert_eq!(other.completion_labels, Some(vec![1, 0, 0, 1]));
        assert_eq!(other.row_phases, check.row_phases);
        Ok(())
    }
    #[test]
    fn connectivity_matching_and_phase_consistency_are_separate_conditions() -> Result<()> {
        let disconnected = analyze(2, &cells(&[(0, 0, 0), (1, 1, 0)]), &[0, 1])?;
        assert!(!disconnected.connected && disconnected.perfect_matching.is_some());
        assert!(disconnected.completion_labels.is_none());
        // The connected cross covers all vertices but has no perfect matching.
        let cross = analyze(
            3,
            &cells(&[(0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 0, 1), (2, 0, 2)]),
            &[0, 1, 2],
        )?;
        assert!(cross.connected && cross.observed_phase_consistent);
        assert!(cross.perfect_matching.is_none());
        let conflict = analyze(
            2,
            &cells(&[(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]),
            &[0, 1],
        )?;
        assert!(
            conflict.connected
                && conflict.perfect_matching.is_some()
                && !conflict.observed_phase_consistent
        );
        assert!(conflict.row_phases.is_none() && conflict.completion_labels.is_none());
        Ok(())
    }
    #[test]
    fn augmenting_paths_recover_a_matching_and_bad_input_is_rejected() -> Result<()> {
        let check = analyze(
            3,
            &cells(&[(0, 0, 0), (0, 1, 1), (1, 0, 1), (2, 1, 0), (2, 2, 1)]),
            &[0, 1, 2],
        )?;
        assert_eq!(check.perfect_matching, Some(vec![1, 0, 2]));
        assert!(analyze(2, &cells(&[(0, 0, 0), (0, 0, 1)]), &[0, 1]).is_err());
        assert!(analyze(2, &cells(&[(0, 0, 0)]), &[0, 0]).is_err());
        assert!(analyze(usize::MAX, &[], &[]).is_err());
        Ok(())
    }
}

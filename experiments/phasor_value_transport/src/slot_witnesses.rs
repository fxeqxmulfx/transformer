//! Pool distinct ordered output pairs observed across common hidden keys.
use super::{ABSENT, Components};
use std::collections::HashMap;
#[path = "slot_symmetry.rs"]
mod symmetry;
pub use symmetry::RoleEvidence;

#[derive(Clone, Copy, Default)]
struct Values([u64; 4]);

impl Values {
    fn insert(&mut self, y: usize) {
        self.0[y / 64] |= 1 << (y % 64);
    }
    fn intersect(&mut self, other: &Self) {
        for (a, b) in self.0.iter_mut().zip(other.0) {
            *a &= b;
        }
    }
    fn unique(&self) -> Option<u32> {
        if self.0.iter().map(|word| word.count_ones()).sum::<u32>() != 1 {
            return None;
        }
        let (i, word) = self.0.iter().enumerate().find(|(_, word)| **word != 0)?;
        Some((64 * i) as u32 + word.trailing_zeros())
    }
}

/// Pool signatures supported by six distinct learned output pairs and read
/// their common admissible image from at most sixteen learned reference rows.
pub fn fill_from_witnesses(
    slots: usize,
    bank: &[u32],
    priority: &[usize],
    constant: &[u32],
    output: &mut [u32],
) {
    read_round(slots, bank, priority, constant, output, 16, false);
}

/// Further rounds can use answers established at other keys. All source
/// values survive. The next refresh always starts from the learned bank.
pub fn fill_in_rounds(
    slots: usize,
    bank: &[u32],
    priority: &[usize],
    constant: &[u32],
    output: &mut [u32],
    roles: Option<RoleEvidence>,
) {
    if !output
        .iter()
        .zip(constant)
        .any(|(&y, &c)| y == ABSENT && c == ABSENT)
    {
        return;
    }
    let mut evidence = bank.to_vec();
    let relative = roles
        .is_some_and(|terms| symmetry::extend(slots, bank, &mut evidence, constant, output, terms));
    for budget in [16, 32, 32] {
        read_round(
            slots, &evidence, priority, constant, output, budget, relative,
        );
        if !output
            .iter()
            .zip(constant)
            .any(|(&read, &c)| read == ABSENT && c == ABSENT)
        {
            break;
        }
        let mut changed = false;
        for ((value, &c), &transport) in evidence.iter_mut().zip(constant).zip(output.iter()) {
            if *value == ABSENT && (c != ABSENT || transport != ABSENT) {
                *value = if c != ABSENT { c } else { transport };
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn read_round(
    slots: usize,
    bank: &[u32],
    priority: &[usize],
    constant: &[u32],
    output: &mut [u32],
    peer_limit: usize,
    relative: bool,
) {
    const THRESHOLD: u8 = 6;
    assert!(slots > 0 && slots <= 256);
    assert!(bank.len().is_multiple_of(slots));
    assert_eq!(constant.len(), bank.len());
    assert_eq!(output.len(), bank.len());
    let contexts = bank.len() / slots;
    assert_eq!(priority.len(), contexts);
    if !output
        .iter()
        .zip(constant)
        .any(|(a, b)| *a == ABSENT && *b == ABSENT)
    {
        return;
    }
    let rows: Vec<_> = bank.chunks_exact(slots).collect();
    let mut peers: Vec<_> = priority
        .iter()
        .copied()
        .map(|row| (row, rows[row].iter().filter(|&&v| v != ABSENT).count()))
        .filter(|&(_, count)| count >= usize::from(THRESHOLD))
        .collect();
    // Stable sorting keeps the learned prototype order for equal support counts.
    peers.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    peers.truncate(peer_limit);
    if peers.len() < 2 {
        return;
    }
    let pairs = peers.len() * contexts;
    let mut signatures = vec![Vec::new(); pairs];
    let mut postings = vec![Vec::new(); slots * slots];
    let mut counts = vec![0_u16; pairs];
    let mut lengths = vec![0_usize; pairs];
    let mut groups = Components::new(pairs);
    for (index, &(source, _)) in peers.iter().enumerate() {
        for (target, &row) in rows.iter().enumerate() {
            if source == target {
                continue;
            }
            let id = index * contexts + target;
            let observed = &mut signatures[id];
            for (&u, &v) in rows[source].iter().zip(row) {
                if u != ABSENT && v != ABSENT {
                    observed.push(u as usize * slots + v as usize);
                }
            }
            observed.sort_unstable();
            observed.dedup();
            if observed.len() < usize::from(THRESHOLD) {
                continue;
            }
            lengths[id] = observed.len();
            let mut touched = Vec::new();
            for &uv in observed.iter() {
                for &previous in &postings[uv] {
                    let count: &mut u16 = &mut counts[previous];
                    if *count == 0 {
                        touched.push(previous);
                    }
                    *count = count.saturating_add(1);
                    let required = if relative {
                        lengths[id]
                            .min(lengths[previous])
                            .div_ceil(2)
                            .max(usize::from(THRESHOLD))
                    } else {
                        usize::from(THRESHOLD)
                    };
                    if usize::from(*count) == required {
                        groups.join(id, previous);
                    }
                }
                postings[uv].push(id);
            }
            for previous in touched {
                counts[previous] = 0;
            }
        }
    }
    groups.flatten();
    let roots = &groups.parent;
    let mut members = vec![0_usize; pairs];
    for &root in roots {
        members[root as usize] += 1;
    }
    let mut pooled: HashMap<usize, Vec<Values>> = HashMap::new();
    for (id, signature) in signatures.iter().enumerate() {
        let root = roots[id] as usize;
        if members[root] < 2 {
            continue;
        }
        let table = pooled
            .entry(root)
            .or_insert_with(|| vec![Values::default(); slots]);
        for &uv in signature {
            table[uv / slots].insert(uv % slots);
        }
    }
    for (target, (reads, constants)) in output
        .chunks_exact_mut(slots)
        .zip(constant.chunks_exact(slots))
        .enumerate()
    {
        for (key, (read, &constant)) in reads.iter_mut().zip(constants).enumerate() {
            if *read != ABSENT || constant != ABSENT {
                continue;
            }
            let mut candidates = Values([u64::MAX; 4]);
            let mut supporting_rows = 0;
            for (index, &(source, _)) in peers.iter().enumerate() {
                let source_value = rows[source][key];
                if source == target || source_value == ABSENT {
                    continue;
                }
                if let Some(table) = pooled.get(&(roots[index * contexts + target] as usize)) {
                    let options = table[source_value as usize];
                    if options.0 != [0; 4] {
                        candidates.intersect(&options);
                        supporting_rows += 1;
                        if candidates.0 == [0; 4] {
                            break;
                        }
                    }
                }
            }
            if supporting_rows >= 2
                && let Some(y) = candidates.unique()
            {
                *read = y;
            }
        }
    }
}

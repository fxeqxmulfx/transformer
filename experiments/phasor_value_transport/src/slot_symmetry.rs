//! Transform learned hidden correspondences by permutations of their three roles.
use super::ABSENT;
#[path = "slot_graph_classes.rs"]
mod graph_classes;

const PERMUTATIONS: [[usize; 3]; 5] = [[1, 0, 2], [0, 2, 1], [2, 1, 0], [1, 2, 0], [2, 0, 1]];

/// Fixed-weight ablations change one read at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoleEvidence {
    pub role_proposals: bool,
    pub relative_overlap: bool,
    pub graph_layers: bool,
}

impl RoleEvidence {
    pub const OBSERVED_ROLES: Self = Self {
        role_proposals: true,
        relative_overlap: true,
        graph_layers: false,
    };
    pub const WITH_GRAPH: Self = Self {
        graph_layers: true,
        ..Self::OBSERVED_ROLES
    };
}

/// Return whether a supported role permutation enables the relative overlap
/// test. Both the bank and the read retain their previous answers.
pub fn extend(
    slots: usize,
    observed: &[u32],
    bank: &mut [u32],
    constant: &[u32],
    output: &mut [u32],
    terms: RoleEvidence,
) -> bool {
    let (other, accepted) = proposals(slots, observed);
    if accepted == 0 {
        return false;
    }
    if terms.role_proposals {
        fill(bank, constant, output, other);
    }
    if terms.graph_layers
        && accepted & 1 != 0
        && let Some(layers) = graph_classes::proposals(slots, bank, observed)
    {
        fill(bank, constant, output, layers);
    }
    terms.relative_overlap
}

fn fill(bank: &mut [u32], constant: &[u32], output: &mut [u32], other: Vec<u32>) {
    for (((value, read), &c), proposal) in bank.iter_mut().zip(output).zip(constant).zip(other) {
        if proposal == ABSENT {
            continue;
        }
        if *value == ABSENT {
            *value = proposal;
        }
        if *read == ABSENT && c == ABSENT {
            *read = proposal;
        }
    }
}

/// Accepted coordinate permutations must agree with every observed cell
/// they predict and with each other at each proposed destination.
pub fn proposals(n: usize, bank: &[u32]) -> (Vec<u32>, u8) {
    assert_eq!(bank.len(), n * n);
    let mut result = vec![ABSENT; bank.len()];
    let mut conflict = vec![false; bank.len()];
    let mut accepted = 0;
    for (i, axes) in PERMUTATIONS.iter().enumerate() {
        let mut candidate = vec![ABSENT; bank.len()];
        let mut support = 0;
        let mut valid = true;
        for (cell, &value) in bank.iter().enumerate() {
            if value == ABSENT {
                continue;
            }
            let triple = [cell / n, cell % n, value as usize];
            let [a, b, y] = axes.map(|d| triple[d]);
            let dst = a * n + b;
            let y = y as u32;
            if (bank[dst] != ABSENT && bank[dst] != y)
                || (candidate[dst] != ABSENT && candidate[dst] != y)
            {
                valid = false;
                break;
            }
            if dst != cell {
                support += usize::from(bank[dst] == y);
                candidate[dst] = y;
            }
        }
        if !valid || support < 12 {
            continue;
        }
        accepted |= 1 << i;
        for ((current, clash), next) in result.iter_mut().zip(&mut conflict).zip(candidate) {
            if next == ABSENT {
                continue;
            }
            if *current != ABSENT && *current != next {
                *clash = true;
            }
            *current = next;
        }
    }
    for (y, clash) in result.iter_mut().zip(conflict) {
        if clash {
            *y = ABSENT;
        }
    }
    (result, accepted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exchanging_roles_requires_nontrivial_agreement_and_rejects_conflicts() {
        let n = 13;
        let mut bank: Vec<_> = (0..n * n).map(|i| ((i / n + i % n) % n) as u32).collect();
        let cell = n + 2;
        bank[cell] = ABSENT;
        let (filled, mask) = proposals(n, &bank);
        assert_eq!(mask, 1);
        assert_eq!(filled[cell], 3);
        assert_eq!(filled[0], ABSENT);
        bank[3 * n + 4] = 0;
        assert_eq!(proposals(n, &bank).1, 0);
        let diagonal: Vec<_> = (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    (i / n) as u32
                } else {
                    ABSENT
                }
            })
            .collect();
        assert_eq!(proposals(n, &diagonal).1, 0);
    }

    #[test]
    fn all_roles_can_agree_under_a_common_permutation_of_the_dictionary() {
        let n = 16;
        let rename = |x: usize| (5 * x + 3) % n;
        let mut bank = vec![ABSENT; n * n];
        for a in 0..n {
            for b in 0..n {
                bank[n * rename(a) + rename(b)] = rename(a ^ b) as u32;
            }
        }
        let cell = n * rename(5) + rename(9);
        bank[cell] = ABSENT;
        let (proposed, mask) = proposals(n, &bank);
        assert_eq!(mask, 31);
        assert_eq!(proposed[cell], rename(12) as u32);
        for terms in [
            RoleEvidence::OBSERVED_ROLES,
            RoleEvidence {
                role_proposals: false,
                ..RoleEvidence::OBSERVED_ROLES
            },
            RoleEvidence {
                relative_overlap: false,
                ..RoleEvidence::OBSERVED_ROLES
            },
        ] {
            let mut copy = bank.clone();
            let absent = vec![ABSENT; bank.len()];
            let mut read = absent.clone();
            read[n * rename(2) + rename(1)] = rename(4) as u32;
            let relative = extend(n, &bank, &mut copy, &absent, &mut read, terms);
            assert_eq!(relative, terms.relative_overlap);
            let expected = if !terms.role_proposals {
                ABSENT
            } else {
                rename(12) as u32
            };
            assert_eq!(copy[cell], expected);
            assert_eq!(read[cell], expected);
            assert_eq!(read[n * rename(2) + rename(1)], rename(4) as u32);
        }
    }
}

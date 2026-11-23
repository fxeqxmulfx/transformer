//! Use the existing triple closure on pairs of distinct keys that share an observed image.
use super::{ABSENT, TripleGraph};

#[path = "slot_fiber_chart.rs"]
mod chart;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiberRead {
    PartnerValues,
    PartnerThenChart,
    Chart,
}

fn partners(n: usize, bank: &[u32], transpose: bool) -> Vec<u32> {
    let mut result = vec![ABSENT; bank.len()];
    let mut keys = vec![[0; 2]; n];
    let mut sizes = vec![0usize; n];
    for row in 0..n {
        sizes.fill(0);
        for k in 0..n {
            let at = if transpose { k * n + row } else { row * n + k };
            let image = bank[at];
            if image == ABSENT {
                continue;
            }
            let count = &mut sizes[image as usize];
            if *count < 2 {
                keys[image as usize][*count] = k;
            }
            *count += 1;
        }
        for (&size, &[a, b]) in sizes.iter().zip(&keys) {
            if size == 2 {
                result[row * n + a] = b as u32;
                result[row * n + b] = a as u32;
            }
        }
    }
    result
}

/// The optional partner-value control accepts a single answer or two
/// agreeing answers. The chart-only model uses partners to find its order.
fn partner_value(n: usize, bank: &[u32], repaired: &[Vec<u32>; 2], at: usize) -> u32 {
    let (r, k) = (at / n, at % n);
    let row = repaired[0][at];
    let col = repaired[1][k * n + r];
    let x = if row == ABSENT {
        ABSENT
    } else {
        bank[r * n + row as usize]
    };
    let y = if col == ABSENT {
        ABSENT
    } else {
        bank[col as usize * n + k]
    };
    match (x, y) {
        (ABSENT, v) | (v, ABSENT) => v,
        (a, b) if a == b => a,
        _ => ABSENT,
    }
}

pub fn extend(
    n: usize,
    bank: &[u32],
    priority: &[usize],
    constant: &[u32],
    transport: &mut [u32],
    graphs: &mut [TripleGraph; 2],
    mode: FiberRead,
) {
    // A chart-only head requires an inverse at every nonzero coordinate.
    // A composite capacity always abstains, independent of the learned bank.
    if mode == FiberRead::Chart && !chart::admissible(n) {
        return;
    }
    let open: Vec<_> = (0..bank.len())
        .filter(|&i| {
            (mode != FiberRead::PartnerValues || bank[i] == ABSENT)
                && constant[i] == ABSENT
                && transport[i] == ABSENT
        })
        .collect();
    if open.is_empty() {
        return;
    }
    let mates = [partners(n, bank, false), partners(n, bank, true)];
    let repaired: [_; 2] =
        std::array::from_fn(|i| graphs[i].complete(n, &mates[i], priority).transport);
    let chart = (mode != FiberRead::PartnerValues)
        .then(|| chart::infer(priority, bank, &mates, &repaired))
        .flatten();
    for at in open {
        let partner = if mode != FiberRead::Chart && bank[at] == ABSENT {
            partner_value(n, bank, &repaired, at)
        } else {
            ABSENT
        };
        transport[at] = if partner == ABSENT {
            chart.as_ref().map_or(ABSENT, |values| values[at])
        } else {
            partner
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_dense_bank_discovers_the_same_values_after_renaming_every_prototype() {
        let n = 17usize;
        let identity: Vec<_> = (0..n).collect();
        let renamed = [3, 7, 0, 10, 1, 16, 9, 14, 11, 2, 5, 4, 13, 15, 6, 8, 12];
        for names in [&identity[..], &renamed] {
            let mut bank = vec![ABSENT; n * n];
            let mut output = bank.clone();
            let constant = bank.clone();
            let image = |r: usize, k: usize| names[(k * k + r * k + r.pow(3) + 2 * r) % n] as u32;
            for r in 0..n {
                for k in 0..n {
                    if r < 2 || !(k + 2 * r).is_multiple_of(5) {
                        bank[names[r] * n + names[k]] = image(r, k);
                    }
                }
            }
            extend(
                n,
                &bank,
                names,
                &constant,
                &mut output,
                &mut Default::default(),
                FiberRead::Chart,
            );
            for r in 0..n {
                for k in 0..n {
                    assert_eq!(output[names[r] * n + names[k]], image(r, k));
                }
            }
        }
    }

    #[test]
    fn only_pairs_have_observed_partners() {
        let n = 6;
        let mut bank = vec![ABSENT; n * n];
        bank[..n].copy_from_slice(&[4, 2, 2, 4, 2, ABSENT]);
        assert_eq!(
            &partners(n, &bank, false)[..n],
            &[3, ABSENT, ABSENT, 0, ABSENT, ABSENT]
        );
    }

    #[test]
    fn another_row_closes_a_pair_for_the_missing_observation() {
        let n = 6;
        let mut bank = vec![ABSENT; n * n];
        bank[..6].copy_from_slice(&[0, 1, ABSENT, 0, 1, 2]);
        bank[6..12].copy_from_slice(&[0, 1, 2, 0, 1, 2]);
        let priority = [0, 1, 2, 3, 4, 5];
        let constant = vec![ABSENT; n * n];
        let mut output = constant.clone();
        super::super::fill_from_fibers(
            n,
            &bank,
            &priority,
            &constant,
            &mut output,
            &mut Default::default(),
            super::super::FiberRead::PartnerValues,
        );
        assert_eq!(output[2], 2);
        for (mode, expected) in [(FiberRead::PartnerThenChart, 2), (FiberRead::Chart, ABSENT)] {
            let mut only = constant.clone();
            extend(
                n,
                &bank,
                &priority,
                &constant,
                &mut only,
                &mut Default::default(),
                mode,
            );
            assert_eq!(only[2], expected);
        }
        output[2] = 5;
        extend(
            n,
            &bank,
            &priority,
            &constant,
            &mut output,
            &mut Default::default(),
            FiberRead::PartnerValues,
        );
        assert_eq!(output[2], 5);
    }
}

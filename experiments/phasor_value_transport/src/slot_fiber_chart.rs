//! Discover coordinates by walking a composition of two completed hidden
//! partner maps. Every operation on slot names follows a learned permutation.
use super::ABSENT;

pub(super) fn admissible(n: usize) -> bool {
    n >= 2 && (2..=n.isqrt()).all(|factor| !n.is_multiple_of(factor))
}

struct Chart {
    order: Vec<usize>,
    position: Vec<usize>,
    inverse: Vec<usize>,
}

fn completed_involution(observed: &[u32], repaired: &[u32]) -> Option<Vec<usize>> {
    let n = observed.len();
    let mut used = vec![false; n];
    let mut map = vec![n; n];
    let mut missing = None;
    for (i, (&old, &proposal)) in observed.iter().zip(repaired).enumerate() {
        let j = if old == ABSENT { proposal } else { old };
        if j == ABSENT {
            if missing.replace(i).is_some() {
                return None;
            }
        } else {
            let j = j as usize;
            if used[j] {
                return None;
            }
            used[j] = true;
            map[i] = j;
        }
    }
    if let Some(i) = missing {
        map[i] = used.iter().position(|&v| !v)?;
    }
    (0..n).all(|i| map[map[i]] == i).then_some(map)
}

impl Chart {
    fn from_pair(origin: usize, first: &[usize], second: &[usize]) -> Option<Self> {
        let n = first.len();
        let mut order = Vec::with_capacity(n);
        let mut position = vec![n; n];
        let mut key = origin;
        while position[key] == n {
            position[key] = order.len();
            order.push(key);
            key = second[first[key]];
        }
        if key != origin || order.len() != n || n < 2 {
            return None;
        }
        let mut inverse = vec![0; n];
        for (i, v) in inverse.iter_mut().enumerate().skip(1) {
            *v = (1..n).find(|&j| i * j % n == 1)?;
        }
        Some(Self {
            order,
            position,
            inverse,
        })
    }

    fn learn(priority: &[usize], observed: &[u32], repaired: &[u32]) -> Option<Self> {
        let n = priority.len();
        let mut maps: Vec<Vec<usize>> = Vec::new();
        for &r in priority {
            let range = r * n..(r + 1) * n;
            let Some(next) = completed_involution(&observed[range.clone()], &repaired[range])
            else {
                continue;
            };
            if maps.contains(&next) {
                continue;
            }
            for earlier in &maps {
                if let Some(chart) = Self::from_pair(priority[0], earlier, &next) {
                    return Some(chart);
                }
            }
            maps.push(next);
            if maps.len() == 4 {
                break;
            }
        }
        None
    }

    /// Newton interpolation. At each round select an observed mismatch.
    /// Accept only after checking all observations, with more than twice as
    /// many observed values as coefficients and at most eight coefficients.
    fn row(&self, values: &[u32]) -> Option<Vec<u32>> {
        let n = self.order.len();
        let samples: Vec<_> = self
            .order
            .iter()
            .enumerate()
            .filter_map(|(x, &key)| {
                let v = values[key];
                (v != ABSENT).then(|| (x, self.position[v as usize]))
            })
            .collect();
        let limit = samples.len().saturating_sub(1) / 2;
        if limit == 0 {
            return None;
        }
        let mut prediction = vec![0; n];
        let mut basis = vec![1; n];
        let maximum = limit.min(8);
        for _ in 0..maximum {
            let Some(&(x, y)) = samples.iter().find(|&&(x, y)| prediction[x] != y) else {
                break;
            };
            let den = basis[x];
            if den == 0 {
                return None;
            }
            let delta = (y + n - prediction[x]) * self.inverse[den] % n;
            for t in 0..n {
                prediction[t] = (prediction[t] + delta * basis[t]) % n;
                basis[t] = basis[t] * (t + n - x) % n;
            }
        }
        if !samples.iter().all(|&(x, y)| prediction[x] == y) {
            return None;
        }
        Some(
            self.position
                .iter()
                .map(|&x| self.order[prediction[x]] as u32)
                .collect(),
        )
    }

    fn proposals(&self, bank: &[u32]) -> Vec<u32> {
        let n = self.order.len();
        let mut result = vec![ABSENT; n * n];
        for transpose in [false, true] {
            for r in 0..n {
                let index = |k| if transpose { k * n + r } else { r * n + k };
                let values: Vec<_> = (0..n).map(|k| bank[index(k)]).collect();
                let Some(predictions) = self.row(&values) else {
                    continue;
                };
                for (k, &v) in predictions.iter().enumerate() {
                    let i = index(k);
                    // A validated fit has enough distinct witnesses left when
                    // this query is erased. It can train the shared read gate
                    // on an address with a direct value as well.
                    result[i] = if result[i] == ABSENT || result[i] == v {
                        v
                    } else {
                        ABSENT
                    };
                }
            }
        }
        result
    }
}

pub(super) fn infer(
    priority: &[usize],
    bank: &[u32],
    mates: &[Vec<u32>; 2],
    repaired: &[Vec<u32>; 2],
) -> Option<Vec<u32>> {
    mates.iter().zip(repaired).find_map(|(raw, complete)| {
        Chart::learn(priority, raw, complete).map(|chart| chart.proposals(bank))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chart_capacity_agrees_with_all_coordinate_inverses() {
        for n in 0..=256usize {
            let complete = n >= 2 && (1..n).all(|i| (1..n).any(|j| i * j % n == 1));
            assert_eq!(admissible(n), complete, "dictionary capacity {n}");
        }
    }

    #[test]
    fn orbit_interpolation_transports_polynomial_outputs_through_an_arbitrary_renaming() {
        let n = 17;
        let names = [3, 7, 0, 10, 1, 16, 9, 14, 11, 2, 5, 4, 13, 15, 6, 8, 12];
        let mut first = vec![0; n];
        let mut second = first.clone();
        let mut row = vec![ABSENT; n];
        let value = |x: usize| (x.pow(4) + 3 * x + 1) % n;
        for x in 0..n {
            first[names[x]] = names[(n - x) % n];
            second[names[x]] = names[(1 + n - x) % n];
            if x % 3 != 0 {
                row[names[x]] = names[value(x)] as u32;
            }
        }
        let chart = Chart::from_pair(names[2], &first, &second).unwrap();
        let got = chart.row(&row).unwrap();
        for x in 0..n {
            assert_eq!(got[names[x]], names[value(x)] as u32);
        }
        row[names[8]] = names[(value(8) + 1) % n] as u32;
        assert!(chart.row(&row).is_none());
        assert!(Chart::from_pair(names[0], &first, &first).is_none());
    }

    #[test]
    fn supported_predictions_at_observed_addresses_recover_after_erasing_the_query() {
        let n = 17;
        let left: Vec<_> = (0..n).map(|x| (n - x) % n).collect();
        let right: Vec<_> = (0..n).map(|x| (1 + n - x) % n).collect();
        let chart = Chart::from_pair(3, &left, &right).unwrap();
        let mut bank = vec![ABSENT; n * n];
        for (x, v) in bank[..11].iter_mut().enumerate() {
            *v = ((x * x + 3 * x + 4) % n) as u32;
        }
        let complete = chart.proposals(&bank);
        for x in 0..n {
            let mut erased = bank.clone();
            erased[x] = ABSENT;
            assert_eq!(complete[x], ((x * x + 3 * x + 4) % n) as u32);
            assert_eq!(chart.proposals(&erased)[x], complete[x]);
        }
    }

    #[test]
    fn completing_a_permutation_requires_unique_images_and_at_most_one_hole() {
        assert_eq!(
            completed_involution(&[3, ABSENT, ABSENT, 0], &[ABSENT, 2, ABSENT, ABSENT]),
            Some(vec![3, 2, 1, 0])
        );
        assert!(completed_involution(&[3, ABSENT, ABSENT, 0], &[ABSENT; 4]).is_none());
        assert!(completed_involution(&[1, 2, 0], &[ABSENT; 3]).is_none());
        assert!(completed_involution(&[1, 0, 0], &[ABSENT; 3]).is_none());
    }
}

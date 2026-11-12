//! Minimum-cost square assignment, using integral shortest augmenting paths.
//! This is one exact coordinate update, not a global decoder-search oracle.
use crate::Result;

pub fn minimize(cost: &[usize], n: usize) -> Result<Vec<usize>> {
    if !(2..=193).contains(&n) || cost.len() != n * n || cost.iter().any(|&x| x > n * n) {
        return Err("invalid bounded assignment matrix".into());
    }
    let (mut u, mut v, mut p, mut way) = (
        vec![0i64; n + 1],
        vec![0i64; n + 1],
        vec![0usize; n + 1],
        vec![0usize; n + 1],
    );
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut min = vec![i64::MAX / 4; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = i64::MAX / 4;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let c = cost[(i0 - 1) * n + j - 1] as i64 - u[i0] - v[j];
                    if c < min[j] {
                        min[j] = c;
                        way[j] = j0;
                    }
                    if min[j] < delta {
                        delta = min[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut result = vec![0; n];
    for j in 1..=n {
        result[p[j] - 1] = j - 1;
    }
    Ok(result)
}

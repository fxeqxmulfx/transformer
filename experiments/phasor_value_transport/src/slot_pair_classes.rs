//! Complete missing reads by matching ordered pairs of hidden slots in
//! separate key, context and output roles. See PairClasses.row_candidate.
use super::pair_index::ImageIndex;
use super::{ABSENT, Components, values::Values};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairRead {
    Unique,
    FirstOfTwo,
}

impl PairRead {
    fn select(self, options: Values, rank: &[usize]) -> Option<u32> {
        if self == Self::Unique {
            return options.unique();
        }
        if options.len() > 2 {
            return None;
        }
        options.members().min_by_key(|&i| rank[i]).map(|i| i as u32)
    }
}

type Observations = Vec<Vec<(usize, usize)>>;

fn representatives(n: usize, rows: &Observations, columns: &Observations) -> Vec<u32> {
    let mut classes = Components::new(3 * n * n);
    for (axis, lists) in [rows, columns].into_iter().enumerate() {
        for anchors in lists {
            for &(a, u) in anchors {
                let source = (axis * n + a) * n;
                let target = (2 * n + u) * n;
                for &(b, v) in anchors {
                    if a != b {
                        classes.join(source + b, target + v);
                    }
                }
            }
        }
    }
    classes.flatten();
    classes.parent
}

pub fn extend(
    n: usize,
    bank: &[u32],
    priority: &[usize],
    constant: &[u32],
    output: &mut [u32],
    read: PairRead,
) {
    let mut rows = vec![Vec::new(); n];
    let mut columns = vec![Vec::new(); n];
    let mut observed = Values::default();
    for (cell, &stored) in bank.iter().enumerate() {
        let image = if stored != ABSENT {
            stored
        } else {
            constant[cell].min(output[cell])
        };
        if image != ABSENT {
            let (r, k) = (cell / n, cell % n);
            rows[r].push((k, image as usize));
            columns[k].push((r, image as usize));
            observed.insert(image as usize);
        }
    }
    if observed.len() < 3 {
        return;
    }
    let open = |cell: usize| {
        bank[cell] == ABSENT
            && constant[cell] == ABSENT
            && output[cell] == ABSENT
            && !rows[cell / n].is_empty()
            && !columns[cell % n].is_empty()
    };
    if !(0..bank.len()).any(open) {
        return;
    }
    let classes = representatives(n, &rows, &columns);
    let images = ImageIndex::new(n, &classes, observed);
    let mut rank = vec![0; n];
    for (i, &v) in priority.iter().enumerate() {
        rank[v] = i;
    }
    for (cell, destination) in output.iter_mut().enumerate() {
        if *destination != ABSENT || bank[cell] != ABSENT || constant[cell] != ABSENT {
            continue;
        }
        let (r, k) = (cell / n, cell % n);
        if rows[r].is_empty() || columns[k].is_empty() {
            continue;
        }
        let mut options = observed;
        let mut support = 0;
        'constraints: for (axis, (query, anchors)) in
            [(k, &rows[r]), (r, &columns[k])].into_iter().enumerate()
        {
            for &(key, u) in anchors {
                if key == query {
                    continue;
                }
                let root = classes[(axis * n + key) * n + query];
                if let Some(allowed) = images.get(u, root)
                    && allowed != observed
                {
                    options.intersect(allowed);
                    support += 1;
                    if options.is_empty() {
                        break 'constraints;
                    }
                }
            }
        }
        if support >= 3
            && let Some(y) = read.select(options, &rank)
        {
            *destination = y;
        }
    }
}

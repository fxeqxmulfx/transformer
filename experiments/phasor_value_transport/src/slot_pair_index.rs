//! Rank a present image entry among bitmap words and store only its nonempty mask.
use super::values::Values;

pub(super) struct ImageIndex {
    classes: Vec<u32>,
    words_per_image: usize,
    present: Vec<u64>,
    starts: Vec<u32>,
    values: Vec<Values>,
}

impl ImageIndex {
    pub fn new(n: usize, representatives: &[u32], images: Values) -> Self {
        let active: Vec<_> = images.members().collect();
        let pairs = &representatives[2 * n * n..];
        let mut classes = vec![u32::MAX; representatives.len()];
        let mut count = 0_u32;
        for &u in &active {
            for &v in &active {
                let id = &mut classes[pairs[u * n + v] as usize];
                if *id == u32::MAX {
                    *id = count;
                    count += 1;
                }
            }
        }
        let stride = (count as usize).div_ceil(64);
        let mut present = vec![0_u64; n * stride];
        for &u in &active {
            for &v in &active {
                let id = classes[pairs[u * n + v] as usize] as usize;
                present[u * stride + id / 64] |= 1_u64 << (id % 64);
            }
        }
        let mut size = 0;
        let starts = present
            .iter()
            .map(|word| {
                let start = size;
                size += word.count_ones();
                start
            })
            .collect();
        let mut index = Self {
            classes,
            words_per_image: stride,
            present,
            starts,
            values: vec![Values::default(); size as usize],
        };
        for &u in &active {
            for &v in &active {
                let at = index
                    .position(u, pairs[u * n + v])
                    .expect("indexed an observed output pair");
                index.values[at].insert(v);
            }
        }
        index
    }

    fn position(&self, image: usize, root: u32) -> Option<usize> {
        let id = self.classes[root as usize];
        if id == u32::MAX {
            return None;
        }
        let id = id as usize;
        let block = image * self.words_per_image + id / 64;
        let mask = 1_u64 << (id % 64);
        let present = self.present[block];
        if present & mask == 0 {
            return None;
        }
        Some((self.starts[block] + (present & (mask - 1)).count_ones()) as usize)
    }

    pub fn get(&self, image: usize, root: u32) -> Option<Values> {
        self.position(image, root).map(|i| self.values[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn bitmap_ranks_recover_masks_across_words_and_missing_classes() {
        for n in [1, 7, 64, 97, 193, 256] {
            let mut observed = Values::default();
            let sizes = [0, n / 3, n];
            for count in sizes {
                for u in 0..count {
                    observed.insert((u * 5) % n);
                }
                let mut state = 17_u32;
                let mut next = || {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    state as usize
                };
                let entries = 3 * n * n;
                let roots: Vec<u32> = (0..entries).map(|_| (next() % entries) as u32).collect();
                let index = ImageIndex::new(n, &roots, observed);
                let mut masks: HashMap<(usize, u32), Values> = HashMap::new();
                for a in observed.members() {
                    for b in observed.members() {
                        masks
                            .entry((a, roots[2 * n * n + a * n + b]))
                            .or_default()
                            .insert(b);
                    }
                }
                for (&(a, root), &mask) in &masks {
                    assert!(index.get(a, root) == Some(mask));
                }
                for _ in 0..5_000 {
                    let k = (next() % n, (next() % entries) as u32);
                    assert!(index.get(k.0, k.1) == masks.get(&k).copied());
                }
            }
        }
    }
}
